-- 3db Unified Database Ecosystem - PostgreSQL Schema
-- This script sets up the core PostgreSQL tables for the unified database system

-- =====================================================================================
-- EXTENSION SETUP
-- =====================================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgvector for vector operations (if using same DB instance)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable Apache AGE for graph operations (if using same DB instance)
CREATE EXTENSION IF NOT EXISTS age;

-- =====================================================================================
-- CORE ENTITY TABLES
-- =====================================================================================

-- Generic entities table for unified entity tracking
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    name VARCHAR(500),
    description TEXT,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1
);

-- Users table (example entity type)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100) DEFAULT 'user',
    name VARCHAR(200) NOT NULL,
    email VARCHAR(255) UNIQUE,
    bio TEXT,
    interests TEXT,
    preferences JSONB DEFAULT '{}',
    profile_data JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1
);

-- Documents table (example entity type)
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100) DEFAULT 'document',
    title VARCHAR(500) NOT NULL,
    content TEXT,
    author_id VARCHAR(255),
    category VARCHAR(100),
    tags TEXT[],
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'published',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1
);

-- Products table (example entity type)
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100) DEFAULT 'product',
    name VARCHAR(300) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    price DECIMAL(10,2),
    currency VARCHAR(3) DEFAULT 'USD',
    availability_status VARCHAR(50) DEFAULT 'available',
    attributes JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1
);

-- =====================================================================================
-- SYNCHRONIZATION METADATA TABLES
-- =====================================================================================

-- Entity synchronization metadata (managed by sync system)
CREATE TABLE IF NOT EXISTS entity_sync_metadata (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    
    -- Database presence flags
    in_postgresql BOOLEAN DEFAULT FALSE,
    in_vector BOOLEAN DEFAULT FALSE,
    in_graph BOOLEAN DEFAULT FALSE,
    
    -- Last synchronization timestamps
    last_sync_postgresql TIMESTAMP WITH TIME ZONE,
    last_sync_vector TIMESTAMP WITH TIME ZONE,
    last_sync_graph TIMESTAMP WITH TIME ZONE,
    
    -- Synchronization status
    sync_status JSONB DEFAULT '{}',
    sync_errors JSONB DEFAULT '[]',
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'
);

-- Synchronization log for tracking sync operations
CREATE TABLE IF NOT EXISTS sync_operations_log (
    id SERIAL PRIMARY KEY,
    operation_id UUID DEFAULT uuid_generate_v4(),
    entity_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    operation_type VARCHAR(50) NOT NULL, -- create, update, delete, sync
    source_database VARCHAR(50) NOT NULL,
    target_databases TEXT[] DEFAULT '{}',
    status VARCHAR(50) NOT NULL, -- pending, success, failed, partial
    error_message TEXT,
    execution_time_ms FLOAT,
    data_payload JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================================================
-- VECTOR DATABASE TABLES (if using same PostgreSQL instance)
-- =====================================================================================

-- Embeddings table for pgvector
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    embedding vector(384) NOT NULL, -- Default dimension, adjust as needed
    metadata JSONB DEFAULT '{}',
    text_content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================================================
-- GRAPH DATABASE METADATA TABLES (for Apache AGE integration)
-- =====================================================================================

-- Graph nodes metadata
CREATE TABLE IF NOT EXISTS graph_nodes_metadata (
    id SERIAL PRIMARY KEY,
    node_id BIGINT UNIQUE NOT NULL,
    entity_id VARCHAR(255) UNIQUE NOT NULL,
    label VARCHAR(100) NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Graph edges metadata
CREATE TABLE IF NOT EXISTS graph_edges_metadata (
    id SERIAL PRIMARY KEY,
    edge_id BIGINT UNIQUE NOT NULL,
    relationship_id VARCHAR(255) UNIQUE NOT NULL,
    from_entity_id VARCHAR(255) NOT NULL,
    to_entity_id VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================================================

-- Core entity indexes
CREATE INDEX IF NOT EXISTS entities_entity_id_idx ON entities (entity_id);
CREATE INDEX IF NOT EXISTS entities_entity_type_idx ON entities (entity_type);
CREATE INDEX IF NOT EXISTS entities_created_at_idx ON entities (created_at);
CREATE INDEX IF NOT EXISTS entities_status_idx ON entities (status);

-- User indexes
CREATE INDEX IF NOT EXISTS users_entity_id_idx ON users (entity_id);
CREATE INDEX IF NOT EXISTS users_email_idx ON users (email);
CREATE INDEX IF NOT EXISTS users_status_idx ON users (status);

-- Document indexes
CREATE INDEX IF NOT EXISTS documents_entity_id_idx ON documents (entity_id);
CREATE INDEX IF NOT EXISTS documents_author_id_idx ON documents (author_id);
CREATE INDEX IF NOT EXISTS documents_category_idx ON documents (category);
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);
CREATE INDEX IF NOT EXISTS documents_tags_gin_idx ON documents USING GIN (tags);

-- Product indexes
CREATE INDEX IF NOT EXISTS products_entity_id_idx ON products (entity_id);
CREATE INDEX IF NOT EXISTS products_category_idx ON products (category);
CREATE INDEX IF NOT EXISTS products_status_idx ON products (status);
CREATE INDEX IF NOT EXISTS products_price_idx ON products (price);

-- Sync metadata indexes
CREATE INDEX IF NOT EXISTS entity_sync_metadata_entity_id_idx ON entity_sync_metadata (entity_id);
CREATE INDEX IF NOT EXISTS entity_sync_metadata_entity_type_idx ON entity_sync_metadata (entity_type);
CREATE INDEX IF NOT EXISTS entity_sync_metadata_updated_at_idx ON entity_sync_metadata (updated_at);

-- Sync log indexes
CREATE INDEX IF NOT EXISTS sync_operations_log_entity_id_idx ON sync_operations_log (entity_id);
CREATE INDEX IF NOT EXISTS sync_operations_log_operation_type_idx ON sync_operations_log (operation_type);
CREATE INDEX IF NOT EXISTS sync_operations_log_status_idx ON sync_operations_log (status);
CREATE INDEX IF NOT EXISTS sync_operations_log_created_at_idx ON sync_operations_log (created_at);

-- Vector embeddings indexes (if using same DB)
CREATE INDEX IF NOT EXISTS embeddings_entity_id_idx ON embeddings (entity_id);
CREATE INDEX IF NOT EXISTS embeddings_entity_type_idx ON embeddings (entity_type);
-- Vector similarity indexes (using ivfflat)
CREATE INDEX IF NOT EXISTS embeddings_cosine_idx ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS embeddings_l2_idx ON embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

-- Graph metadata indexes
CREATE INDEX IF NOT EXISTS graph_nodes_entity_id_idx ON graph_nodes_metadata (entity_id);
CREATE INDEX IF NOT EXISTS graph_nodes_label_idx ON graph_nodes_metadata (label);
CREATE INDEX IF NOT EXISTS graph_edges_relationship_type_idx ON graph_edges_metadata (relationship_type);
CREATE INDEX IF NOT EXISTS graph_edges_from_entity_idx ON graph_edges_metadata (from_entity_id);
CREATE INDEX IF NOT EXISTS graph_edges_to_entity_idx ON graph_edges_metadata (to_entity_id);

-- =====================================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- =====================================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to all main tables
CREATE TRIGGER update_entities_updated_at 
    BEFORE UPDATE ON entities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at 
    BEFORE UPDATE ON products 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_entity_sync_metadata_updated_at 
    BEFORE UPDATE ON entity_sync_metadata 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_embeddings_updated_at 
    BEFORE UPDATE ON embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_graph_nodes_metadata_updated_at 
    BEFORE UPDATE ON graph_nodes_metadata 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_graph_edges_metadata_updated_at 
    BEFORE UPDATE ON graph_edges_metadata 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================================================
-- VIEWS FOR UNIFIED QUERIES
-- =====================================================================================

-- Unified entity view combining data from multiple sources
CREATE OR REPLACE VIEW unified_entities AS
SELECT 
    e.entity_id,
    e.entity_type,
    e.name,
    e.description,
    e.metadata,
    e.status,
    e.created_at,
    e.updated_at,
    e.version,
    
    -- Sync metadata
    sm.in_postgresql,
    sm.in_vector,
    sm.in_graph,
    sm.last_sync_postgresql,
    sm.last_sync_vector,
    sm.last_sync_graph,
    
    -- Embedding info
    CASE WHEN em.entity_id IS NOT NULL THEN true ELSE false END as has_embedding,
    em.created_at as embedding_created_at,
    
    -- Graph info
    CASE WHEN gn.entity_id IS NOT NULL THEN true ELSE false END as has_graph_node,
    gn.label as graph_label,
    gn.created_at as graph_node_created_at

FROM entities e
LEFT JOIN entity_sync_metadata sm ON e.entity_id = sm.entity_id
LEFT JOIN embeddings em ON e.entity_id = em.entity_id
LEFT JOIN graph_nodes_metadata gn ON e.entity_id = gn.entity_id;

-- Sync status summary view
CREATE OR REPLACE VIEW sync_status_summary AS
SELECT 
    entity_type,
    COUNT(*) as total_entities,
    COUNT(*) FILTER (WHERE in_postgresql = true) as in_postgresql,
    COUNT(*) FILTER (WHERE in_vector = true) as in_vector,
    COUNT(*) FILTER (WHERE in_graph = true) as in_graph,
    COUNT(*) FILTER (WHERE in_postgresql = true AND in_vector = true AND in_graph = true) as fully_synced,
    COUNT(*) FILTER (WHERE sync_errors != '[]') as entities_with_errors,
    AVG(version) as avg_version
FROM entity_sync_metadata
GROUP BY entity_type;

-- =====================================================================================
-- STORED PROCEDURES FOR COMMON OPERATIONS
-- =====================================================================================

-- Function to create a unified entity with automatic sync metadata
CREATE OR REPLACE FUNCTION create_unified_entity(
    p_entity_type VARCHAR,
    p_entity_data JSONB
) RETURNS VARCHAR AS $$
DECLARE
    v_entity_id VARCHAR;
    v_name VARCHAR;
    v_description TEXT;
BEGIN
    -- Generate entity ID if not provided
    v_entity_id := COALESCE(p_entity_data->>'entity_id', p_entity_type || '_' || extract(epoch from now()) * 1000);
    v_name := p_entity_data->>'name';
    v_description := p_entity_data->>'description';
    
    -- Insert into entities table
    INSERT INTO entities (entity_id, entity_type, name, description, metadata)
    VALUES (v_entity_id, p_entity_type, v_name, v_description, p_entity_data)
    ON CONFLICT (entity_id) DO UPDATE SET
        name = EXCLUDED.name,
        description = EXCLUDED.description,
        metadata = EXCLUDED.metadata,
        updated_at = CURRENT_TIMESTAMP,
        version = entities.version + 1;
    
    -- Insert into sync metadata
    INSERT INTO entity_sync_metadata (entity_id, entity_type, in_postgresql)
    VALUES (v_entity_id, p_entity_type, true)
    ON CONFLICT (entity_id) DO UPDATE SET
        in_postgresql = true,
        last_sync_postgresql = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP;
    
    RETURN v_entity_id;
END;
$$ LANGUAGE plpgsql;

-- Function to mark entity as synced to a specific database
CREATE OR REPLACE FUNCTION mark_entity_synced(
    p_entity_id VARCHAR,
    p_database_type VARCHAR
) RETURNS BOOLEAN AS $$
BEGIN
    IF p_database_type = 'postgresql' THEN
        UPDATE entity_sync_metadata 
        SET in_postgresql = true, 
            last_sync_postgresql = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE entity_id = p_entity_id;
    ELSIF p_database_type = 'vector' THEN
        UPDATE entity_sync_metadata 
        SET in_vector = true, 
            last_sync_vector = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE entity_id = p_entity_id;
    ELSIF p_database_type = 'graph' THEN
        UPDATE entity_sync_metadata 
        SET in_graph = true, 
            last_sync_graph = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE entity_id = p_entity_id;
    ELSE
        RETURN false;
    END IF;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================================
-- SAMPLE DATA (OPTIONAL - FOR TESTING)
-- =====================================================================================

-- Sample users
INSERT INTO users (entity_id, name, email, bio, interests) VALUES
('user_demo_1', 'Alice Johnson', 'alice@example.com', 'AI researcher and data scientist', 'machine learning, neural networks, data analysis'),
('user_demo_2', 'Bob Smith', 'bob@example.com', 'Software engineer specializing in databases', 'databases, distributed systems, cloud computing'),
('user_demo_3', 'Carol Davis', 'carol@example.com', 'Product manager for tech startups', 'product management, user experience, agile development')
ON CONFLICT (entity_id) DO NOTHING;

-- Sample documents
INSERT INTO documents (entity_id, title, content, author_id, category, tags) VALUES
('doc_demo_1', 'Introduction to Vector Databases', 'Vector databases are a new paradigm for storing and querying high-dimensional data...', 'user_demo_1', 'technology', ARRAY['databases', 'vectors', 'AI']),
('doc_demo_2', 'Graph Database Use Cases', 'Graph databases excel at modeling and querying connected data...', 'user_demo_2', 'technology', ARRAY['graphs', 'databases', 'relationships']),
('doc_demo_3', 'Building Unified Database Systems', 'Modern applications require multiple data models working together...', 'user_demo_1', 'technology', ARRAY['databases', 'architecture', 'systems'])
ON CONFLICT (entity_id) DO NOTHING;

-- Sample products
INSERT INTO products (entity_id, name, description, category, price) VALUES
('product_demo_1', 'Database Optimization Toolkit', 'Professional tools for optimizing database performance', 'software', 299.99),
('product_demo_2', 'Vector Search Engine', 'High-performance vector similarity search solution', 'software', 499.99),
('product_demo_3', 'Graph Analytics Platform', 'Comprehensive platform for graph data analysis', 'software', 799.99)
ON CONFLICT (entity_id) DO NOTHING;

-- Initialize sync metadata for sample data
INSERT INTO entity_sync_metadata (entity_id, entity_type, in_postgresql) 
SELECT entity_id, entity_type, true FROM entities
ON CONFLICT (entity_id) DO NOTHING;

-- =====================================================================================
-- CLEANUP AND MAINTENANCE FUNCTIONS
-- =====================================================================================

-- Function to clean up old sync logs
CREATE OR REPLACE FUNCTION cleanup_old_sync_logs(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sync_operations_log 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_3db_statistics()
RETURNS TABLE (
    table_name TEXT,
    row_count BIGINT,
    last_updated TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'entities'::TEXT,
        COUNT(*)::BIGINT,
        MAX(updated_at)
    FROM entities
    UNION ALL
    SELECT 
        'users'::TEXT,
        COUNT(*)::BIGINT,
        MAX(updated_at)
    FROM users
    UNION ALL
    SELECT 
        'documents'::TEXT,
        COUNT(*)::BIGINT,
        MAX(updated_at)
    FROM documents
    UNION ALL
    SELECT 
        'products'::TEXT,
        COUNT(*)::BIGINT,
        MAX(updated_at)
    FROM products
    UNION ALL
    SELECT 
        'embeddings'::TEXT,
        COUNT(*)::BIGINT,
        MAX(updated_at)
    FROM embeddings
    UNION ALL
    SELECT 
        'entity_sync_metadata'::TEXT,
        COUNT(*)::BIGINT,
        MAX(updated_at)
    FROM entity_sync_metadata;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your environment)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO your_application_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_application_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO your_application_user;

-- Final message
DO $$
BEGIN
    RAISE NOTICE '🎉 3db PostgreSQL schema setup completed successfully!';
    RAISE NOTICE 'Tables created: entities, users, documents, products, embeddings, sync metadata';
    RAISE NOTICE 'Indexes, triggers, views, and functions ready for use.';
    RAISE NOTICE 'Sample data inserted for testing (use for development only).';
END $$;
