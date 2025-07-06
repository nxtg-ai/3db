-- 3db Unified Database Ecosystem - Apache AGE Graph Schema
-- This script sets up the Apache AGE graph database for the unified system

-- =====================================================================================
-- APACHE AGE EXTENSION SETUP
-- =====================================================================================

-- Load AGE extension
CREATE EXTENSION IF NOT EXISTS age;

-- Load AGE into search path
LOAD 'age';

-- Set search path to include AGE
SET search_path = ag_catalog, "$user", public;

-- =====================================================================================
-- GRAPH CREATION
-- =====================================================================================

-- Create the main unified graph
SELECT create_graph('unified_graph');

-- Create additional specialized graphs (optional)
SELECT create_graph('user_relationships');
SELECT create_graph('content_graph');
SELECT create_graph('product_recommendations');

-- =====================================================================================
-- GRAPH SCHEMA DEFINITION (CYPHER QUERIES)
-- =====================================================================================

-- Create sample node labels and relationships
-- This section uses Cypher queries through AGE SQL syntax

-- Create User nodes with properties
SELECT * FROM cypher('unified_graph', $$
    // Sample user nodes
    CREATE (u1:User {
        entity_id: 'user_demo_1',
        name: 'Alice Johnson',
        email: 'alice@example.com',
        bio: 'AI researcher and data scientist',
        interests: ['machine learning', 'neural networks', 'data analysis'],
        created_at: timestamp()
    })
    CREATE (u2:User {
        entity_id: 'user_demo_2',
        name: 'Bob Smith',
        email: 'bob@example.com',
        bio: 'Software engineer specializing in databases',
        interests: ['databases', 'distributed systems', 'cloud computing'],
        created_at: timestamp()
    })
    CREATE (u3:User {
        entity_id: 'user_demo_3',
        name: 'Carol Davis',
        email: 'carol@example.com',
        bio: 'Product manager for tech startups',
        interests: ['product management', 'user experience', 'agile development'],
        created_at: timestamp()
    })
    RETURN u1, u2, u3
$$) as (u1 agtype, u2 agtype, u3 agtype);

-- Create Document nodes
SELECT * FROM cypher('unified_graph', $$
    CREATE (d1:Document {
        entity_id: 'doc_demo_1',
        title: 'Introduction to Vector Databases',
        category: 'technology',
        tags: ['databases', 'vectors', 'AI'],
        author_id: 'user_demo_1',
        created_at: timestamp()
    })
    CREATE (d2:Document {
        entity_id: 'doc_demo_2',
        title: 'Graph Database Use Cases',
        category: 'technology',
        tags: ['graphs', 'databases', 'relationships'],
        author_id: 'user_demo_2',
        created_at: timestamp()
    })
    CREATE (d3:Document {
        entity_id: 'doc_demo_3',
        title: 'Building Unified Database Systems',
        category: 'technology',
        tags: ['databases', 'architecture', 'systems'],
        author_id: 'user_demo_1',
        created_at: timestamp()
    })
    RETURN d1, d2, d3
$$) as (d1 agtype, d2 agtype, d3 agtype);

-- Create Product nodes
SELECT * FROM cypher('unified_graph', $$
    CREATE (p1:Product {
        entity_id: 'product_demo_1',
        name: 'Database Optimization Toolkit',
        category: 'software',
        price: 299.99,
        description: 'Professional tools for optimizing database performance',
        created_at: timestamp()
    })
    CREATE (p2:Product {
        entity_id: 'product_demo_2',
        name: 'Vector Search Engine',
        category: 'software',
        price: 499.99,
        description: 'High-performance vector similarity search solution',
        created_at: timestamp()
    })
    CREATE (p3:Product {
        entity_id: 'product_demo_3',
        name: 'Graph Analytics Platform',
        category: 'software',
        price: 799.99,
        description: 'Comprehensive platform for graph data analysis',
        created_at: timestamp()
    })
    RETURN p1, p2, p3
$$) as (p1 agtype, p2 agtype, p3 agtype);

-- Create Category nodes
SELECT * FROM cypher('unified_graph', $$
    CREATE (c1:Category {
        entity_id: 'cat_technology',
        name: 'Technology',
        description: 'Technology-related content and products'
    })
    CREATE (c2:Category {
        entity_id: 'cat_software',
        name: 'Software',
        description: 'Software products and tools'
    })
    CREATE (c3:Category {
        entity_id: 'cat_ai',
        name: 'Artificial Intelligence',
        description: 'AI and machine learning related content'
    })
    RETURN c1, c2, c3
$$) as (c1 agtype, c2 agtype, c3 agtype);

-- =====================================================================================
-- RELATIONSHIPS CREATION
-- =====================================================================================

-- User relationships
SELECT * FROM cypher('unified_graph', $$
    MATCH (u1:User {entity_id: 'user_demo_1'})
    MATCH (u2:User {entity_id: 'user_demo_2'})
    MATCH (u3:User {entity_id: 'user_demo_3'})
    CREATE (u1)-[:FOLLOWS {created_at: timestamp(), strength: 0.8}]->(u2)
    CREATE (u2)-[:FOLLOWS {created_at: timestamp(), strength: 0.9}]->(u1)
    CREATE (u1)-[:COLLABORATES {created_at: timestamp(), project: 'Database Research'}]->(u3)
    CREATE (u2)-[:KNOWS {created_at: timestamp(), context: 'Professional'}]->(u3)
    RETURN u1, u2, u3
$$) as (u1 agtype, u2 agtype, u3 agtype);

-- Author relationships (User -> Document)
SELECT * FROM cypher('unified_graph', $$
    MATCH (u1:User {entity_id: 'user_demo_1'})
    MATCH (u2:User {entity_id: 'user_demo_2'})
    MATCH (d1:Document {entity_id: 'doc_demo_1'})
    MATCH (d2:Document {entity_id: 'doc_demo_2'})
    MATCH (d3:Document {entity_id: 'doc_demo_3'})
    CREATE (u1)-[:AUTHORED {created_at: timestamp()}]->(d1)
    CREATE (u2)-[:AUTHORED {created_at: timestamp()}]->(d2)
    CREATE (u1)-[:AUTHORED {created_at: timestamp()}]->(d3)
    RETURN u1, u2, d1, d2, d3
$$) as (u1 agtype, u2 agtype, d1 agtype, d2 agtype, d3 agtype);

-- Interest relationships (User -> Category)
SELECT * FROM cypher('unified_graph', $$
    MATCH (u1:User {entity_id: 'user_demo_1'})
    MATCH (u2:User {entity_id: 'user_demo_2'})
    MATCH (u3:User {entity_id: 'user_demo_3'})
    MATCH (c1:Category {entity_id: 'cat_technology'})
    MATCH (c2:Category {entity_id: 'cat_software'})
    MATCH (c3:Category {entity_id: 'cat_ai'})
    CREATE (u1)-[:INTERESTED_IN {strength: 0.9, created_at: timestamp()}]->(c3)
    CREATE (u1)-[:INTERESTED_IN {strength: 0.8, created_at: timestamp()}]->(c1)
    CREATE (u2)-[:INTERESTED_IN {strength: 0.9, created_at: timestamp()}]->(c2)
    CREATE (u2)-[:INTERESTED_IN {strength: 0.7, created_at: timestamp()}]->(c1)
    CREATE (u3)-[:INTERESTED_IN {strength: 0.6, created_at: timestamp()}]->(c1)
    RETURN u1, u2, u3, c1, c2, c3
$$) as (u1 agtype, u2 agtype, u3 agtype, c1 agtype, c2 agtype, c3 agtype);

-- Document categorization
SELECT * FROM cypher('unified_graph', $$
    MATCH (d1:Document {entity_id: 'doc_demo_1'})
    MATCH (d2:Document {entity_id: 'doc_demo_2'})
    MATCH (d3:Document {entity_id: 'doc_demo_3'})
    MATCH (c1:Category {entity_id: 'cat_technology'})
    MATCH (c3:Category {entity_id: 'cat_ai'})
    CREATE (d1)-[:BELONGS_TO {created_at: timestamp()}]->(c1)
    CREATE (d1)-[:BELONGS_TO {created_at: timestamp()}]->(c3)
    CREATE (d2)-[:BELONGS_TO {created_at: timestamp()}]->(c1)
    CREATE (d3)-[:BELONGS_TO {created_at: timestamp()}]->(c1)
    RETURN d1, d2, d3, c1, c3
$$) as (d1 agtype, d2 agtype, d3 agtype, c1 agtype, c3 agtype);

-- Product categorization
SELECT * FROM cypher('unified_graph', $$
    MATCH (p1:Product {entity_id: 'product_demo_1'})
    MATCH (p2:Product {entity_id: 'product_demo_2'})
    MATCH (p3:Product {entity_id: 'product_demo_3'})
    MATCH (c2:Category {entity_id: 'cat_software'})
    CREATE (p1)-[:BELONGS_TO {created_at: timestamp()}]->(c2)
    CREATE (p2)-[:BELONGS_TO {created_at: timestamp()}]->(c2)
    CREATE (p3)-[:BELONGS_TO {created_at: timestamp()}]->(c2)
    RETURN p1, p2, p3, c2
$$) as (p1 agtype, p2 agtype, p3 agtype, c2 agtype);

-- Similar products relationships
SELECT * FROM cypher('unified_graph', $$
    MATCH (p1:Product {entity_id: 'product_demo_1'})
    MATCH (p2:Product {entity_id: 'product_demo_2'})
    MATCH (p3:Product {entity_id: 'product_demo_3'})
    CREATE (p1)-[:SIMILAR_TO {similarity_score: 0.7, reason: 'Both are database tools'}]->(p2)
    CREATE (p2)-[:SIMILAR_TO {similarity_score: 0.6, reason: 'Both handle data analysis'}]->(p3)
    CREATE (p1)-[:SIMILAR_TO {similarity_score: 0.5, reason: 'Both are software tools'}]->(p3)
    RETURN p1, p2, p3
$$) as (p1 agtype, p2 agtype, p3 agtype);

-- User reading/viewing relationships
SELECT * FROM cypher('unified_graph', $$
    MATCH (u1:User {entity_id: 'user_demo_1'})
    MATCH (u2:User {entity_id: 'user_demo_2'})
    MATCH (u3:User {entity_id: 'user_demo_3'})
    MATCH (d1:Document {entity_id: 'doc_demo_1'})
    MATCH (d2:Document {entity_id: 'doc_demo_2'})
    MATCH (d3:Document {entity_id: 'doc_demo_3'})
    CREATE (u2)-[:READ {created_at: timestamp(), rating: 5, time_spent: 300}]->(d1)
    CREATE (u3)-[:READ {created_at: timestamp(), rating: 4, time_spent: 180}]->(d1)
    CREATE (u1)-[:READ {created_at: timestamp(), rating: 4, time_spent: 240}]->(d2)
    CREATE (u3)-[:READ {created_at: timestamp(), rating: 5, time_spent: 420}]->(d2)
    CREATE (u2)-[:READ {created_at: timestamp(), rating: 5, time_spent: 360}]->(d3)
    RETURN u1, u2, u3, d1, d2, d3
$$) as (u1 agtype, u2 agtype, u3 agtype, d1 agtype, d2 agtype, d3 agtype);

-- =====================================================================================
-- UTILITY FUNCTIONS FOR GRAPH OPERATIONS
-- =====================================================================================

-- Function to get graph statistics
CREATE OR REPLACE FUNCTION get_graph_statistics(graph_name text)
RETURNS TABLE (
    statistic_name text,
    statistic_value bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'total_nodes'::text, 
           (SELECT count(*)::bigint FROM cypher(graph_name, $cypher$ MATCH (n) RETURN count(n) $cypher$) as (node_count agtype))
    UNION ALL
    SELECT 'total_edges'::text,
           (SELECT count(*)::bigint FROM cypher(graph_name, $cypher$ MATCH ()-[r]->() RETURN count(r) $cypher$) as (edge_count agtype))
    UNION ALL
    SELECT 'user_nodes'::text,
           (SELECT count(*)::bigint FROM cypher(graph_name, $cypher$ MATCH (n:User) RETURN count(n) $cypher$) as (user_count agtype))
    UNION ALL
    SELECT 'document_nodes'::text,
           (SELECT count(*)::bigint FROM cypher(graph_name, $cypher$ MATCH (n:Document) RETURN count(n) $cypher$) as (doc_count agtype))
    UNION ALL
    SELECT 'product_nodes'::text,
           (SELECT count(*)::bigint FROM cypher(graph_name, $cypher$ MATCH (n:Product) RETURN count(n) $cypher$) as (product_count agtype))
    UNION ALL
    SELECT 'category_nodes'::text,
           (SELECT count(*)::bigint FROM cypher(graph_name, $cypher$ MATCH (n:Category) RETURN count(n) $cypher$) as (category_count agtype));
END;
$$ LANGUAGE plpgsql;

-- Function to find highly connected users
CREATE OR REPLACE FUNCTION get_influential_users(graph_name text, limit_count int DEFAULT 10)
RETURNS TABLE (
    entity_id text,
    name text,
    connection_count bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (result.user_data->>'entity_id')::text,
        (result.user_data->>'name')::text,
        (result.connections)::bigint
    FROM cypher(graph_name, 
        format($cypher$
            MATCH (u:User)-[r]-()
            RETURN u as user_data, count(r) as connections
            ORDER BY connections DESC
            LIMIT %s
        $cypher$, limit_count)
    ) as result(user_data agtype, connections agtype);
END;
$$ LANGUAGE plpgsql;

-- Function to get content recommendations for a user
CREATE OR REPLACE FUNCTION get_content_recommendations(graph_name text, user_entity_id text, limit_count int DEFAULT 5)
RETURNS TABLE (
    content_entity_id text,
    content_title text,
    recommendation_score numeric
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (result.content->>'entity_id')::text,
        (result.content->>'title')::text,
        (result.score)::numeric
    FROM cypher(graph_name, 
        format($cypher$
            MATCH (target_user:User {entity_id: '%s'})
            MATCH (target_user)-[:INTERESTED_IN]->(category:Category)<-[:BELONGS_TO]-(content:Document)
            WHERE NOT (target_user)-[:READ]->(content)
            MATCH (content)<-[:AUTHORED]-(author:User)
            MATCH (target_user)-[:FOLLOWS]->(author)
            RETURN content, 
                   (1.0 + count(DISTINCT category) * 0.3 + 
                    CASE WHEN (target_user)-[:FOLLOWS]->(author) THEN 0.5 ELSE 0 END) as score
            ORDER BY score DESC
            LIMIT %s
        $cypher$, user_entity_id, limit_count)
    ) as result(content agtype, score agtype);
END;
$$ LANGUAGE plpgsql;

-- Function to find shortest path between two entities
CREATE OR REPLACE FUNCTION find_entity_path(graph_name text, from_entity_id text, to_entity_id text)
RETURNS TABLE (
    path_info jsonb
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        jsonb_build_object(
            'path_length', result.path_length,
            'path_exists', CASE WHEN result.path_length IS NOT NULL THEN true ELSE false END
        )
    FROM cypher(graph_name, 
        format($cypher$
            MATCH (start {entity_id: '%s'}), (end {entity_id: '%s'})
            MATCH path = shortestPath((start)-[*]-(end))
            RETURN length(path) as path_length
        $cypher$, from_entity_id, to_entity_id)
    ) as result(path_length agtype);
END;
$$ LANGUAGE plpgsql;

-- =====================================================================================
-- SAMPLE QUERIES FOR TESTING
-- =====================================================================================

-- Test query 1: Find all users and their interests
-- SELECT * FROM cypher('unified_graph', $$
--     MATCH (u:User)-[:INTERESTED_IN]->(c:Category)
--     RETURN u.name as user_name, c.name as interest
-- $$) as (user_name agtype, interest agtype);

-- Test query 2: Find documents and their authors
-- SELECT * FROM cypher('unified_graph', $$
--     MATCH (u:User)-[:AUTHORED]->(d:Document)
--     RETURN u.name as author, d.title as document_title
-- $$) as (author agtype, document_title agtype);

-- Test query 3: Find users who follow each other
-- SELECT * FROM cypher('unified_graph', $$
--     MATCH (u1:User)-[:FOLLOWS]->(u2:User)-[:FOLLOWS]->(u1)
--     RETURN u1.name as user1, u2.name as user2
-- $$) as (user1 agtype, user2 agtype);

-- Test query 4: Product recommendations based on similarity
-- SELECT * FROM cypher('unified_graph', $$
--     MATCH (p1:Product)-[:SIMILAR_TO]->(p2:Product)
--     RETURN p1.name as product, p2.name as similar_product, 
--            p1.price as price, p2.price as similar_price
-- $$) as (product agtype, similar_product agtype, price agtype, similar_price agtype);

-- Test query 5: Find influential users (most connections)
-- SELECT * FROM cypher('unified_graph', $$
--     MATCH (u:User)-[r]-()
--     RETURN u.name as user_name, u.entity_id as entity_id, count(r) as connections
--     ORDER BY connections DESC
--     LIMIT 5
-- $$) as (user_name agtype, entity_id agtype, connections agtype);

-- =====================================================================================
-- CLEANUP FUNCTIONS
-- =====================================================================================

-- Function to clean up test data
CREATE OR REPLACE FUNCTION cleanup_graph_demo_data(graph_name text)
RETURNS boolean AS $$
BEGIN
    -- Remove all demo relationships
    PERFORM cypher(graph_name, $$
        MATCH ()-[r]-()
        WHERE r.created_at IS NOT NULL
        DELETE r
    $$);
    
    -- Remove all demo nodes
    PERFORM cypher(graph_name, $$
        MATCH (n)
        WHERE n.entity_id STARTS WITH 'user_demo_' 
           OR n.entity_id STARTS WITH 'doc_demo_'
           OR n.entity_id STARTS WITH 'product_demo_'
           OR n.entity_id STARTS WITH 'cat_'
        DELETE n
    $$);
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Function to validate graph integrity
CREATE OR REPLACE FUNCTION validate_graph_integrity(graph_name text)
RETURNS TABLE (
    check_name text,
    status text,
    details text
) AS $$
BEGIN
    -- Check for orphaned nodes
    RETURN QUERY
    SELECT 
        'orphaned_nodes'::text,
        CASE WHEN orphan_count > 0 THEN 'WARNING' ELSE 'OK' END::text,
        format('Found %s nodes with no relationships', orphan_count)::text
    FROM (
        SELECT count(*)::int as orphan_count
        FROM cypher(graph_name, $$
            MATCH (n)
            WHERE NOT (n)-[]-()
            RETURN count(n)
        $$) as (count agtype)
    ) orphan_check
    
    UNION ALL
    
    -- Check for nodes missing entity_id
    SELECT 
        'missing_entity_ids'::text,
        CASE WHEN missing_count > 0 THEN 'ERROR' ELSE 'OK' END::text,
        format('Found %s nodes missing entity_id property', missing_count)::text
    FROM (
        SELECT count(*)::int as missing_count
        FROM cypher(graph_name, $$
            MATCH (n)
            WHERE n.entity_id IS NULL OR n.entity_id = ''
            RETURN count(n)
        $$) as (count agtype)
    ) missing_check;
END;
$$ LANGUAGE plpgsql;

-- Final notification
DO $$
BEGIN
    RAISE NOTICE '🎉 Apache AGE graph schema setup completed successfully!';
    RAISE NOTICE 'Graph "unified_graph" created with sample nodes and relationships.';
    RAISE NOTICE 'Available node types: User, Document, Product, Category';
    RAISE NOTICE 'Available relationship types: FOLLOWS, COLLABORATES, KNOWS, AUTHORED, INTERESTED_IN, BELONGS_TO, SIMILAR_TO, READ';
    RAISE NOTICE 'Use the provided utility functions to interact with the graph.';
    RAISE NOTICE 'Run: SELECT * FROM get_graph_statistics(''unified_graph''); to see statistics.';
END $$;
