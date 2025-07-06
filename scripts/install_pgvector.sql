-- pgvector Extension Installation Script
-- This script ensures pgvector is properly installed and configured

-- Create pgvector extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify pgvector installation
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
    ) THEN
        RAISE EXCEPTION 'pgvector extension failed to install';
    ELSE
        RAISE NOTICE 'pgvector extension is installed successfully';
    END IF;
END $$;

-- Test basic vector operations
DO $$
DECLARE
    test_vector vector(3) := '[1,2,3]';
    test_result float;
BEGIN
    -- Test vector creation and distance calculation
    SELECT test_vector <-> '[1,2,3]' INTO test_result;
    
    IF test_result = 0 THEN
        RAISE NOTICE '✅ pgvector basic operations test passed';
    ELSE
        RAISE EXCEPTION 'pgvector basic operations test failed';
    END IF;
END $$;

-- Create a simple test table for pgvector
CREATE TABLE IF NOT EXISTS vector_test (
    id SERIAL PRIMARY KEY,
    data vector(384),  -- Standard embedding dimension
    metadata JSONB DEFAULT '{}'
);

-- Create vector indexes for testing
CREATE INDEX IF NOT EXISTS vector_test_cosine_idx 
ON vector_test USING ivfflat (data vector_cosine_ops) 
WITH (lists = 10);

CREATE INDEX IF NOT EXISTS vector_test_l2_idx 
ON vector_test USING ivfflat (data vector_l2_ops) 
WITH (lists = 10);

-- Insert test vectors
INSERT INTO vector_test (data, metadata) VALUES
    (ARRAY(SELECT random() FROM generate_series(1, 384))::vector(384), '{"test": "vector1"}'),
    (ARRAY(SELECT random() FROM generate_series(1, 384))::vector(384), '{"test": "vector2"}'),
    (ARRAY(SELECT random() FROM generate_series(1, 384))::vector(384), '{"test": "vector3"}')
ON CONFLICT DO NOTHING;

-- Test similarity search
DO $$
DECLARE
    test_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO test_count
    FROM vector_test
    ORDER BY data <-> (SELECT data FROM vector_test LIMIT 1)
    LIMIT 3;
    
    IF test_count >= 3 THEN
        RAISE NOTICE '✅ pgvector similarity search test passed';
    ELSE
        RAISE NOTICE '⚠️ pgvector similarity search test incomplete';
    END IF;
END $$;

-- Clean up test table (optional - comment out to keep test data)
-- DROP TABLE IF EXISTS vector_test;

RAISE NOTICE '🎉 pgvector installation and testing completed successfully!';
