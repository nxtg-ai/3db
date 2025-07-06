"""
3db Unified Database Ecosystem - Test Configuration

This module provides configuration and fixtures for testing the 3db system.
"""

import os
import sys
import pytest
import asyncio
import asyncpg
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.config import DatabaseConfig, UnifiedConfig
from unified import Database3D
from databases.postgresql.crud import PostgreSQLDatabase
from databases.vector.embeddings import VectorDatabase
from databases.graph.relationships import GraphDatabase


# Test configuration
TEST_CONFIG = {
    'postgresql': DatabaseConfig(
        host='localhost',
        port=5432,
        database='test_3db',
        username='postgres',
        password='test_password'
    ),
    'vector': DatabaseConfig(
        host='localhost',
        port=5432,
        database='test_3db_vector',
        username='postgres',
        password='test_password',
        embedding_dimension=384,
        similarity_threshold=0.8,
        embedding_model='all-MiniLM-L6-v2'
    ),
    'graph': DatabaseConfig(
        host='localhost',
        port=5432,
        database='test_3db_graph',
        username='postgres',
        password='test_password',
        graph_name='test_graph',
        max_traversal_depth=5
    )
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Provide mock configuration for testing."""
    config = MagicMock()
    config.postgresql = TEST_CONFIG['postgresql']
    config.vector = TEST_CONFIG['vector']
    config.graph = TEST_CONFIG['graph']
    
    # Mock sync configuration
    config.sync = MagicMock()
    config.sync.eda_enabled = True
    config.sync.cdc_enabled = True
    config.sync.cdc_batch_size = 10
    
    # Mock cache configuration
    config.cache = MagicMock()
    config.cache.redis_url = "redis://localhost:6379/1"
    
    return config


@pytest.fixture
async def mock_postgresql_db():
    """Provide mock PostgreSQL database for testing."""
    db = AsyncMock(spec=PostgreSQLDatabase)
    db.is_connected = True
    db.database_type = "postgresql"
    
    # Mock successful operations
    db.connect.return_value = True
    db.health_check.return_value = True
    db.disconnect.return_value = None
    
    # Mock query results
    from core.base import QueryResult, DatabaseType
    
    db.execute_query.return_value = QueryResult(
        success=True,
        data=[{"id": 1, "entity_id": "test_entity_1", "name": "Test Entity"}],
        source_database=DatabaseType.POSTGRESQL,
        affected_rows=1
    )
    
    db.create.return_value = QueryResult(
        success=True,
        data={"id": 1, "entity_id": "test_entity_1", "name": "Test Entity"},
        source_database=DatabaseType.POSTGRESQL,
        affected_rows=1
    )
    
    db.read.return_value = QueryResult(
        success=True,
        data=[{"id": 1, "entity_id": "test_entity_1", "name": "Test Entity"}],
        source_database=DatabaseType.POSTGRESQL,
        affected_rows=1
    )
    
    return db


@pytest.fixture
async def mock_vector_db():
    """Provide mock vector database for testing."""
    db = AsyncMock(spec=VectorDatabase)
    db.is_connected = True
    db.database_type = "vector"
    db._model_loaded = True
    
    # Mock successful operations
    db.connect.return_value = True
    db.health_check.return_value = True
    db.disconnect.return_value = None
    
    # Mock embedding generation
    db.generate_embedding.return_value = [0.1] * 384
    
    # Mock vector operations
    from core.base import QueryResult, DatabaseType
    
    db.insert_embedding.return_value = QueryResult(
        success=True,
        data={"entity_id": "test_entity_1", "embedding": [0.1] * 384},
        source_database=DatabaseType.VECTOR,
        affected_rows=1
    )
    
    db.similarity_search.return_value = QueryResult(
        success=True,
        data=[
            {"entity_id": "test_entity_1", "similarity": 0.95},
            {"entity_id": "test_entity_2", "similarity": 0.87}
        ],
        source_database=DatabaseType.VECTOR,
        affected_rows=2
    )
    
    return db


@pytest.fixture
async def mock_graph_db():
    """Provide mock graph database for testing."""
    db = AsyncMock(spec=GraphDatabase)
    db.is_connected = True
    db.database_type = "graph"
    
    # Mock successful operations
    db.connect.return_value = True
    db.health_check.return_value = True
    db.disconnect.return_value = None
    
    # Mock graph operations
    from core.base import QueryResult, DatabaseType
    
    db.create_node.return_value = QueryResult(
        success=True,
        data={
            "node_id": 1,
            "entity_id": "test_entity_1",
            "label": "TestEntity",
            "properties": {"name": "Test Entity"}
        },
        source_database=DatabaseType.GRAPH,
        affected_rows=1
    )
    
    db.graph_traversal.return_value = QueryResult(
        success=True,
        data=[
            {"path": [{"entity_id": "test_entity_1"}, {"entity_id": "test_entity_2"}], "depth": 1}
        ],
        source_database=DatabaseType.GRAPH,
        affected_rows=1
    )
    
    return db


@pytest.fixture
async def mock_3db_system(mock_config, mock_postgresql_db, mock_vector_db, mock_graph_db):
    """Provide mock 3db system for testing."""
    db3d = Database3D(mock_config)
    
    # Replace real databases with mocks
    db3d.postgresql_db = mock_postgresql_db
    db3d.vector_db = mock_vector_db
    db3d.graph_db = mock_graph_db
    
    # Mock initialization
    db3d.is_initialized = True
    db3d.is_running = True
    
    # Mock sync components
    db3d.metadata_manager = AsyncMock()
    db3d.metadata_manager.get_sync_statistics.return_value = {
        "total_entities": 10,
        "in_postgresql": 10,
        "in_vector": 8,
        "in_graph": 6,
        "fully_synced": 6
    }
    
    db3d.sync_handler = AsyncMock()
    db3d.sync_handler.sync_entity.return_value = True
    
    return db3d


@pytest.fixture
def sample_entity_data():
    """Provide sample entity data for testing."""
    return {
        "user": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "bio": "Software engineer with expertise in databases and AI",
            "interests": "databases, artificial intelligence, distributed systems"
        },
        "document": {
            "title": "Introduction to 3db Systems",
            "content": "This document explains the concepts behind unified database architectures",
            "author_id": "user_123",
            "category": "technology",
            "tags": ["databases", "architecture", "technology"]
        },
        "product": {
            "name": "Database Optimization Tool",
            "description": "Advanced tool for optimizing database performance",
            "category": "software",
            "price": 299.99,
            "features": ["Query optimization", "Index tuning", "Performance monitoring"]
        }
    }


# Test utilities
class TestDatabase:
    """Utility class for database testing."""
    
    @staticmethod
    async def create_test_tables(connection):
        """Create test tables for integration testing."""
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS test_entities (
                id SERIAL PRIMARY KEY,
                entity_id VARCHAR(255) UNIQUE NOT NULL,
                entity_type VARCHAR(100) NOT NULL,
                name VARCHAR(500),
                data JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    @staticmethod
    async def cleanup_test_data(connection):
        """Clean up test data after testing."""
        await connection.execute("DELETE FROM test_entities WHERE entity_id LIKE 'test_%'")
    
    @staticmethod
    async def insert_test_entity(connection, entity_id: str, entity_type: str, name: str, data: dict = None):
        """Insert test entity for testing."""
        data = data or {}
        await connection.execute(
            "INSERT INTO test_entities (entity_id, entity_type, name, data) VALUES ($1, $2, $3, $4)",
            entity_id, entity_type, name, data
        )


# Performance testing utilities
class PerformanceTest:
    """Utility class for performance testing."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        import time
        self.metrics[operation] = {"start": time.time()}
    
    def end_timer(self, operation: str):
        """End timing an operation."""
        import time
        if operation in self.metrics:
            self.metrics[operation]["end"] = time.time()
            self.metrics[operation]["duration"] = (
                self.metrics[operation]["end"] - self.metrics[operation]["start"]
            )
    
    def get_duration(self, operation: str) -> float:
        """Get duration of an operation."""
        return self.metrics.get(operation, {}).get("duration", 0.0)
    
    def assert_performance(self, operation: str, max_duration: float):
        """Assert that operation completed within time limit."""
        duration = self.get_duration(operation)
        assert duration <= max_duration, f"Operation {operation} took {duration:.3f}s, expected <= {max_duration}s"


# Integration test markers
pytest.mark.integration = pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true",
    reason="Integration tests disabled"
)

pytest.mark.performance = pytest.mark.skipif(
    os.getenv("SKIP_PERFORMANCE_TESTS", "false").lower() == "true",
    reason="Performance tests disabled"
)

pytest.mark.slow = pytest.mark.skipif(
    os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true",
    reason="Slow tests disabled"
)
