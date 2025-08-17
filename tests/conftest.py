"""
3db Unified Database Ecosystem - Test Configuration

This module provides configuration and fixtures for testing the 3db system.
"""

import os
import sys
import pytest
import pytest_asyncio
import asyncio
import types
import importlib
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

# Provide lightweight stubs for optional heavy dependencies used in the main
# project modules. This keeps tests fast and self-contained.
try:  # pragma: no cover
    import asyncpg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    asyncpg = types.ModuleType("asyncpg")
    asyncpg.create_pool = AsyncMock()  # type: ignore
    asyncpg.Pool = AsyncMock  # type: ignore
    sys.modules["asyncpg"] = asyncpg

try:  # pragma: no cover
    import numpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    numpy = types.ModuleType("numpy")
    sys.modules["numpy"] = numpy

try:  # pragma: no cover
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = types.SimpleNamespace(
        configure=lambda *a, **k: None,
        get_logger=lambda *a, **k: MagicMock(),
        stdlib=types.SimpleNamespace(
            filter_by_level=MagicMock(),
            add_logger_name=MagicMock(),
            add_log_level=MagicMock(),
            PositionalArgumentsFormatter=lambda: MagicMock(),
            LoggerFactory=MagicMock,
            BoundLogger=MagicMock,
        ),
        processors=types.SimpleNamespace(
            TimeStamper=lambda *a, **k: MagicMock(),
            StackInfoRenderer=MagicMock,
            format_exc_info=MagicMock,
            UnicodeDecoder=MagicMock,
            JSONRenderer=MagicMock,
        ),
    )
    sys.modules["structlog"] = structlog

try:  # pragma: no cover
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    redis = MagicMock()  # type: ignore
    sys.modules["redis"] = redis

# Add project root to path to allow `src` package imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Create a lightweight namespace package for `src` to avoid executing its
# heavy package initialization during tests.
src_package = types.ModuleType("src")
src_package.__path__ = [os.path.join(project_root, "src")]
sys.modules.setdefault("src", src_package)


# Provide a lightweight stub for sentence_transformers if the dependency is
# not installed. This avoids pulling in the heavy package for unit tests that
# simply need the class definition for mocking.
try:  # pragma: no cover - used only when dependency is missing
    import sentence_transformers  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - testing fallback
    sentence_transformers = types.ModuleType("sentence_transformers")

    class SentenceTransformer:  # minimal stub
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):  # pragma: no cover
            return []

    sentence_transformers.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers

from src.core.config import DatabaseConfig, UnifiedConfig

# Alias src.databases.core to src.core for modules using relative imports
core_pkg = importlib.import_module("src.core")
alias_pkg = types.ModuleType("src.databases.core")
alias_pkg.__path__ = getattr(core_pkg, "__path__", [])
sys.modules["src.databases.core"] = alias_pkg
sys.modules["src.databases.core.base"] = importlib.import_module("src.core.base")

# Alias src.databases.utils to src.utils
utils_pkg = importlib.import_module("src.utils")
utils_alias = types.ModuleType("src.databases.utils")
utils_alias.__path__ = getattr(utils_pkg, "__path__", []) if hasattr(utils_pkg, "__path__") else []
sys.modules["src.databases.utils"] = utils_alias
sys.modules["src.databases.utils.logging"] = importlib.import_module("src.utils.logging")

# Stub out sync.handler to avoid parsing heavy module
sync_handler_stub = types.ModuleType("src.sync.handler")
class UnifiedSynchronizationHandler:  # type: ignore
    pass

class EventBroker:  # type: ignore
    pass

class EntitySyncMetadataManager:  # type: ignore
    pass

class SyncEvent:  # type: ignore
    pass

class SyncEventType:  # type: ignore
    pass

sync_handler_stub.UnifiedSynchronizationHandler = UnifiedSynchronizationHandler
sync_handler_stub.EventBroker = EventBroker
sync_handler_stub.EntitySyncMetadataManager = EntitySyncMetadataManager
sync_handler_stub.SyncEvent = SyncEvent
sync_handler_stub.SyncEventType = SyncEventType
sys.modules["src.sync.handler"] = sync_handler_stub

from src.unified import Database3D
from src.databases.postgresql.crud import PostgreSQLDatabase
from src.databases.vector.embeddings import VectorDatabase
from src.databases.graph.relationships import GraphDatabase


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
        password='test_password'
    ),
    'graph': DatabaseConfig(
        host='localhost',
        port=5432,
        database='test_3db_graph',
        username='postgres',
        password='test_password'
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


@pytest_asyncio.fixture
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
    from src.core.base import QueryResult, DatabaseType
    
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


@pytest_asyncio.fixture
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
    from src.core.base import QueryResult, DatabaseType
    
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


@pytest_asyncio.fixture
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
    from src.core.base import QueryResult, DatabaseType
    
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


@pytest_asyncio.fixture
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
