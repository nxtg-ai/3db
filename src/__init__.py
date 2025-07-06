"""
3db Unified Database Ecosystem

An intelligent database system that seamlessly combines PostgreSQL CRUD operations,
pgvector similarity search, and Apache AGE graph relationships into a unified,
brain-like database ecosystem.

Key Features:
- Automatic synchronization across database types
- Semantic similarity search with AI embeddings  
- Graph relationship analysis and traversals
- AI-powered recommendations
- Federated queries spanning multiple databases
- Real-time monitoring and performance optimization

Quick Start:
    from src.unified import Database3D
    
    # Initialize the system
    db3d = Database3D()
    await db3d.initialize()
    
    # Create an entity (automatically synced across all databases)
    result = await db3d.create_entity('user', {
        'name': 'Alice Johnson',
        'email': 'alice@example.com',
        'bio': 'AI researcher specializing in neural networks'
    })
    
    # Perform semantic search
    search_results = await db3d.search_similar(
        'artificial intelligence research',
        entity_type='user',
        limit=10
    )
    
    # Get AI recommendations
    recommendations = await db3d.get_recommendations(
        user_id=result['entity_id'],
        recommendation_type='content'
    )
    
    # Analyze entity network
    network = await db3d.analyze_entity_network(
        entity_id=result['entity_id'],
        max_depth=3
    )
    
    # Cleanup
    await db3d.shutdown()

For more examples and documentation, see:
- README.md for complete setup instructions
- examples/complete_demo.py for full demonstration
- api/main.py for REST API usage
- cli/main.py for command-line interface
"""

__version__ = "1.0.0"
__author__ = "3db Development Team"
__email__ = "support@3db.ai"
__license__ = "MIT"

# Import main classes for easy access
from .unified import Database3D
from .core.config import get_config, UnifiedConfig
from .core.base import DatabaseType, SyncMode, QueryResult

# Import database implementations
from .databases.postgresql.crud import PostgreSQLDatabase
from .databases.vector.embeddings import VectorDatabase
from .databases.graph.relationships import GraphDatabase

# Import synchronization components
from .sync.handler import UnifiedSynchronizationHandler, SyncEvent, SyncEventType

# Import query coordination
from .query.coordinator import UnifiedQueryInterface, QueryType

# Export public API
__all__ = [
    # Main classes
    'Database3D',
    'get_config',
    'UnifiedConfig',
    
    # Enums and data structures
    'DatabaseType',
    'SyncMode',
    'QueryResult',
    'QueryType',
    'SyncEventType',
    
    # Database implementations
    'PostgreSQLDatabase',
    'VectorDatabase', 
    'GraphDatabase',
    
    # Coordination components
    'UnifiedSynchronizationHandler',
    'UnifiedQueryInterface',
    'SyncEvent',
    
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]


def get_version():
    """Get the current version of 3db."""
    return __version__


def get_system_info():
    """Get system information and component versions."""
    import sys
    import platform
    
    return {
        '3db_version': __version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'components': {
            'postgresql': 'PostgreSQL 15+',
            'pgvector': 'pgvector 0.5+',
            'apache_age': 'Apache AGE 1.4+',
            'fastapi': 'FastAPI 0.103+',
            'asyncpg': 'asyncpg 0.29+',
            'sentence_transformers': 'sentence-transformers 2.2+'
        }
    }
