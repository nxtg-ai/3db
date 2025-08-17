"""
3db Unified Database Ecosystem - Core Base Classes

This module provides the foundational base classes and interfaces for the unified
database ecosystem, ensuring consistent behavior across PostgreSQL CRUD, pgvector,
and Apache AGE graph databases.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Enumeration of supported database types."""
    POSTGRESQL = "postgresql"
    VECTOR = "vector"
    GRAPH = "graph"


class SyncMode(Enum):
    """Synchronization modes for data consistency."""
    IMMEDIATE = "immediate"  # Synchronous updates
    EVENTUAL = "eventual"    # Asynchronous updates via CDC
    HYBRID = "hybrid"        # Mix of immediate and eventual


@dataclass
class EntityMetadata:
    """Metadata for entities across the unified ecosystem."""
    
    entity_id: str
    entity_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    # Database presence tracking
    in_postgresql: bool = False
    in_vector: bool = False
    in_graph: bool = False
    
    # Synchronization tracking
    sync_status: Dict[str, datetime] = field(default_factory=dict)
    last_sync_attempt: Optional[datetime] = None
    sync_errors: List[str] = field(default_factory=list)
    
    def mark_synced(self, database_type: DatabaseType) -> None:
        """Mark entity as synchronized in a specific database."""
        self.sync_status[database_type.value] = datetime.utcnow()
        self.last_sync_attempt = datetime.utcnow()
    
    def has_sync_errors(self) -> bool:
        """Check if entity has any synchronization errors."""
        return len(self.sync_errors) > 0
    
    def clear_sync_errors(self) -> None:
        """Clear all synchronization errors."""
        self.sync_errors.clear()


@dataclass
class QueryResult:
    """Standardized query result across all database types."""
    
    success: bool
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    error: Optional[str] = None
    
    # Database-specific information
    source_database: Optional[DatabaseType] = None
    query_id: Optional[str] = None
    affected_rows: int = 0
    
    def __post_init__(self):
        if not self.success and not self.error:
            self.error = "Unknown error occurred"


class BaseDatabase(ABC):
    """Abstract base class for all database implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.database_type = self._get_database_type()
        self.connection_pool = None
        self.is_connected = False
        self._metrics = {
            'queries_executed': 0,
            'queries_failed': 0,
            'total_execution_time': 0.0,
            'connections_opened': 0,
            'connections_closed': 0
        }
    
    @abstractmethod
    def _get_database_type(self) -> DatabaseType:
        """Return the database type for this implementation."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform a health check on the database connection."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a query and return standardized result."""
        pass
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get database metrics."""
        return {
            **self._metrics,
            'database_type': self.database_type.value,
            'is_connected': self.is_connected,
            'avg_execution_time': (
                self._metrics['total_execution_time'] / self._metrics['queries_executed']
                if self._metrics['queries_executed'] > 0 else 0
            )
        }
    
    def _update_metrics(self, execution_time: float, success: bool = True) -> None:
        """Update internal metrics."""
        self._metrics['total_execution_time'] += execution_time
        if success:
            self._metrics['queries_executed'] += 1
        else:
            self._metrics['queries_failed'] += 1
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        # Base implementation - subclasses should override
        yield
    
    async def bulk_execute(self, queries: List[Tuple[str, Optional[Dict[str, Any]]]]) -> List[QueryResult]:
        """Execute multiple queries in batch."""
        results = []
        for query, params in queries:
            result = await self.execute_query(query, params)
            results.append(result)
        return results


class CRUDOperations(ABC):
    """Abstract base class for CRUD operations."""
    
    @abstractmethod
    async def create(self, table: str, data: Dict[str, Any]) -> QueryResult:
        """Create a new record."""
        pass
    
    @abstractmethod
    async def read(self, table: str, filters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Read records with optional filters."""
        pass
    
    @abstractmethod
    async def update(self, table: str, data: Dict[str, Any], filters: Dict[str, Any]) -> QueryResult:
        """Update records matching filters."""
        pass
    
    @abstractmethod
    async def delete(self, table: str, filters: Dict[str, Any]) -> QueryResult:
        """Delete records matching filters."""
        pass
    
    @abstractmethod
    async def upsert(self, table: str, data: Dict[str, Any], conflict_columns: List[str]) -> QueryResult:
        """Insert or update record based on conflict columns."""
        pass


class VectorOperations(ABC):
    """Abstract base class for vector database operations."""
    
    @abstractmethod
    async def insert_embedding(self, entity_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Insert a vector embedding."""
        pass
    
    @abstractmethod
    async def similarity_search(self, query_embedding: List[float], limit: int = 10, threshold: float = 0.8) -> QueryResult:
        """Perform similarity search on embeddings."""
        pass
    
    @abstractmethod
    async def update_embedding(self, entity_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Update an existing embedding."""
        pass
    
    @abstractmethod
    async def delete_embedding(self, entity_id: str) -> QueryResult:
        """Delete an embedding."""
        pass
    
    @abstractmethod
    async def get_embedding(self, entity_id: str) -> QueryResult:
        """Retrieve a specific embedding."""
        pass


class GraphOperations(ABC):
    """Abstract base class for graph database operations."""
    
    @abstractmethod
    async def create_node(self, label: str, properties: Dict[str, Any]) -> QueryResult:
        """Create a new node in the graph."""
        pass
    
    @abstractmethod
    async def create_edge(self, from_node_id: str, to_node_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Create a relationship between two nodes."""
        pass
    
    @abstractmethod
    async def find_nodes(self, label: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Find nodes matching criteria."""
        pass
    
    @abstractmethod
    async def find_relationships(self, node_id: str, relationship_type: Optional[str] = None, direction: str = "both") -> QueryResult:
        """Find relationships for a node."""
        pass
    
    @abstractmethod
    async def graph_traversal(self, start_node_id: str, max_depth: int = 3, filters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Perform graph traversal from a starting node."""
        pass
    
    @abstractmethod
    async def shortest_path(self, from_node_id: str, to_node_id: str) -> QueryResult:
        """Find shortest path between two nodes."""
        pass
    
    @abstractmethod
    async def delete_node(self, node_id: str) -> QueryResult:
        """Delete a node and its relationships."""
        pass
    
    @abstractmethod
    async def delete_relationship(self, relationship_id: str) -> QueryResult:
        """Delete a specific relationship."""
        pass


class SynchronizationHandler(ABC):
    """Abstract base class for synchronization between databases."""
    
    @abstractmethod
    async def sync_entity(self, entity_metadata: EntityMetadata, source_db: DatabaseType, target_dbs: List[DatabaseType]) -> bool:
        """Synchronize an entity across databases."""
        pass
    
    @abstractmethod
    async def handle_conflict(self, entity_id: str, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle synchronization conflicts."""
        pass
    
    @abstractmethod
    async def get_sync_status(self, entity_id: str) -> Dict[str, Any]:
        """Get synchronization status for an entity."""
        pass


class EventHandler(ABC):
    """Abstract base class for event-driven synchronization."""
    
    @abstractmethod
    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Publish an event to the message broker."""
        pass
    
    @abstractmethod
    async def subscribe_to_events(self, event_types: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to specific event types."""
        pass
    
    @abstractmethod
    async def handle_event(self, event: Dict[str, Any]) -> bool:
        """Handle a received event."""
        pass


class MetadataManager(ABC):
    """Abstract base class for entity metadata management."""
    
    @abstractmethod
    async def store_metadata(self, metadata: EntityMetadata) -> bool:
        """Store entity metadata."""
        pass
    
    @abstractmethod
    async def get_metadata(self, entity_id: str) -> Optional[EntityMetadata]:
        """Retrieve entity metadata."""
        pass
    
    @abstractmethod
    async def update_metadata(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity metadata."""
        pass
    
    @abstractmethod
    async def delete_metadata(self, entity_id: str) -> bool:
        """Delete entity metadata."""
        pass
    
    @abstractmethod
    async def list_entities_by_type(self, entity_type: str) -> List[EntityMetadata]:
        """List all entities of a specific type."""
        pass


class UnifiedDatabase:
    """
    Main orchestrator for the unified database ecosystem.
    Coordinates operations across PostgreSQL, pgvector, and Apache AGE.
    """
    
    def __init__(self, 
                 postgresql_db: BaseDatabase,
                 vector_db: BaseDatabase,
                 graph_db: BaseDatabase,
                 sync_handler: SynchronizationHandler,
                 metadata_manager: MetadataManager):
        
        self.postgresql = postgresql_db
        self.vector = vector_db
        self.graph = graph_db
        self.sync_handler = sync_handler
        self.metadata_manager = metadata_manager
        
        self.databases = {
            DatabaseType.POSTGRESQL: postgresql_db,
            DatabaseType.VECTOR: vector_db,
            DatabaseType.GRAPH: graph_db
        }
    
    async def initialize(self) -> bool:
        """Initialize all database connections."""
        try:
            for db_type, db in self.databases.items():
                success = await db.connect()
                if not success:
                    logger.error(f"Failed to connect to {db_type.value} database")
                    return False
                    
                health = await db.health_check()
                if not health:
                    logger.error(f"{db_type.value} database health check failed")
                    return False
            
            logger.info("All databases initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown all database connections."""
        for db in self.databases.values():
            await db.disconnect()
        logger.info("All databases disconnected")
    
    async def get_database(self, db_type: DatabaseType) -> BaseDatabase:
        """Get a specific database instance."""
        return self.databases.get(db_type)
    
    async def execute_unified_query(self,
                                   query_type: str,
                                   entity_data: Dict[str, Any],
                                   target_databases: Optional[List[DatabaseType]] = None) -> Dict[DatabaseType, QueryResult]:
        """Execute a query against one or more databases.

        This method provides a thin orchestration layer that dispatches
        operations to the appropriate database handler based on ``query_type``.
        The ``entity_data`` dictionary is passed as keyword arguments to the
        handler method. Results from each database are aggregated into a single
        dictionary keyed by :class:`DatabaseType`.

        Parameters
        ----------
        query_type:
            Name of the handler method to execute (e.g. ``"create"`` or
            ``"execute_query"``).
        entity_data:
            Parameters to forward to the handler method.
        target_databases:
            Optional list of databases to target.  When ``None`` all registered
            databases are used.

        Returns
        -------
        Dict[DatabaseType, QueryResult]
            Mapping of database type to the resulting :class:`QueryResult`.
        """
        if target_databases is None:
            target_databases = list(self.databases.keys())

        results: Dict[DatabaseType, QueryResult] = {}

        entity_id = entity_data.get("entity_id")
        entity_type = entity_data.get("entity_type", "unknown")
        metadata: Optional[EntityMetadata] = None

        if entity_id and self.metadata_manager:
            try:
                metadata = await self.metadata_manager.get_metadata(entity_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Metadata lookup failed: {exc}")

            if metadata is None:
                metadata = EntityMetadata(entity_id=entity_id, entity_type=entity_type)

        for db_type in target_databases:
            db = self.databases.get(db_type)
            if not db:
                continue

            try:
                handler = getattr(db, query_type, None)
                if not callable(handler):
                    raise AttributeError(f"{db_type.value} database has no '{query_type}' method")

                result = await handler(**entity_data)

                if not isinstance(result, QueryResult):
                    result = QueryResult(success=True, data=result, source_database=db_type)

                results[db_type] = result

                if metadata and result.success:
                    setattr(metadata, f"in_{db_type.value}", True)
                    metadata.mark_synced(db_type)
                elif metadata and not result.success:
                    metadata.sync_errors.append(result.error or f"{db_type.value} {query_type} failed")

            except Exception as exc:
                logger.error(f"{db_type.value} {query_type} failed: {exc}")
                results[db_type] = QueryResult(
                    success=False,
                    data=None,
                    error=str(exc),
                    source_database=db_type,
                )
                if metadata:
                    metadata.sync_errors.append(str(exc))

        if metadata and self.metadata_manager:
            try:
                existing = await self.metadata_manager.get_metadata(metadata.entity_id)
                if existing:
                    await self.metadata_manager.update_metadata(metadata.entity_id, asdict(metadata))
                else:
                    await self.metadata_manager.store_metadata(metadata)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Failed to update metadata for {metadata.entity_id}: {exc}")

        return results
    
    async def get_unified_metrics(self) -> Dict[str, Any]:
        """Get metrics from all databases."""
        metrics = {}
        for db_type, db in self.databases.items():
            metrics[db_type.value] = await db.get_metrics()
        return metrics


# Utility functions for common operations
async def create_unified_entity(unified_db: UnifiedDatabase,
                               entity_type: str,
                               entity_data: Dict[str, Any],
                               sync_mode: SyncMode = SyncMode.HYBRID) -> str:
    """Create an entity across all databases and record metadata.

    The function generates an ``entity_id`` if one is not provided and then
    performs the appropriate create/insert operation on each database handled by
    ``unified_db``.  Metadata is updated to reflect which databases contain the
    entity.  Basic error handling ensures that failures in one database do not
    prevent attempts on others.

    Parameters
    ----------
    unified_db:
        Instance of :class:`UnifiedDatabase` coordinating the databases.
    entity_type:
        Logical type of the entity being created (e.g. ``"user"``).
    entity_data:
        Data describing the entity.  The dictionary is forwarded to the
        underlying database implementations.
    sync_mode:
        Determines whether the operation should target all databases immediately
        or defer to eventual consistency.  ``SyncMode.EVENTUAL`` only stores the
        entity in PostgreSQL and records metadata for later synchronization.

    Returns
    -------
    str
        The identifier for the created entity.
    """
    entity_id = entity_data.get("entity_id") or f"{entity_type}_{int(datetime.utcnow().timestamp() * 1000)}"
    entity_data["entity_id"] = entity_id
    entity_data.setdefault("entity_type", entity_type)

    metadata = EntityMetadata(entity_id=entity_id, entity_type=entity_type)

    # Always create in PostgreSQL as the source of truth
    try:
        pg_result = await unified_db.postgresql.create(f"{entity_type}s", entity_data)
        if pg_result.success:
            metadata.in_postgresql = True
            metadata.mark_synced(DatabaseType.POSTGRESQL)
        else:
            metadata.sync_errors.append(pg_result.error or "postgresql create failed")
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"PostgreSQL create failed: {exc}")
        metadata.sync_errors.append(str(exc))

    # If immediate sync is requested, attempt to store in vector and graph DBs
    if sync_mode != SyncMode.EVENTUAL:
        # Vector database – insert text embedding if provided
        try:
            if hasattr(unified_db.vector, "insert_text_embedding") and (
                entity_data.get("text_content") or entity_data.get("embedding")
            ):
                text = entity_data.get("text_content")
                if text is not None:
                    v_result = await unified_db.vector.insert_text_embedding(entity_id, text, {"entity_type": entity_type})
                else:
                    v_result = await unified_db.vector.insert_embedding(entity_id, entity_data["embedding"], {"entity_type": entity_type})
                if v_result.success:
                    metadata.in_vector = True
                    metadata.mark_synced(DatabaseType.VECTOR)
                else:
                    metadata.sync_errors.append(v_result.error or "vector insert failed")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Vector insert failed: {exc}")
            metadata.sync_errors.append(str(exc))

        # Graph database – create node
        try:
            if hasattr(unified_db.graph, "create_node"):
                node_props = dict(entity_data)
                g_result = await unified_db.graph.create_node(entity_type, node_props)
                if g_result.success:
                    metadata.in_graph = True
                    metadata.mark_synced(DatabaseType.GRAPH)
                else:
                    metadata.sync_errors.append(g_result.error or "graph create failed")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Graph create failed: {exc}")
            metadata.sync_errors.append(str(exc))

    # Store metadata
    if unified_db.metadata_manager:
        try:
            await unified_db.metadata_manager.store_metadata(metadata)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to store metadata for {entity_id}: {exc}")

    return entity_id


async def query_across_databases(unified_db: UnifiedDatabase,
                                query_spec: Dict[str, Any]) -> Dict[str, QueryResult]:
    """Execute queries on multiple databases and aggregate the results.

    The ``query_spec`` dictionary describes which method to call on each
    database and the parameters to use.  Keys can be either
    :class:`DatabaseType` members or their string representations.  Each value
    must at least contain a ``"method"`` entry specifying the handler name.

    Example
    -------
    >>> await query_across_databases(db, {
    ...     DatabaseType.POSTGRESQL: {"method": "execute_query", "query": "SELECT 1"},
    ...     "vector": {"method": "similarity_search", "query_embedding": [0.1, 0.2], "limit": 5}
    ... })

    Parameters
    ----------
    unified_db:
        The :class:`UnifiedDatabase` instance.
    query_spec:
        Specification of database operations to execute.

    Returns
    -------
    Dict[str, QueryResult]
        Mapping of database name to :class:`QueryResult` objects.
    """
    results: Dict[str, QueryResult] = {}

    for db_key, spec in query_spec.items():
        try:
            db_type = db_key if isinstance(db_key, DatabaseType) else DatabaseType(db_key)
        except Exception:
            logger.error(f"Invalid database type: {db_key}")
            continue

        db = await unified_db.get_database(db_type)
        if db is None:
            logger.error(f"Database handler not available for {db_type.value}")
            continue

        method_name = spec.get("method", "execute_query")
        params = {k: v for k, v in spec.items() if k != "method"}

        try:
            method = getattr(db, method_name)
            result = await method(**params)
            if not isinstance(result, QueryResult):
                result = QueryResult(success=True, data=result, source_database=db_type)
            results[db_type.value] = result
        except Exception as exc:
            logger.error(f"Query on {db_type.value} failed: {exc}")
            results[db_type.value] = QueryResult(
                success=False,
                data=None,
                error=str(exc),
                source_database=db_type,
            )

    return results


# Example usage and testing
if __name__ == "__main__":
    # This section would contain example usage and basic tests
    print("3db Core Base Classes Loaded Successfully")
    print("Available Database Types:", [db_type.value for db_type in DatabaseType])
    print("Available Sync Modes:", [sync_mode.value for sync_mode in SyncMode])
