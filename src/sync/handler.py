"""
3db Unified Database Ecosystem - Synchronization System

This module provides the core synchronization functionality that coordinates
data consistency across PostgreSQL CRUD, pgvector, and Apache AGE databases.
Implements both Event-Driven Architecture (EDA) and Change Data Capture (CDC).
"""

import asyncio
import redis
import json
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import uuid

from ..core.base import (
    SynchronizationHandler, EventHandler, MetadataManager, EntityMetadata,
    DatabaseType, SyncMode, UnifiedDatabase
)
from ..core.config import get_config
from ..utils.logging import get_logger, monitor_performance, monitor_sync_performance

logger = get_logger("sync_system")


class SyncEventType(Enum):
    """Types of synchronization events."""
    ENTITY_CREATED = "entity_created"
    ENTITY_UPDATED = "entity_updated"
    ENTITY_DELETED = "entity_deleted"
    EMBEDDING_GENERATED = "embedding_generated"
    RELATIONSHIP_CREATED = "relationship_created"
    RELATIONSHIP_DELETED = "relationship_deleted"
    SYNC_CONFLICT = "sync_conflict"
    SYNC_ERROR = "sync_error"


@dataclass
class SyncEvent:
    """Synchronization event data structure."""
    
    event_id: str
    event_type: SyncEventType
    entity_id: str
    entity_type: str
    source_database: DatabaseType
    target_databases: List[DatabaseType]
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=high, 2=medium, 3=low
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'source_database': self.source_database.value,
            'target_databases': [db.value for db in self.target_databases],
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'retries': self.retries,
            'max_retries': self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncEvent':
        """Create from dictionary."""
        return cls(
            event_id=data['event_id'],
            event_type=SyncEventType(data['event_type']),
            entity_id=data['entity_id'],
            entity_type=data['entity_type'],
            source_database=DatabaseType(data['source_database']),
            target_databases=[DatabaseType(db) for db in data['target_databases']],
            data=data['data'],
            metadata=data['metadata'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            priority=data.get('priority', 1),
            retries=data.get('retries', 0),
            max_retries=data.get('max_retries', 3)
        )


class EventBroker:
    """Redis-based event broker for real-time synchronization."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub = None
        
        # Event queues by priority
        self.high_priority_queue = "3db:sync:high"
        self.medium_priority_queue = "3db:sync:medium"
        self.low_priority_queue = "3db:sync:low"
        self.error_queue = "3db:sync:errors"
        
        # Channel for real-time events
        self.sync_channel = "3db:sync:events"
    
    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            self.pubsub = self.redis_client.pubsub()
            logger.info("Connected to Redis event broker")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.pubsub:
            await asyncio.get_event_loop().run_in_executor(None, self.pubsub.close)
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.close)
        logger.info("Disconnected from Redis event broker")
    
    def _get_queue_by_priority(self, priority: int) -> str:
        """Get queue name by priority."""
        if priority == 1:
            return self.high_priority_queue
        elif priority == 2:
            return self.medium_priority_queue
        else:
            return self.low_priority_queue
    
    async def publish_event(self, event: SyncEvent) -> bool:
        """Publish synchronization event."""
        try:
            event_data = json.dumps(event.to_dict())
            
            # Add to priority queue
            queue_name = self._get_queue_by_priority(event.priority)
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.lpush, queue_name, event_data
            )
            
            # Publish to real-time channel
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.publish, self.sync_channel, event_data
            )
            
            logger.debug(f"Published sync event: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            return False
    
    async def consume_events(self, batch_size: int = 10) -> List[SyncEvent]:
        """Consume events from priority queues."""
        events = []
        
        try:
            # Process high priority first
            for queue in [self.high_priority_queue, self.medium_priority_queue, self.low_priority_queue]:
                while len(events) < batch_size:
                    event_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.rpop, queue
                    )
                    
                    if not event_data:
                        break
                    
                    try:
                        event_dict = json.loads(event_data)
                        event = SyncEvent.from_dict(event_dict)
                        events.append(event)
                    except Exception as e:
                        logger.error(f"Failed to parse event: {e}")
                        # Move to error queue
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.redis_client.lpush, self.error_queue, event_data
                        )
                
                if len(events) >= batch_size:
                    break
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to consume events: {e}")
            return []
    
    async def requeue_event(self, event: SyncEvent) -> bool:
        """Requeue failed event with incremented retry count."""
        try:
            event.retries += 1
            
            if event.retries >= event.max_retries:
                # Move to error queue
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.lpush, self.error_queue, 
                    json.dumps(event.to_dict())
                )
                logger.warning(f"Event {event.event_id} moved to error queue after {event.retries} retries")
                return False
            else:
                # Requeue with lower priority
                new_priority = min(event.priority + 1, 3)
                event.priority = new_priority
                queue_name = self._get_queue_by_priority(new_priority)
                
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.lpush, queue_name, 
                    json.dumps(event.to_dict())
                )
                logger.debug(f"Requeued event {event.event_id} with priority {new_priority}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to requeue event {event.event_id}: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics about queue lengths."""
        try:
            stats = {}
            for name, queue in [
                ("high_priority", self.high_priority_queue),
                ("medium_priority", self.medium_priority_queue),
                ("low_priority", self.low_priority_queue),
                ("errors", self.error_queue)
            ]:
                length = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.llen, queue
                )
                stats[name] = length
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}


class EntitySyncMetadataManager(MetadataManager):
    """Manages entity metadata for synchronization tracking."""
    
    def __init__(self, postgresql_db):
        self.postgresql_db = postgresql_db
        self.table_name = "entity_sync_metadata"
    
    async def initialize(self) -> bool:
        """Initialize metadata tables."""
        try:
            # Create metadata table
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
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
                )
            """
            
            result = await self.postgresql_db.execute_query(create_table_sql)
            
            if result.success:
                # Create indexes
                indexes = [
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_entity_id_idx ON {self.table_name} (entity_id)",
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_entity_type_idx ON {self.table_name} (entity_type)",
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_updated_at_idx ON {self.table_name} (updated_at)"
                ]
                
                for index_sql in indexes:
                    await self.postgresql_db.execute_query(index_sql)
                
                logger.info("Entity sync metadata tables initialized")
                return True
            else:
                logger.error(f"Failed to create metadata table: {result.error}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize metadata manager: {e}")
            return False
    
    @monitor_performance("metadata_store", "postgresql")
    async def store_metadata(self, metadata: EntityMetadata) -> bool:
        """Store entity metadata."""
        try:
            data = {
                'entity_id': metadata.entity_id,
                'entity_type': metadata.entity_type,
                'version': metadata.version,
                'in_postgresql': metadata.in_postgresql,
                'in_vector': metadata.in_vector,
                'in_graph': metadata.in_graph,
                'sync_status': json.dumps(metadata.sync_status),
                'sync_errors': json.dumps(metadata.sync_errors),
                'metadata': json.dumps({})
            }
            
            result = await self.postgresql_db.upsert(
                self.table_name, 
                data, 
                ['entity_id']
            )
            
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to store metadata for {metadata.entity_id}: {e}")
            return False
    
    @monitor_performance("metadata_get", "postgresql")
    async def get_metadata(self, entity_id: str) -> Optional[EntityMetadata]:
        """Retrieve entity metadata."""
        try:
            result = await self.postgresql_db.read(
                self.table_name,
                {'entity_id': entity_id}
            )
            
            if result.success and result.data:
                data = result.data[0]
                
                # Parse JSON fields
                sync_status = json.loads(data['sync_status']) if data['sync_status'] else {}
                sync_errors = json.loads(data['sync_errors']) if data['sync_errors'] else []
                
                metadata = EntityMetadata(
                    entity_id=data['entity_id'],
                    entity_type=data['entity_type'],
                    created_at=data['created_at'],
                    updated_at=data['updated_at'],
                    version=data['version'],
                    in_postgresql=data['in_postgresql'],
                    in_vector=data['in_vector'],
                    in_graph=data['in_graph'],
                    sync_status={key: datetime.fromisoformat(value) for key, value in sync_status.items()},
                    sync_errors=sync_errors
                )
                
                return metadata
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get metadata for {entity_id}: {e}")
            return None
    
    @monitor_performance("metadata_update", "postgresql")
    async def update_metadata(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity metadata."""
        try:
            # Handle special JSON fields
            if 'sync_status' in updates and isinstance(updates['sync_status'], dict):
                updates['sync_status'] = json.dumps({
                    key: value.isoformat() if isinstance(value, datetime) else value
                    for key, value in updates['sync_status'].items()
                })
            
            if 'sync_errors' in updates:
                updates['sync_errors'] = json.dumps(updates['sync_errors'])
            
            # Always update the timestamp
            updates['updated_at'] = datetime.utcnow()
            
            result = await self.postgresql_db.update(
                self.table_name,
                updates,
                {'entity_id': entity_id}
            )
            
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to update metadata for {entity_id}: {e}")
            return False
    
    @monitor_performance("metadata_delete", "postgresql")
    async def delete_metadata(self, entity_id: str) -> bool:
        """Delete entity metadata."""
        try:
            result = await self.postgresql_db.delete(
                self.table_name,
                {'entity_id': entity_id}
            )
            
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to delete metadata for {entity_id}: {e}")
            return False
    
    async def list_entities_by_type(self, entity_type: str) -> List[EntityMetadata]:
        """List all entities of a specific type."""
        try:
            result = await self.postgresql_db.read(
                self.table_name,
                {'entity_type': entity_type}
            )
            
            if result.success and result.data:
                entities = []
                for data in result.data:
                    sync_status = json.loads(data['sync_status']) if data['sync_status'] else {}
                    sync_errors = json.loads(data['sync_errors']) if data['sync_errors'] else []
                    
                    metadata = EntityMetadata(
                        entity_id=data['entity_id'],
                        entity_type=data['entity_type'],
                        created_at=data['created_at'],
                        updated_at=data['updated_at'],
                        version=data['version'],
                        in_postgresql=data['in_postgresql'],
                        in_vector=data['in_vector'],
                        in_graph=data['in_graph'],
                        sync_status={key: datetime.fromisoformat(value) for key, value in sync_status.items()},
                        sync_errors=sync_errors
                    )
                    entities.append(metadata)
                
                return entities
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to list entities of type {entity_type}: {e}")
            return []
    
    async def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        try:
            stats_sql = f"""
                SELECT 
                    COUNT(*) as total_entities,
                    COUNT(*) FILTER (WHERE in_postgresql = true) as in_postgresql,
                    COUNT(*) FILTER (WHERE in_vector = true) as in_vector,
                    COUNT(*) FILTER (WHERE in_graph = true) as in_graph,
                    COUNT(*) FILTER (WHERE in_postgresql = true AND in_vector = true AND in_graph = true) as fully_synced,
                    COUNT(*) FILTER (WHERE sync_errors != '[]') as entities_with_errors,
                    AVG(version) as avg_version
                FROM {self.table_name}
            """
            
            result = await self.postgresql_db.execute_query(stats_sql)
            
            if result.success and result.data:
                return result.data[0]
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get sync statistics: {e}")
            return {}


class UnifiedSynchronizationHandler(SynchronizationHandler):
    """Main synchronization handler coordinating all databases."""
    
    def __init__(self, 
                 unified_db: UnifiedDatabase,
                 event_broker: EventBroker,
                 metadata_manager: EntitySyncMetadataManager):
        self.unified_db = unified_db
        self.event_broker = event_broker
        self.metadata_manager = metadata_manager
        
        # Sync configuration
        config = get_config()
        self.eda_enabled = config.sync.eda_enabled
        self.cdc_enabled = config.sync.cdc_enabled
        self.batch_size = config.sync.cdc_batch_size
        
        # Sync rules - define which data goes where
        self.sync_rules = self._initialize_sync_rules()
    
    def _initialize_sync_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize synchronization rules for different entity types."""
        return {
            'user': {
                'postgresql': True,  # Always store in PostgreSQL
                'vector': True,      # Generate embeddings for user profiles
                'graph': True,       # Track user relationships
                'embedding_fields': ['name', 'bio', 'interests'],
                'graph_relationships': ['follows', 'friends', 'collaborates']
            },
            'document': {
                'postgresql': True,  # Metadata and content
                'vector': True,      # Content embeddings for search
                'graph': False,      # Documents don't need graph relationships by default
                'embedding_fields': ['title', 'content', 'tags'],
                'graph_relationships': []
            },
            'product': {
                'postgresql': True,  # Product catalog
                'vector': True,      # Product similarity and recommendations
                'graph': True,       # Product categories and recommendations
                'embedding_fields': ['name', 'description', 'category'],
                'graph_relationships': ['similar_to', 'belongs_to_category', 'bought_together']
            }
        }
    
    @monitor_sync_performance
    async def sync_entity(self, 
                         entity_metadata: EntityMetadata, 
                         source_db: DatabaseType, 
                         target_dbs: List[DatabaseType]) -> bool:
        """Synchronize an entity across databases."""
        try:
            logger.info(
                f"Starting sync for entity {entity_metadata.entity_id}",
                entity_type=entity_metadata.entity_type,
                source_db=source_db.value,
                target_dbs=[db.value for db in target_dbs]
            )
            
            success = True
            
            # Get entity data from source database
            source_data = await self._get_entity_data(entity_metadata.entity_id, source_db)
            if not source_data:
                logger.error(f"Could not retrieve entity data from source: {source_db.value}")
                return False
            
            # Sync to each target database
            for target_db in target_dbs:
                if target_db == source_db:
                    continue
                
                sync_success = await self._sync_to_database(
                    entity_metadata, source_data, target_db
                )
                
                if sync_success:
                    entity_metadata.mark_synced(target_db)
                    logger.debug(f"Successfully synced to {target_db.value}")
                else:
                    success = False
                    entity_metadata.sync_errors.append(
                        f"Failed to sync to {target_db.value} at {datetime.utcnow()}"
                    )
                    logger.error(f"Failed to sync to {target_db.value}")
            
            # Update metadata
            await self.metadata_manager.store_metadata(entity_metadata)
            
            return success
            
        except Exception as e:
            logger.error(f"Entity sync failed for {entity_metadata.entity_id}: {e}")
            return False
    
    async def _get_entity_data(self, entity_id: str, source_db: DatabaseType) -> Optional[Dict[str, Any]]:
        """Retrieve entity data from source database."""
        try:
            if source_db == DatabaseType.POSTGRESQL:
                # Try multiple common table patterns
                for table in ['entities', 'users', 'documents', 'products']:
                    result = await self.unified_db.postgresql.read(table, {'entity_id': entity_id})
                    if result.success and result.data:
                        return result.data[0]
                
            elif source_db == DatabaseType.VECTOR:
                result = await self.unified_db.vector.get_embedding(entity_id)
                if result.success:
                    return result.data
                
            elif source_db == DatabaseType.GRAPH:
                result = await self.unified_db.graph.find_nodes(properties={'entity_id': entity_id})
                if result.success and result.data:
                    return result.data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get entity data from {source_db.value}: {e}")
            return None
    
    async def _sync_to_database(self, 
                               entity_metadata: EntityMetadata, 
                               source_data: Dict[str, Any], 
                               target_db: DatabaseType) -> bool:
        """Sync entity to a specific target database."""
        try:
            entity_type = entity_metadata.entity_type
            sync_rules = self.sync_rules.get(entity_type, {})
            
            if target_db == DatabaseType.POSTGRESQL:
                # Sync to PostgreSQL CRUD
                return await self._sync_to_postgresql(entity_metadata, source_data)
                
            elif target_db == DatabaseType.VECTOR:
                # Generate and store embeddings
                if sync_rules.get('vector', False):
                    return await self._sync_to_vector(entity_metadata, source_data, sync_rules)
                
            elif target_db == DatabaseType.GRAPH:
                # Create graph nodes and relationships
                if sync_rules.get('graph', False):
                    return await self._sync_to_graph(entity_metadata, source_data, sync_rules)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync to {target_db.value}: {e}")
            return False
    
    async def _sync_to_postgresql(self, 
                                 entity_metadata: EntityMetadata, 
                                 source_data: Dict[str, Any]) -> bool:
        """Sync entity to PostgreSQL."""
        try:
            # Determine target table based on entity type
            table_name = f"{entity_metadata.entity_type}s"  # Simple pluralization
            
            # Prepare data for PostgreSQL
            pg_data = source_data.copy()
            pg_data['entity_id'] = entity_metadata.entity_id
            pg_data['entity_type'] = entity_metadata.entity_type
            pg_data['synced_at'] = datetime.utcnow()
            
            # Upsert to PostgreSQL
            result = await self.unified_db.postgresql.upsert(
                table_name, pg_data, ['entity_id']
            )
            
            return result.success
            
        except Exception as e:
            logger.error(f"PostgreSQL sync failed: {e}")
            return False
    
    async def _sync_to_vector(self, 
                             entity_metadata: EntityMetadata, 
                             source_data: Dict[str, Any], 
                             sync_rules: Dict[str, Any]) -> bool:
        """Sync entity to vector database."""
        try:
            # Get fields to embed
            embedding_fields = sync_rules.get('embedding_fields', [])
            
            # Create text for embedding
            text_parts = []
            for field in embedding_fields:
                if field in source_data and source_data[field]:
                    text_parts.append(str(source_data[field]))
            
            if not text_parts:
                logger.warning(f"No text found for embedding entity {entity_metadata.entity_id}")
                return True
            
            text_content = " ".join(text_parts)
            
            # Generate and store embedding
            vector_metadata = {
                'entity_type': entity_metadata.entity_type,
                'text_content': text_content,
                'source_fields': embedding_fields,
                'synced_at': datetime.utcnow().isoformat()
            }
            
            result = await self.unified_db.vector.insert_text_embedding(
                entity_metadata.entity_id, text_content, vector_metadata
            )
            
            return result.success
            
        except Exception as e:
            logger.error(f"Vector sync failed: {e}")
            return False
    
    async def _sync_to_graph(self, 
                            entity_metadata: EntityMetadata, 
                            source_data: Dict[str, Any], 
                            sync_rules: Dict[str, Any]) -> bool:
        """Sync entity to graph database."""
        try:
            # Create node properties
            node_properties = {
                'entity_id': entity_metadata.entity_id,
                'entity_type': entity_metadata.entity_type,
                'synced_at': datetime.utcnow().isoformat()
            }
            
            # Add relevant source data
            for key, value in source_data.items():
                if key not in ['id', '_id', 'embedding'] and value is not None:
                    node_properties[key] = value
            
            # Create or update node
            result = await self.unified_db.graph.create_node(
                entity_metadata.entity_type, node_properties
            )
            
            return result.success
            
        except Exception as e:
            logger.error(f"Graph sync failed: {e}")
            return False
    
    async def handle_conflict(self, entity_id: str, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle synchronization conflicts using last-write-wins strategy."""
        try:
            logger.warning(f"Handling sync conflict for entity {entity_id}")
            
            # Simple last-write-wins resolution
            latest_conflict = max(conflicts, key=lambda x: x.get('timestamp', datetime.min))
            
            # Log conflict resolution
            logger.info(
                f"Conflict resolved for {entity_id}",
                resolution_strategy="last_write_wins",
                winning_source=latest_conflict.get('source_database'),
                conflicts_count=len(conflicts)
            )
            
            return {
                'resolution_strategy': 'last_write_wins',
                'winning_data': latest_conflict,
                'conflicts_resolved': len(conflicts)
            }
            
        except Exception as e:
            logger.error(f"Failed to handle conflict for {entity_id}: {e}")
            return {'error': str(e)}
    
    async def get_sync_status(self, entity_id: str) -> Dict[str, Any]:
        """Get synchronization status for an entity."""
        try:
            metadata = await self.metadata_manager.get_metadata(entity_id)
            
            if metadata:
                return {
                    'entity_id': entity_id,
                    'entity_type': metadata.entity_type,
                    'version': metadata.version,
                    'in_postgresql': metadata.in_postgresql,
                    'in_vector': metadata.in_vector,
                    'in_graph': metadata.in_graph,
                    'last_sync_attempt': metadata.last_sync_attempt,
                    'sync_errors': metadata.sync_errors,
                    'fully_synced': metadata.in_postgresql and metadata.in_vector and metadata.in_graph
                }
            else:
                return {'entity_id': entity_id, 'status': 'not_found'}
                
        except Exception as e:
            logger.error(f"Failed to get sync status for {entity_id}: {e}")
            return {'entity_id': entity_id, 'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_sync_system():
        print("🧠 Testing 3db Synchronization System")
        
        # Create test event
        test_event = SyncEvent(
            event_id=str(uuid.uuid4()),
            event_type=SyncEventType.ENTITY_CREATED,
            entity_id="test_user_123",
            entity_type="user",
            source_database=DatabaseType.POSTGRESQL,
            target_databases=[DatabaseType.VECTOR, DatabaseType.GRAPH],
            data={'name': 'Test User', 'email': 'test@example.com'},
            metadata={'test': True},
            timestamp=datetime.utcnow()
        )
        
        print(f"✅ Created test sync event: {test_event.event_id}")
        print(f"Event type: {test_event.event_type.value}")
        print(f"Source: {test_event.source_database.value}")
        print(f"Targets: {[db.value for db in test_event.target_databases]}")
    
    # Run test
    asyncio.run(test_sync_system())
