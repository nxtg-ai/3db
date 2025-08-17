"""
3db Unified Database Ecosystem - Main Orchestrator

This module provides the main UnifiedDatabase3D class that orchestrates
PostgreSQL CRUD, pgvector, and Apache AGE databases into a cohesive,
intelligent database ecosystem.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

from .core.config import get_config, UnifiedConfig
from .core.base import UnifiedDatabase, DatabaseType, EntityMetadata
from .databases.postgresql.crud import PostgreSQLDatabase
from .databases.vector.embeddings import VectorDatabase
from .databases.graph.relationships import GraphDatabase
from .sync.handler import (
    UnifiedSynchronizationHandler, EventBroker, EntitySyncMetadataManager, SyncEvent, SyncEventType
)
from .query.coordinator import UnifiedQueryInterface, QueryType
from .utils.logging import get_logger, monitor_performance

logger = get_logger("unified_3db")


class Database3D:
    """
    Main orchestrator for the 3db Unified Database Ecosystem.
    
    This class provides the primary interface for interacting with the unified
    database system, automatically coordinating operations across PostgreSQL,
    pgvector, and Apache AGE databases.
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        """Initialize the 3D database ecosystem."""
        
        # Load configuration
        self.config = config or get_config()
        
        # Initialize individual database components
        self.postgresql_db = PostgreSQLDatabase(self.config.postgresql)
        self.vector_db = VectorDatabase(self.config.vector)
        self.graph_db = GraphDatabase(self.config.graph)
        
        # Create unified database wrapper
        self.unified_db = None
        
        # Initialize synchronization components
        self.event_broker = None
        self.metadata_manager = None
        self.sync_handler = None
        
        # Initialize query interface
        self.query_interface = None
        
        # System state
        self.is_initialized = False
        self.is_running = False
        
        # Background tasks
        self._sync_tasks = []
    
    @monitor_performance("3db_initialize", "unified")
    async def initialize(self) -> bool:
        """Initialize the entire 3db ecosystem."""
        try:
            logger.info("🚀 Initializing 3db Unified Database Ecosystem")
            
            # 1. Initialize individual databases
            if not await self._initialize_databases():
                return False
            
            # 2. Create unified database wrapper
            self._create_unified_wrapper()
            
            # 3. Initialize synchronization system
            if not await self._initialize_synchronization():
                return False
            
            # 4. Initialize query interface
            self._initialize_query_interface()
            
            # 5. Start background services
            await self._start_background_services()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("✅ 3db Ecosystem initialized successfully")
            
            # Run initial health check
            health_status = await self.health_check()
            if health_status['overall_health']:
                logger.info("✅ All systems healthy")
            else:
                logger.warning("⚠️ Some systems not fully healthy", health_status=health_status)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize 3db ecosystem: {e}")
            return False
    
    async def _initialize_databases(self) -> bool:
        """Initialize all database connections."""
        logger.info("Initializing database connections...")
        
        # Connect to PostgreSQL
        if not await self.postgresql_db.connect():
            logger.error("Failed to connect to PostgreSQL")
            return False
        logger.info("✅ PostgreSQL connected")
        
        # Connect to Vector DB
        if not await self.vector_db.connect():
            logger.error("Failed to connect to Vector DB")
            return False
        logger.info("✅ Vector DB connected")
        
        # Connect to Graph DB
        if not await self.graph_db.connect():
            logger.error("Failed to connect to Graph DB")
            return False
        logger.info("✅ Graph DB connected")
        
        return True
    
    def _create_unified_wrapper(self) -> None:
        """Create the unified database wrapper."""
        # Initialize metadata manager first
        self.metadata_manager = EntitySyncMetadataManager(self.postgresql_db)
        
        # Create unified database instance (simplified version)
        self.unified_db = UnifiedDatabase(
            postgresql_db=self.postgresql_db,
            vector_db=self.vector_db,
            graph_db=self.graph_db,
            sync_handler=None,  # Will be set after sync initialization
            metadata_manager=self.metadata_manager
        )
        
        logger.info("✅ Unified database wrapper created")
    
    async def _initialize_synchronization(self) -> bool:
        """Initialize the synchronization system."""
        if not self.config.sync.eda_enabled and not self.config.sync.cdc_enabled:
            logger.info("⏭️ Synchronization disabled")
            return True
        
        logger.info("Initializing synchronization system...")
        
        try:
            # Initialize metadata manager tables
            if not await self.metadata_manager.initialize():
                logger.error("Failed to initialize metadata manager")
                return False
            
            # Initialize event broker if EDA is enabled
            if self.config.sync.eda_enabled:
                self.event_broker = EventBroker(self.config.cache.redis_url)
                if not await self.event_broker.connect():
                    logger.error("Failed to connect to event broker")
                    return False
                logger.info("✅ Event broker connected")
            
            # Create synchronization handler
            self.sync_handler = UnifiedSynchronizationHandler(
                self.unified_db,
                self.event_broker,
                self.metadata_manager
            )
            
            # Update unified database with sync handler
            self.unified_db.sync_handler = self.sync_handler
            
            logger.info("✅ Synchronization system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize synchronization: {e}")
            return False
    
    def _initialize_query_interface(self) -> None:
        """Initialize the unified query interface."""
        self.query_interface = UnifiedQueryInterface(self.unified_db)
        logger.info("✅ Query interface initialized")
    
    async def _start_background_services(self) -> None:
        """Start background synchronization services."""
        if not self.config.sync.eda_enabled:
            return
        
        logger.info("Starting background synchronization services...")
        
        # Start event processing task
        sync_task = asyncio.create_task(self._process_sync_events())
        self._sync_tasks.append(sync_task)
        
        logger.info("✅ Background services started")
    
    async def _process_sync_events(self) -> None:
        """Background task to process synchronization events."""
        logger.info("Starting sync event processor")
        
        while self.is_running:
            try:
                if self.event_broker:
                    # Consume events in batches
                    events = await self.event_broker.consume_events(
                        batch_size=self.config.sync.cdc_batch_size
                    )
                    
                    for event in events:
                        success = await self._process_single_event(event)
                        if not success:
                            await self.event_broker.requeue_event(event)
                
                # Wait before next batch
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error in sync event processor: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _process_single_event(self, event: SyncEvent) -> bool:
        """Process a single synchronization event."""
        try:
            logger.debug(f"Processing sync event: {event.event_id}")
            
            # Get entity metadata
            metadata = await self.metadata_manager.get_metadata(event.entity_id)
            if not metadata:
                # Create new metadata
                metadata = EntityMetadata(
                    entity_id=event.entity_id,
                    entity_type=event.entity_type
                )
            
            # Perform synchronization
            success = await self.sync_handler.sync_entity(
                metadata, event.source_database, event.target_databases
            )
            
            if success:
                logger.debug(f"Successfully processed event: {event.event_id}")
            else:
                logger.warning(f"Failed to process event: {event.event_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the 3db ecosystem."""
        logger.info("🛑 Shutting down 3db ecosystem")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self._sync_tasks:
            task.cancel()
        
        if self._sync_tasks:
            await asyncio.gather(*self._sync_tasks, return_exceptions=True)
        
        # Disconnect from event broker
        if self.event_broker:
            await self.event_broker.disconnect()
        
        # Disconnect from databases
        if self.postgresql_db:
            await self.postgresql_db.disconnect()
        
        if self.vector_db:
            await self.vector_db.disconnect()
        
        if self.graph_db:
            await self.graph_db.disconnect()
        
        logger.info("✅ 3db ecosystem shutdown complete")
    
    # High-level unified operations
    
    @monitor_performance("3db_create_entity", "unified")
    async def create_entity(self, 
                           entity_type: str, 
                           entity_data: Dict[str, Any],
                           sync_immediately: bool = True) -> Dict[str, Any]:
        """Create an entity across all relevant databases."""
        try:
            # Generate entity ID if not provided
            entity_id = entity_data.get('entity_id')
            if not entity_id:
                entity_id = f"{entity_type}_{int(datetime.utcnow().timestamp() * 1000)}"
                entity_data['entity_id'] = entity_id
            
            logger.info(f"Creating entity: {entity_id} of type: {entity_type}")
            
            # Create entity metadata
            metadata = EntityMetadata(
                entity_id=entity_id,
                entity_type=entity_type
            )
            
            results = {}
            
            # 1. Create in PostgreSQL (always the source of truth)
            pg_result = await self.postgresql_db.create(f"{entity_type}s", entity_data)
            results['postgresql'] = pg_result.success
            
            if pg_result.success:
                metadata.in_postgresql = True
                metadata.mark_synced(DatabaseType.POSTGRESQL)
            
            # 2. Create vector embedding if applicable
            sync_rules = self.sync_handler.sync_rules.get(entity_type, {}) if self.sync_handler else {}
            
            if sync_rules.get('vector', False):
                embedding_fields = sync_rules.get('embedding_fields', [])
                text_parts = []
                
                for field in embedding_fields:
                    if field in entity_data and entity_data[field]:
                        text_parts.append(str(entity_data[field]))
                
                if text_parts:
                    text_content = " ".join(text_parts)
                    vector_metadata = {
                        'entity_type': entity_type,
                        'text_content': text_content
                    }
                    
                    vector_result = await self.vector_db.insert_text_embedding(
                        entity_id, text_content, vector_metadata
                    )
                    results['vector'] = vector_result.success
                    
                    if vector_result.success:
                        metadata.in_vector = True
                        metadata.mark_synced(DatabaseType.VECTOR)
            
            # 3. Create graph node if applicable
            if sync_rules.get('graph', False):
                node_properties = entity_data.copy()
                node_properties['entity_id'] = entity_id
                node_properties['entity_type'] = entity_type
                
                graph_result = await self.graph_db.create_node(entity_type, node_properties)
                results['graph'] = graph_result.success
                
                if graph_result.success:
                    metadata.in_graph = True
                    metadata.mark_synced(DatabaseType.GRAPH)
            
            # Store metadata
            await self.metadata_manager.store_metadata(metadata)
            
            # Publish sync event if EDA is enabled
            if sync_immediately and self.event_broker:
                sync_event = SyncEvent(
                    event_id=f"create_{entity_id}_{int(datetime.utcnow().timestamp() * 1000)}",
                    event_type=SyncEventType.ENTITY_CREATED,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    source_database=DatabaseType.POSTGRESQL,
                    target_databases=[DatabaseType.VECTOR, DatabaseType.GRAPH],
                    data=entity_data,
                    metadata={'sync_rules': sync_rules},
                    timestamp=datetime.utcnow()
                )
                
                await self.event_broker.publish_event(sync_event)
            
            success = any(results.values())
            
            logger.info(
                f"Entity creation completed: {entity_id}",
                success=success,
                results=results
            )
            
            return {
                'entity_id': entity_id,
                'success': success,
                'results': results,
                'metadata': metadata.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to create entity: {e}")
            return {
                'entity_id': entity_data.get('entity_id', 'unknown'),
                'success': False,
                'error': str(e)
            }
    
    async def search_similar(self, 
                           query_text: str, 
                           entity_type: Optional[str] = None,
                           limit: int = 10,
                           include_relationships: bool = False) -> Dict[str, Any]:
        """Search for similar entities with optional relationship analysis."""
        try:
            result = await self.query_interface.search_similar_content(
                query_text=query_text,
                content_type=entity_type,
                limit=limit,
                include_metadata=True
            )
            
            response = {
                'query': query_text,
                'success': result.success,
                'results': result.data,
                'execution_time': result.total_execution_time
            }
            
            # Add relationship analysis if requested
            if include_relationships and result.success and result.data:
                relationships = {}
                
                if isinstance(result.data, dict) and 'results' in result.data:
                    vector_results = result.data['results'].get('vector', [])
                    
                    for item in vector_results[:5]:  # Limit to top 5 for relationships
                        entity_id = item.get('entity_id')
                        if entity_id:
                            rel_result = await self.query_interface.analyze_relationships(
                                entity_id, max_depth=2
                            )
                            if rel_result.success:
                                relationships[entity_id] = rel_result.data
                
                response['relationships'] = relationships
            
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                'query': query_text,
                'success': False,
                'error': str(e)
            }
    
    async def get_recommendations(self, 
                                user_id: str, 
                                recommendation_type: str = 'content',
                                limit: int = 20) -> Dict[str, Any]:
        """Get personalized recommendations for a user."""
        try:
            result = await self.query_interface.get_recommendations(
                user_id=user_id,
                recommendation_type=recommendation_type,
                limit=limit
            )
            
            return {
                'user_id': user_id,
                'recommendation_type': recommendation_type,
                'success': result.success,
                'recommendations': result.data,
                'execution_time': result.total_execution_time,
                'error': result.error
            }
            
        except Exception as e:
            logger.error(f"Recommendations failed: {e}")
            return {
                'user_id': user_id,
                'success': False,
                'error': str(e)
            }
    
    async def analyze_entity_network(self, 
                                   entity_id: str, 
                                   max_depth: int = 3) -> Dict[str, Any]:
        """Analyze the network of relationships around an entity."""
        try:
            result = await self.query_interface.analyze_relationships(
                entity_id=entity_id,
                max_depth=max_depth,
                include_similarity=True
            )
            
            return {
                'entity_id': entity_id,
                'max_depth': max_depth,
                'success': result.success,
                'network_analysis': result.data,
                'execution_time': result.total_execution_time,
                'error': result.error
            }
            
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            return {
                'entity_id': entity_id,
                'success': False,
                'error': str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all system components."""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_health': True,
            'components': {}
        }
        
        try:
            # Check PostgreSQL
            pg_health = await self.postgresql_db.health_check()
            health_status['components']['postgresql'] = {
                'healthy': pg_health,
                'connected': self.postgresql_db.is_connected
            }
            
            # Check Vector DB
            vector_health = await self.vector_db.health_check()
            health_status['components']['vector'] = {
                'healthy': vector_health,
                'connected': self.vector_db.is_connected,
                'model_loaded': self.vector_db._model_loaded
            }
            
            # Check Graph DB
            graph_health = await self.graph_db.health_check()
            health_status['components']['graph'] = {
                'healthy': graph_health,
                'connected': self.graph_db.is_connected
            }
            
            # Check Event Broker
            if self.event_broker:
                try:
                    queue_stats = await self.event_broker.get_queue_stats()
                    health_status['components']['event_broker'] = {
                        'healthy': True,
                        'queue_stats': queue_stats
                    }
                except Exception:
                    health_status['components']['event_broker'] = {
                        'healthy': False,
                        'error': 'Failed to get queue stats'
                    }
            
            # Check synchronization
            if self.metadata_manager:
                try:
                    sync_stats = await self.metadata_manager.get_sync_statistics()
                    health_status['components']['synchronization'] = {
                        'healthy': True,
                        'statistics': sync_stats
                    }
                except Exception:
                    health_status['components']['synchronization'] = {
                        'healthy': False,
                        'error': 'Failed to get sync stats'
                    }
            
            # Determine overall health
            component_health = [
                comp['healthy'] for comp in health_status['components'].values()
            ]
            health_status['overall_health'] = all(component_health)
            health_status['healthy_components'] = sum(component_health)
            health_status['total_components'] = len(component_health)
            
        except Exception as e:
            health_status['overall_health'] = False
            health_status['error'] = str(e)
        
        return health_status
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'databases': {},
            'synchronization': {},
            'system': {
                'initialized': self.is_initialized,
                'running': self.is_running,
                'active_sync_tasks': len(self._sync_tasks)
            }
        }
        
        try:
            # Database metrics
            if self.postgresql_db:
                metrics['databases']['postgresql'] = await self.postgresql_db.get_metrics()
            
            if self.vector_db:
                metrics['databases']['vector'] = await self.vector_db.get_metrics()
                
                # Add vector-specific stats
                stats_result = await self.vector_db.get_embedding_stats()
                if stats_result.success:
                    metrics['databases']['vector']['embedding_stats'] = stats_result.data
            
            if self.graph_db:
                metrics['databases']['graph'] = await self.graph_db.get_metrics()
                
                # Add graph-specific stats
                stats_result = await self.graph_db.get_graph_statistics()
                if stats_result.success:
                    metrics['databases']['graph']['graph_stats'] = stats_result.data
            
            # Synchronization metrics
            if self.metadata_manager:
                sync_stats = await self.metadata_manager.get_sync_statistics()
                metrics['synchronization'] = sync_stats
            
            if self.event_broker:
                queue_stats = await self.event_broker.get_queue_stats()
                metrics['synchronization']['queue_stats'] = queue_stats
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics


# Convenience factory function
async def create_3db_system(config_path: Optional[str] = None) -> Database3D:
    """Create and initialize a 3db system."""
    
    # Load configuration
    if config_path:
        from .core.config import reload_config
        config = reload_config(config_path)
    else:
        config = get_config()
    
    # Create and initialize system
    db3d = Database3D(config)
    
    if await db3d.initialize():
        logger.info("🎉 3db system ready!")
        return db3d
    else:
        logger.error("💥 Failed to create 3db system")
        raise Exception("Failed to initialize 3db system")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def demo_3db_system():
        """Demonstrate the 3db unified database ecosystem."""
        
        print("🧠 3db Unified Database Ecosystem Demo")
        print("=" * 50)
        
        try:
            # Create 3db system
            db3d = Database3D()
            
            if await db3d.initialize():
                print("✅ 3db system initialized successfully!")
                
                # Health check
                health = await db3d.health_check()
                print(f"🏥 System health: {'✅ Healthy' if health['overall_health'] else '❌ Issues detected'}")
                
                # Create a test entity
                test_entity = {
                    'name': 'Test User',
                    'email': 'test@example.com',
                    'bio': 'A test user for the 3db system demonstration',
                    'interests': 'artificial intelligence, databases, technology'
                }
                
                result = await db3d.create_entity('user', test_entity)
                print(f"👤 Created test entity: {result['entity_id']} (Success: {result['success']})")
                
                # Search for similar content
                search_result = await db3d.search_similar(
                    'artificial intelligence and databases',
                    entity_type='user',
                    limit=5
                )
                print(f"🔍 Search completed: {search_result['success']}")
                
                # Get system metrics
                metrics = await db3d.get_system_metrics()
                print(f"📊 System metrics collected: {len(metrics['databases'])} databases")
                
                # Shutdown
                await db3d.shutdown()
                print("👋 System shutdown complete")
                
            else:
                print("❌ Failed to initialize 3db system")
                
        except Exception as e:
            print(f"💥 Demo failed: {e}")
    
    # Run demo
    asyncio.run(demo_3db_system())
