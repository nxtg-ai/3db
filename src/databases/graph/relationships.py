"""
3db Unified Database Ecosystem - Graph Database Implementation (Apache AGE)

This module provides the Apache AGE implementation for graph operations,
serving as the "neural pathways" component of the unified database brain.
"""

import asyncio
import asyncpg
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
from contextlib import asynccontextmanager

from ..core.base import (
    BaseDatabase, GraphOperations, DatabaseType, QueryResult, EntityMetadata
)
from ..core.config import DatabaseConfig
from ..utils.logging import get_logger, monitor_performance, CircuitBreaker

logger = get_logger("graph_db")


class GraphDatabase(BaseDatabase, GraphOperations):
    """
    Apache AGE implementation for graph operations in the unified ecosystem.
    Handles relationships, graph traversals, and network analysis.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config.dict())
        self.config = config
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)
        
        # Graph-specific settings
        self.graph_name = getattr(config, 'graph_name', 'unified_graph')
        self.max_traversal_depth = getattr(config, 'max_traversal_depth', 10)
        
        # AGE configuration
        self.age_schema = 'ag_catalog'
    
    def _get_database_type(self) -> DatabaseType:
        """Return Graph database type."""
        return DatabaseType.GRAPH
    
    @monitor_performance("graph_connect", "graph")
    async def connect(self) -> bool:
        """Establish Apache AGE connection pool and setup graph."""
        try:
            # Connection pool configuration
            pool_config = {
                'host': self.config.host,
                'port': self.config.port,
                'database': self.config.database,
                'user': self.config.username,
                'password': self.config.password,
                'min_size': self.config.pool_min_size,
                'max_size': self.config.pool_max_size,
                'command_timeout': self.config.pool_timeout,
            }
            
            self.connection_pool = await asyncpg.create_pool(**pool_config)
            
            # Setup Apache AGE extension and graph
            async with self.connection_pool.acquire() as conn:
                # Load AGE extension
                await conn.execute('CREATE EXTENSION IF NOT EXISTS age')
                
                # Load AGE into search path
                await conn.execute('LOAD \'age\'')
                
                # Set search path
                await conn.execute('SET search_path = ag_catalog, "$user", public')
                
                # Create graph if it doesn't exist
                try:
                    await conn.execute(f"SELECT create_graph('{self.graph_name}')")
                except Exception as e:
                    # Graph might already exist
                    if "already exists" not in str(e):
                        logger.warning(f"Graph creation warning: {e}")
                
                # Test basic graph functionality
                await conn.execute('SELECT 1')
            
            # Initialize graph metadata tables
            await self._initialize_graph_metadata()
            
            self.is_connected = True
            self._metrics['connections_opened'] += 1
            
            logger.info(
                "Apache AGE connection pool established",
                host=self.config.host,
                database=self.config.database,
                graph_name=self.graph_name
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Apache AGE: {e}")
            self.is_connected = False
            return False
    
    async def _initialize_graph_metadata(self) -> None:
        """Initialize metadata tables for graph management."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Create node metadata table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS graph_nodes_metadata (
                        id SERIAL PRIMARY KEY,
                        node_id BIGINT UNIQUE NOT NULL,
                        entity_id VARCHAR(255) UNIQUE NOT NULL,
                        label VARCHAR(100) NOT NULL,
                        properties JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create edge metadata table
                await conn.execute("""
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
                    )
                """)
                
                # Create indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS graph_nodes_entity_id_idx 
                    ON graph_nodes_metadata (entity_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS graph_nodes_label_idx 
                    ON graph_nodes_metadata (label)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS graph_edges_relationship_type_idx 
                    ON graph_edges_metadata (relationship_type)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS graph_edges_from_entity_idx 
                    ON graph_edges_metadata (from_entity_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS graph_edges_to_entity_idx 
                    ON graph_edges_metadata (to_entity_id)
                """)
                
                logger.info("Graph metadata tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize graph metadata tables: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Apache AGE connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            self.connection_pool = None
            self.is_connected = False
            self._metrics['connections_closed'] += 1
            logger.info("Apache AGE connection pool closed")
    
    @monitor_performance("graph_health_check", "graph")
    async def health_check(self) -> bool:
        """Perform Apache AGE health check."""
        if not self.connection_pool:
            return False
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Test AGE functionality
                result = await conn.fetchval(f"""
                    SELECT * FROM cypher('{self.graph_name}', $$
                        RETURN 1 as test
                    $$) as (test agtype)
                """)
                return result is not None
        except Exception as e:
            logger.error(f"Apache AGE health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with proper error handling."""
        if not self.connection_pool:
            raise Exception("Database not connected")
        
        conn = None
        try:
            conn = await self.connection_pool.acquire()
            # Set up AGE environment for each connection
            await conn.execute('LOAD \'age\'')
            await conn.execute('SET search_path = ag_catalog, "$user", public')
            yield conn
        finally:
            if conn:
                await self.connection_pool.release(conn)
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute an Apache AGE query and return standardized result."""
        start_time = datetime.utcnow()
        
        try:
            async def _execute():
                async with self.get_connection() as conn:
                    if params:
                        param_values = list(params.values())
                        formatted_query = query
                        for i, key in enumerate(params.keys(), 1):
                            formatted_query = formatted_query.replace(f":{key}", f"${i}")
                        result = await conn.fetch(formatted_query, *param_values)
                    else:
                        result = await conn.fetch(query)
                    return result
            
            raw_result = await self.circuit_breaker.call(_execute)
            data = [dict(record) for record in raw_result] if raw_result else []
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(execution_time, success=True)
            
            return QueryResult(
                success=True,
                data=data,
                metadata={'rows_returned': len(data), 'execution_time_ms': execution_time * 1000},
                execution_time=execution_time,
                source_database=DatabaseType.GRAPH,
                affected_rows=len(data)
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(execution_time, success=False)
            
            error_msg = f"Apache AGE query failed: {e}"
            logger.error(error_msg, query=query[:100])
            
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=execution_time,
                source_database=DatabaseType.GRAPH
            )
    
    async def execute_cypher(self, cypher_query: str, return_columns: Optional[List[str]] = None) -> QueryResult:
        """Execute a Cypher query through Apache AGE."""
        try:
            # Build column specification for the query
            if return_columns:
                columns_spec = ", ".join([f"{col} agtype" for col in return_columns])
            else:
                columns_spec = "result agtype"
            
            # Wrap Cypher query in AGE SQL syntax
            age_query = f"""
                SELECT * FROM cypher('{self.graph_name}', $$
                    {cypher_query}
                $$) as ({columns_spec})
            """
            
            return await self.execute_query(age_query)
            
        except Exception as e:
            error_msg = f"Cypher query execution failed: {e}"
            logger.error(error_msg, cypher_query=cypher_query)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    # Graph Operations Implementation
    
    @monitor_performance("graph_create_node", "graph")
    async def create_node(self, label: str, properties: Dict[str, Any]) -> QueryResult:
        """Create a new node in the graph."""
        try:
            entity_id = properties.get('entity_id')
            if not entity_id:
                entity_id = f"{label}_{datetime.utcnow().timestamp()}"
                properties['entity_id'] = entity_id
            
            # Convert properties to AGE format
            props_str = json.dumps(properties).replace('"', '\\"')
            
            # Create node using Cypher
            cypher_query = f"""
                CREATE (n:{label} {{{self._dict_to_cypher_props(properties)}}})
                RETURN n
            """
            
            result = await self.execute_cypher(cypher_query, ['n'])
            
            if result.success and result.data:
                # Extract node ID from AGE result
                node_data = result.data[0]
                node_agtype = node_data.get('n')
                
                # Parse AGE node result to get internal ID
                node_id = self._extract_node_id_from_agtype(node_agtype)
                
                # Store metadata
                await self._store_node_metadata(node_id, entity_id, label, properties)
                
                return QueryResult(
                    success=True,
                    data={
                        'node_id': node_id,
                        'entity_id': entity_id,
                        'label': label,
                        'properties': properties
                    },
                    metadata={'operation': 'create_node', 'label': label},
                    source_database=DatabaseType.GRAPH,
                    affected_rows=1
                )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to create node with label {label}: {e}"
            logger.error(error_msg, label=label, properties=properties)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    @monitor_performance("graph_create_edge", "graph")
    async def create_edge(self, 
                         from_node_id: str, 
                         to_node_id: str, 
                         relationship_type: str, 
                         properties: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Create a relationship between two nodes."""
        try:
            properties = properties or {}
            relationship_id = f"{from_node_id}_{relationship_type}_{to_node_id}_{datetime.utcnow().timestamp()}"
            properties['relationship_id'] = relationship_id
            
            # Find nodes by entity_id
            cypher_query = f"""
                MATCH (a {{entity_id: '{from_node_id}'}}), (b {{entity_id: '{to_node_id}'}})
                CREATE (a)-[r:{relationship_type} {{{self._dict_to_cypher_props(properties)}}}]->(b)
                RETURN r
            """
            
            result = await self.execute_cypher(cypher_query, ['r'])
            
            if result.success and result.data:
                # Extract edge ID from AGE result
                edge_data = result.data[0]
                edge_agtype = edge_data.get('r')
                edge_id = self._extract_edge_id_from_agtype(edge_agtype)
                
                # Store metadata
                await self._store_edge_metadata(
                    edge_id, relationship_id, from_node_id, to_node_id, 
                    relationship_type, properties
                )
                
                return QueryResult(
                    success=True,
                    data={
                        'edge_id': edge_id,
                        'relationship_id': relationship_id,
                        'from_entity_id': from_node_id,
                        'to_entity_id': to_node_id,
                        'relationship_type': relationship_type,
                        'properties': properties
                    },
                    metadata={'operation': 'create_edge', 'relationship_type': relationship_type},
                    source_database=DatabaseType.GRAPH,
                    affected_rows=1
                )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to create edge from {from_node_id} to {to_node_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    @monitor_performance("graph_find_nodes", "graph")
    async def find_nodes(self, 
                        label: Optional[str] = None, 
                        properties: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Find nodes matching criteria."""
        try:
            # Build Cypher query
            if label:
                node_pattern = f"(n:{label}"
            else:
                node_pattern = "(n"
            
            if properties:
                props_str = self._dict_to_cypher_props(properties)
                node_pattern += f" {{{props_str}}}"
            
            node_pattern += ")"
            
            cypher_query = f"""
                MATCH {node_pattern}
                RETURN n
            """
            
            result = await self.execute_cypher(cypher_query, ['n'])
            
            if result.success:
                # Process AGE node results
                processed_data = []
                for row in result.data:
                    node_agtype = row.get('n')
                    node_data = self._parse_agtype_node(node_agtype)
                    processed_data.append(node_data)
                
                return QueryResult(
                    success=True,
                    data=processed_data,
                    metadata={'operation': 'find_nodes', 'label': label, 'filters': properties},
                    source_database=DatabaseType.GRAPH,
                    affected_rows=len(processed_data)
                )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to find nodes: {e}"
            logger.error(error_msg, label=label, properties=properties)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    @monitor_performance("graph_find_relationships", "graph")
    async def find_relationships(self, 
                               node_id: str, 
                               relationship_type: Optional[str] = None, 
                               direction: str = "both") -> QueryResult:
        """Find relationships for a node."""
        try:
            # Build relationship pattern based on direction
            if direction == "outgoing":
                if relationship_type:
                    rel_pattern = f"-[r:{relationship_type}]->"
                else:
                    rel_pattern = "-[r]->"
            elif direction == "incoming":
                if relationship_type:
                    rel_pattern = f"<-[r:{relationship_type}]-"
                else:
                    rel_pattern = "<-[r]-"
            else:  # both
                if relationship_type:
                    rel_pattern = f"-[r:{relationship_type}]-"
                else:
                    rel_pattern = "-[r]-"
            
            cypher_query = f"""
                MATCH (n {{entity_id: '{node_id}'}}){rel_pattern}(m)
                RETURN r, m
            """
            
            result = await self.execute_cypher(cypher_query, ['r', 'm'])
            
            if result.success:
                # Process AGE relationship results
                processed_data = []
                for row in result.data:
                    rel_agtype = row.get('r')
                    node_agtype = row.get('m')
                    
                    rel_data = self._parse_agtype_relationship(rel_agtype)
                    node_data = self._parse_agtype_node(node_agtype)
                    
                    processed_data.append({
                        'relationship': rel_data,
                        'connected_node': node_data
                    })
                
                return QueryResult(
                    success=True,
                    data=processed_data,
                    metadata={
                        'operation': 'find_relationships',
                        'node_id': node_id,
                        'relationship_type': relationship_type,
                        'direction': direction
                    },
                    source_database=DatabaseType.GRAPH,
                    affected_rows=len(processed_data)
                )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to find relationships for node {node_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    @monitor_performance("graph_traversal", "graph")
    async def graph_traversal(self, 
                             start_node_id: str, 
                             max_depth: int = 3, 
                             filters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Perform graph traversal from a starting node."""
        try:
            max_depth = min(max_depth, self.max_traversal_depth)
            
            # Build traversal query
            cypher_query = f"""
                MATCH path = (start {{entity_id: '{start_node_id}'}})-[*1..{max_depth}]-(end)
                RETURN path, length(path) as depth
                ORDER BY depth
            """
            
            result = await self.execute_cypher(cypher_query, ['path', 'depth'])
            
            if result.success:
                # Process path results
                processed_data = []
                for row in result.data:
                    path_agtype = row.get('path')
                    depth = row.get('depth')
                    
                    path_data = self._parse_agtype_path(path_agtype)
                    path_data['depth'] = depth
                    
                    processed_data.append(path_data)
                
                return QueryResult(
                    success=True,
                    data=processed_data,
                    metadata={
                        'operation': 'graph_traversal',
                        'start_node_id': start_node_id,
                        'max_depth': max_depth,
                        'filters': filters
                    },
                    source_database=DatabaseType.GRAPH,
                    affected_rows=len(processed_data)
                )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to perform graph traversal from {start_node_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    @monitor_performance("graph_shortest_path", "graph")
    async def shortest_path(self, from_node_id: str, to_node_id: str) -> QueryResult:
        """Find shortest path between two nodes."""
        try:
            cypher_query = f"""
                MATCH path = shortestPath((start {{entity_id: '{from_node_id}'}})-[*]-(end {{entity_id: '{to_node_id}'}}))
                RETURN path, length(path) as path_length
            """
            
            result = await self.execute_cypher(cypher_query, ['path', 'path_length'])
            
            if result.success and result.data:
                path_data = result.data[0]
                path_agtype = path_data.get('path')
                path_length = path_data.get('path_length')
                
                if path_agtype:
                    parsed_path = self._parse_agtype_path(path_agtype)
                    parsed_path['path_length'] = path_length
                    
                    return QueryResult(
                        success=True,
                        data=parsed_path,
                        metadata={
                            'operation': 'shortest_path',
                            'from_node_id': from_node_id,
                            'to_node_id': to_node_id
                        },
                        source_database=DatabaseType.GRAPH,
                        affected_rows=1
                    )
                else:
                    return QueryResult(
                        success=False,
                        data=None,
                        error=f"No path found between {from_node_id} and {to_node_id}",
                        source_database=DatabaseType.GRAPH
                    )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to find shortest path between {from_node_id} and {to_node_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    @monitor_performance("graph_delete_node", "graph")
    async def delete_node(self, node_id: str) -> QueryResult:
        """Delete a node and its relationships."""
        try:
            cypher_query = f"""
                MATCH (n {{entity_id: '{node_id}'}})
                DETACH DELETE n
                RETURN count(n) as deleted_count
            """
            
            result = await self.execute_cypher(cypher_query, ['deleted_count'])
            
            if result.success:
                # Clean up metadata
                await self._delete_node_metadata(node_id)
                
                return QueryResult(
                    success=True,
                    data={'deleted_count': result.data[0].get('deleted_count', 0) if result.data else 0},
                    metadata={'operation': 'delete_node', 'node_id': node_id},
                    source_database=DatabaseType.GRAPH,
                    affected_rows=1
                )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to delete node {node_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    @monitor_performance("graph_delete_relationship", "graph")
    async def delete_relationship(self, relationship_id: str) -> QueryResult:
        """Delete a specific relationship."""
        try:
            cypher_query = f"""
                MATCH ()-[r {{relationship_id: '{relationship_id}'}}]-()
                DELETE r
                RETURN count(r) as deleted_count
            """
            
            result = await self.execute_cypher(cypher_query, ['deleted_count'])
            
            if result.success:
                # Clean up metadata
                await self._delete_edge_metadata(relationship_id)
                
                return QueryResult(
                    success=True,
                    data={'deleted_count': result.data[0].get('deleted_count', 0) if result.data else 0},
                    metadata={'operation': 'delete_relationship', 'relationship_id': relationship_id},
                    source_database=DatabaseType.GRAPH,
                    affected_rows=1
                )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to delete relationship {relationship_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    # Helper methods for AGE data parsing and formatting
    
    def _dict_to_cypher_props(self, properties: Dict[str, Any]) -> str:
        """Convert Python dict to Cypher properties format."""
        if not properties:
            return ""
        
        props = []
        for key, value in properties.items():
            if isinstance(value, str):
                props.append(f"{key}: '{value}'")
            elif isinstance(value, (int, float)):
                props.append(f"{key}: {value}")
            elif isinstance(value, bool):
                props.append(f"{key}: {str(value).lower()}")
            else:
                props.append(f"{key}: '{json.dumps(value)}'")
        
        return ", ".join(props)
    
    def _extract_node_id_from_agtype(self, agtype_data: Any) -> Optional[int]:
        """Extract node ID from AGE agtype result."""
        try:
            # AGE returns complex data structures, parse accordingly
            # This is a simplified implementation
            if isinstance(agtype_data, str):
                # Parse JSON-like string from AGE
                import re
                match = re.search(r'"id":\s*(\d+)', agtype_data)
                if match:
                    return int(match.group(1))
            return None
        except Exception:
            return None
    
    def _extract_edge_id_from_agtype(self, agtype_data: Any) -> Optional[int]:
        """Extract edge ID from AGE agtype result."""
        try:
            # Similar to node ID extraction
            if isinstance(agtype_data, str):
                import re
                match = re.search(r'"id":\s*(\d+)', agtype_data)
                if match:
                    return int(match.group(1))
            return None
        except Exception:
            return None
    
    def _parse_agtype_node(self, agtype_data: Any) -> Dict[str, Any]:
        """Parse AGE node data to standard format."""
        try:
            # Simplified AGE node parsing
            if isinstance(agtype_data, str):
                # This would need proper AGE parsing logic
                return {'raw_data': agtype_data}
            return {'data': agtype_data}
        except Exception:
            return {'error': 'Failed to parse node data'}
    
    def _parse_agtype_relationship(self, agtype_data: Any) -> Dict[str, Any]:
        """Parse AGE relationship data to standard format."""
        try:
            # Simplified AGE relationship parsing
            if isinstance(agtype_data, str):
                return {'raw_data': agtype_data}
            return {'data': agtype_data}
        except Exception:
            return {'error': 'Failed to parse relationship data'}
    
    def _parse_agtype_path(self, agtype_data: Any) -> Dict[str, Any]:
        """Parse AGE path data to standard format."""
        try:
            # Simplified AGE path parsing
            if isinstance(agtype_data, str):
                return {'raw_path': agtype_data}
            return {'path_data': agtype_data}
        except Exception:
            return {'error': 'Failed to parse path data'}
    
    # Metadata management methods
    
    async def _store_node_metadata(self, node_id: int, entity_id: str, label: str, properties: Dict[str, Any]) -> None:
        """Store node metadata in metadata table."""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO graph_nodes_metadata (node_id, entity_id, label, properties)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (entity_id) 
                    DO UPDATE SET 
                        node_id = EXCLUDED.node_id,
                        label = EXCLUDED.label,
                        properties = EXCLUDED.properties,
                        updated_at = CURRENT_TIMESTAMP
                """, node_id, entity_id, label, json.dumps(properties))
        except Exception as e:
            logger.error(f"Failed to store node metadata: {e}")
    
    async def _store_edge_metadata(self, edge_id: int, relationship_id: str, 
                                  from_entity_id: str, to_entity_id: str, 
                                  relationship_type: str, properties: Dict[str, Any]) -> None:
        """Store edge metadata in metadata table."""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO graph_edges_metadata 
                    (edge_id, relationship_id, from_entity_id, to_entity_id, relationship_type, properties)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (relationship_id) 
                    DO UPDATE SET 
                        edge_id = EXCLUDED.edge_id,
                        from_entity_id = EXCLUDED.from_entity_id,
                        to_entity_id = EXCLUDED.to_entity_id,
                        relationship_type = EXCLUDED.relationship_type,
                        properties = EXCLUDED.properties,
                        updated_at = CURRENT_TIMESTAMP
                """, edge_id, relationship_id, from_entity_id, to_entity_id, 
                     relationship_type, json.dumps(properties))
        except Exception as e:
            logger.error(f"Failed to store edge metadata: {e}")
    
    async def _delete_node_metadata(self, entity_id: str) -> None:
        """Delete node metadata."""
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    "DELETE FROM graph_nodes_metadata WHERE entity_id = $1",
                    entity_id
                )
        except Exception as e:
            logger.error(f"Failed to delete node metadata: {e}")
    
    async def _delete_edge_metadata(self, relationship_id: str) -> None:
        """Delete edge metadata."""
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    "DELETE FROM graph_edges_metadata WHERE relationship_id = $1",
                    relationship_id
                )
        except Exception as e:
            logger.error(f"Failed to delete edge metadata: {e}")
    
    # Advanced graph analytics
    
    async def get_graph_statistics(self) -> QueryResult:
        """Get statistics about the graph."""
        try:
            cypher_query = """
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                RETURN 
                    count(DISTINCT n) as node_count,
                    count(DISTINCT r) as edge_count,
                    count(DISTINCT labels(n)) as label_count
            """
            
            result = await self.execute_cypher(cypher_query, ['node_count', 'edge_count', 'label_count'])
            
            if result.success and result.data:
                stats = result.data[0]
                return QueryResult(
                    success=True,
                    data=stats,
                    metadata={'operation': 'get_graph_statistics'},
                    source_database=DatabaseType.GRAPH,
                    affected_rows=1
                )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to get graph statistics: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )
    
    async def find_influential_nodes(self, limit: int = 10) -> QueryResult:
        """Find nodes with the highest degree centrality."""
        try:
            cypher_query = f"""
                MATCH (n)-[r]-()
                RETURN n, count(r) as degree
                ORDER BY degree DESC
                LIMIT {limit}
            """
            
            result = await self.execute_cypher(cypher_query, ['n', 'degree'])
            
            if result.success:
                processed_data = []
                for row in result.data:
                    node_data = self._parse_agtype_node(row.get('n'))
                    node_data['degree'] = row.get('degree')
                    processed_data.append(node_data)
                
                return QueryResult(
                    success=True,
                    data=processed_data,
                    metadata={'operation': 'find_influential_nodes', 'limit': limit},
                    source_database=DatabaseType.GRAPH,
                    affected_rows=len(processed_data)
                )
            else:
                return result
                
        except Exception as e:
            error_msg = f"Failed to find influential nodes: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.GRAPH
            )


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ..core.config import DatabaseConfig
    
    async def test_graph_db():
        # Example configuration
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_3db_graph",
            username="postgres",
            password="password",
            graph_name="test_graph",
            max_traversal_depth=5
        )
        
        db = GraphDatabase(config)
        
        # Test connection
        if await db.connect():
            print("✅ Apache AGE connection successful")
            
            # Test health check
            if await db.health_check():
                print("✅ Apache AGE health check passed")
            
            await db.disconnect()
        else:
            print("❌ Apache AGE connection failed")
    
    # Run test
    asyncio.run(test_graph_db())
