"""
3db Unified Database Ecosystem - Query Coordination System

This module provides federated query execution across PostgreSQL CRUD, 
pgvector, and Apache AGE databases, enabling intelligent cross-database operations.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import json
import time

from ..core.base import (
    UnifiedDatabase, DatabaseType, QueryResult
)
from ..utils.logging import get_logger, monitor_performance, timed_operation

logger = get_logger("query_coordinator")


class QueryType(Enum):
    """Types of unified queries."""
    SIMPLE_READ = "simple_read"           # Single database query
    FEDERATED_JOIN = "federated_join"     # Cross-database joins
    SIMILARITY_SEARCH = "similarity_search"  # Vector similarity with metadata
    GRAPH_TRAVERSAL = "graph_traversal"   # Graph analysis with data enrichment
    HYBRID_ANALYTICS = "hybrid_analytics" # Complex multi-database analytics
    RECOMMENDATION = "recommendation"     # AI-powered recommendations


class QueryStrategy(Enum):
    """Query execution strategies."""
    SEQUENTIAL = "sequential"       # Execute queries one after another
    PARALLEL = "parallel"          # Execute queries in parallel
    OPTIMIZED = "optimized"        # Smart execution based on data dependencies
    MATERIALIZED = "materialized"  # Use pre-computed materialized views


@dataclass
class QueryPlan:
    """Query execution plan for federated operations."""
    
    query_id: str
    query_type: QueryType
    strategy: QueryStrategy
    estimated_cost: float
    
    # Database operations
    postgresql_ops: List[Dict[str, Any]] = field(default_factory=list)
    vector_ops: List[Dict[str, Any]] = field(default_factory=list)
    graph_ops: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution metadata
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    expected_result_size: int = 0
    timeout_seconds: int = 30


@dataclass
class UnifiedQueryResult:
    """Result from federated query execution."""
    
    query_id: str
    success: bool
    data: Any = None
    
    # Individual database results
    postgresql_result: Optional[QueryResult] = None
    vector_result: Optional[QueryResult] = None
    graph_result: Optional[QueryResult] = None
    
    # Execution metadata
    total_execution_time: float = 0.0
    database_execution_times: Dict[str, float] = field(default_factory=dict)
    rows_processed: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class QueryPlanner:
    """Plans optimal execution strategy for federated queries."""
    
    def __init__(self):
        self.cost_estimates = {
            DatabaseType.POSTGRESQL: 1.0,  # Base cost multiplier
            DatabaseType.VECTOR: 2.0,      # Vector ops are more expensive
            DatabaseType.GRAPH: 3.0        # Graph traversals are most expensive
        }
    
    def create_plan(self, 
                   query_spec: Dict[str, Any], 
                   query_type: QueryType = QueryType.SIMPLE_READ) -> QueryPlan:
        """Create an execution plan for a federated query."""
        
        query_plan = QueryPlan(
            query_id=f"query_{int(time.time() * 1000)}",
            query_type=query_type,
            strategy=self._determine_strategy(query_spec, query_type),
            estimated_cost=0.0
        )
        
        # Plan database operations based on query type
        if query_type == QueryType.SIMILARITY_SEARCH:
            query_plan = self._plan_similarity_search(query_spec, query_plan)
        elif query_type == QueryType.GRAPH_TRAVERSAL:
            query_plan = self._plan_graph_traversal(query_spec, query_plan)
        elif query_type == QueryType.FEDERATED_JOIN:
            query_plan = self._plan_federated_join(query_spec, query_plan)
        elif query_type == QueryType.RECOMMENDATION:
            query_plan = self._plan_recommendation(query_spec, query_plan)
        else:
            query_plan = self._plan_simple_query(query_spec, query_plan)
        
        # Calculate estimated cost
        query_plan.estimated_cost = self._calculate_cost(query_plan)
        
        logger.debug(
            f"Created query plan {query_plan.query_id}",
            query_type=query_type.value,
            strategy=query_plan.strategy.value,
            estimated_cost=query_plan.estimated_cost
        )
        
        return query_plan
    
    def _determine_strategy(self, query_spec: Dict[str, Any], query_type: QueryType) -> QueryStrategy:
        """Determine optimal execution strategy."""
        
        # Simple heuristics for strategy selection
        databases_involved = len([
            db for db in ['postgresql', 'vector', 'graph'] 
            if db in query_spec
        ])
        
        if databases_involved == 1:
            return QueryStrategy.SEQUENTIAL
        elif query_type in [QueryType.SIMILARITY_SEARCH, QueryType.RECOMMENDATION]:
            return QueryStrategy.OPTIMIZED
        elif databases_involved > 2:
            return QueryStrategy.PARALLEL
        else:
            return QueryStrategy.SEQUENTIAL
    
    def _plan_similarity_search(self, query_spec: Dict[str, Any], plan: QueryPlan) -> QueryPlan:
        """Plan vector similarity search with metadata enrichment."""
        
        # Vector operation - similarity search
        plan.vector_ops.append({
            'operation': 'similarity_search',
            'query_text': query_spec.get('query_text', ''),
            'query_embedding': query_spec.get('query_embedding'),
            'limit': query_spec.get('limit', 10),
            'threshold': query_spec.get('threshold', 0.8),
            'entity_type': query_spec.get('entity_type')
        })
        
        # PostgreSQL operation - enrich with metadata
        plan.postgresql_ops.append({
            'operation': 'enrich_metadata',
            'depends_on': 'vector_similarity_results',
            'join_column': 'entity_id'
        })
        
        # Set dependencies
        plan.dependencies['postgresql_enrich'] = ['vector_similarity']
        plan.strategy = QueryStrategy.OPTIMIZED
        
        return plan
    
    def _plan_graph_traversal(self, query_spec: Dict[str, Any], plan: QueryPlan) -> QueryPlan:
        """Plan graph traversal with data enrichment."""
        
        # Graph operation - traversal
        plan.graph_ops.append({
            'operation': 'graph_traversal',
            'start_node_id': query_spec.get('start_node_id'),
            'max_depth': query_spec.get('max_depth', 3),
            'relationship_types': query_spec.get('relationship_types'),
            'filters': query_spec.get('filters')
        })
        
        # PostgreSQL operation - get detailed entity data
        plan.postgresql_ops.append({
            'operation': 'get_entity_details',
            'depends_on': 'graph_traversal_results',
            'entity_ids': 'from_graph_result'
        })
        
        # Vector operation - similarity for recommendations (optional)
        if query_spec.get('include_similarity', False):
            plan.vector_ops.append({
                'operation': 'batch_similarity',
                'depends_on': 'graph_traversal_results',
                'entity_ids': 'from_graph_result',
                'limit': query_spec.get('similarity_limit', 5)
            })
        
        plan.dependencies['postgresql_details'] = ['graph_traversal']
        if query_spec.get('include_similarity', False):
            plan.dependencies['vector_similarity'] = ['graph_traversal']
        
        return plan
    
    def _plan_federated_join(self, query_spec: Dict[str, Any], plan: QueryPlan) -> QueryPlan:
        """Plan cross-database joins."""
        
        # Execute queries on each database
        if 'postgresql_query' in query_spec:
            plan.postgresql_ops.append({
                'operation': 'execute_query',
                'query': query_spec['postgresql_query'],
                'params': query_spec.get('postgresql_params')
            })
        
        if 'vector_query' in query_spec:
            plan.vector_ops.append({
                'operation': 'execute_query',
                'query': query_spec['vector_query'],
                'params': query_spec.get('vector_params')
            })
        
        if 'graph_query' in query_spec:
            plan.graph_ops.append({
                'operation': 'execute_cypher',
                'query': query_spec['graph_query'],
                'params': query_spec.get('graph_params')
            })
        
        # Join strategy
        plan.strategy = QueryStrategy.PARALLEL
        
        return plan
    
    def _plan_recommendation(self, query_spec: Dict[str, Any], plan: QueryPlan) -> QueryPlan:
        """Plan AI-powered recommendation query."""
        
        user_id = query_spec.get('user_id')
        recommendation_type = query_spec.get('type', 'content')
        
        # 1. Get user profile from PostgreSQL
        plan.postgresql_ops.append({
            'operation': 'get_user_profile',
            'user_id': user_id,
            'include_preferences': True
        })
        
        # 2. Get user's vector embedding
        plan.vector_ops.append({
            'operation': 'get_user_embedding',
            'entity_id': user_id
        })
        
        # 3. Find similar users via graph relationships
        plan.graph_ops.append({
            'operation': 'find_similar_users',
            'user_id': user_id,
            'relationship_types': ['follows', 'likes', 'shares'],
            'max_depth': 2
        })
        
        # 4. Generate recommendations based on similarity
        plan.vector_ops.append({
            'operation': 'content_recommendations',
            'depends_on': ['user_embedding', 'similar_users'],
            'content_type': recommendation_type,
            'limit': query_spec.get('limit', 20)
        })
        
        # Set complex dependencies
        plan.dependencies = {
            'user_embedding': ['user_profile'],
            'similar_users': ['user_profile'],
            'content_recommendations': ['user_embedding', 'similar_users']
        }
        
        plan.strategy = QueryStrategy.OPTIMIZED
        
        return plan
    
    def _plan_simple_query(self, query_spec: Dict[str, Any], plan: QueryPlan) -> QueryPlan:
        """Plan simple single-database query."""
        
        target_db = query_spec.get('database', 'postgresql')
        
        if target_db == 'postgresql':
            plan.postgresql_ops.append({
                'operation': 'execute_query',
                'query': query_spec.get('query'),
                'params': query_spec.get('params')
            })
        elif target_db == 'vector':
            plan.vector_ops.append({
                'operation': 'execute_query',
                'query': query_spec.get('query'),
                'params': query_spec.get('params')
            })
        elif target_db == 'graph':
            plan.graph_ops.append({
                'operation': 'execute_cypher',
                'query': query_spec.get('query'),
                'params': query_spec.get('params')
            })
        
        plan.strategy = QueryStrategy.SEQUENTIAL
        
        return plan
    
    def _calculate_cost(self, plan: QueryPlan) -> float:
        """Calculate estimated execution cost."""
        total_cost = 0.0
        
        # Cost for each database operation
        total_cost += len(plan.postgresql_ops) * self.cost_estimates[DatabaseType.POSTGRESQL]
        total_cost += len(plan.vector_ops) * self.cost_estimates[DatabaseType.VECTOR]
        total_cost += len(plan.graph_ops) * self.cost_estimates[DatabaseType.GRAPH]
        
        # Penalty for complex dependencies
        dependency_penalty = len(plan.dependencies) * 0.5
        total_cost += dependency_penalty
        
        return total_cost


class QueryExecutor:
    """Executes federated queries according to query plans."""
    
    def __init__(self, unified_db: UnifiedDatabase):
        self.unified_db = unified_db
        self.active_queries: Dict[str, UnifiedQueryResult] = {}
    
    @monitor_performance("execute_federated_query", "unified")
    async def execute_query(self, query_plan: QueryPlan) -> UnifiedQueryResult:
        """Execute a federated query according to the plan."""
        
        result = UnifiedQueryResult(
            query_id=query_plan.query_id,
            success=False
        )
        
        start_time = time.time()
        
        try:
            self.active_queries[query_plan.query_id] = result
            
            logger.info(
                f"Executing federated query {query_plan.query_id}",
                query_type=query_plan.query_type.value,
                strategy=query_plan.strategy.value
            )
            
            # Execute based on strategy
            if query_plan.strategy == QueryStrategy.SEQUENTIAL:
                await self._execute_sequential(query_plan, result)
            elif query_plan.strategy == QueryStrategy.PARALLEL:
                await self._execute_parallel(query_plan, result)
            elif query_plan.strategy == QueryStrategy.OPTIMIZED:
                await self._execute_optimized(query_plan, result)
            else:
                raise ValueError(f"Unknown strategy: {query_plan.strategy}")
            
            result.success = True
            result.total_execution_time = time.time() - start_time
            
            logger.info(
                f"Federated query {query_plan.query_id} completed",
                execution_time=result.total_execution_time,
                success=result.success
            )
            
        except Exception as e:
            result.error = str(e)
            result.total_execution_time = time.time() - start_time
            logger.error(f"Federated query {query_plan.query_id} failed: {e}")
        
        finally:
            del self.active_queries[query_plan.query_id]
        
        return result
    
    async def _execute_sequential(self, plan: QueryPlan, result: UnifiedQueryResult) -> None:
        """Execute operations sequentially."""
        
        # Execute PostgreSQL operations
        for op in plan.postgresql_ops:
            db_result = await self._execute_postgresql_operation(op)
            result.postgresql_result = db_result
            result.database_execution_times['postgresql'] = db_result.execution_time or 0.0
        
        # Execute Vector operations
        for op in plan.vector_ops:
            db_result = await self._execute_vector_operation(op)
            result.vector_result = db_result
            result.database_execution_times['vector'] = db_result.execution_time or 0.0
        
        # Execute Graph operations
        for op in plan.graph_ops:
            db_result = await self._execute_graph_operation(op)
            result.graph_result = db_result
            result.database_execution_times['graph'] = db_result.execution_time or 0.0
        
        # Combine results
        result.data = self._combine_results(result)
    
    async def _execute_parallel(self, plan: QueryPlan, result: UnifiedQueryResult) -> None:
        """Execute operations in parallel where possible."""
        
        tasks = []
        
        # Create parallel tasks
        if plan.postgresql_ops:
            tasks.append(('postgresql', self._execute_postgresql_operations(plan.postgresql_ops)))
        
        if plan.vector_ops:
            tasks.append(('vector', self._execute_vector_operations(plan.vector_ops)))
        
        if plan.graph_ops:
            tasks.append(('graph', self._execute_graph_operations(plan.graph_ops)))
        
        # Execute all tasks in parallel
        if tasks:
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            for i, (db_type, task_result) in enumerate(zip([task[0] for task in tasks], results)):
                if isinstance(task_result, Exception):
                    result.warnings.append(f"{db_type} operations failed: {task_result}")
                else:
                    setattr(result, f"{db_type}_result", task_result)
                    if hasattr(task_result, 'execution_time'):
                        result.database_execution_times[db_type] = task_result.execution_time or 0.0
        
        # Combine results
        result.data = self._combine_results(result)
    
    async def _execute_optimized(self, plan: QueryPlan, result: UnifiedQueryResult) -> None:
        """Execute operations with dependency-aware optimization."""
        
        executed_ops = set()
        operation_results = {}
        
        # Build operation dependency graph
        all_ops = {
            'postgresql': plan.postgresql_ops,
            'vector': plan.vector_ops,
            'graph': plan.graph_ops
        }
        
        # Execute operations respecting dependencies
        while len(executed_ops) < sum(len(ops) for ops in all_ops.values()):
            ready_ops = []
            
            # Find operations that can be executed (dependencies satisfied)
            for db_type, ops in all_ops.items():
                for i, op in enumerate(ops):
                    op_id = f"{db_type}_{i}"
                    if op_id not in executed_ops:
                        dependencies = op.get('depends_on', [])
                        if isinstance(dependencies, str):
                            dependencies = [dependencies]
                        
                        if all(dep in executed_ops for dep in dependencies):
                            ready_ops.append((db_type, op, op_id))
            
            if not ready_ops:
                # Deadlock or circular dependency
                remaining_ops = [
                    f"{db_type}_{i}" for db_type, ops in all_ops.items() 
                    for i in range(len(ops)) 
                    if f"{db_type}_{i}" not in executed_ops
                ]
                result.warnings.append(f"Potential deadlock, remaining ops: {remaining_ops}")
                break
            
            # Execute ready operations in parallel
            if len(ready_ops) == 1:
                db_type, op, op_id = ready_ops[0]
                op_result = await self._execute_single_operation(db_type, op, operation_results)
                operation_results[op_id] = op_result
                executed_ops.add(op_id)
            else:
                # Execute multiple ready operations in parallel
                tasks = []
                op_ids = []
                for db_type, op, op_id in ready_ops:
                    tasks.append(self._execute_single_operation(db_type, op, operation_results))
                    op_ids.append(op_id)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for op_id, op_result in zip(op_ids, results):
                    if not isinstance(op_result, Exception):
                        operation_results[op_id] = op_result
                    executed_ops.add(op_id)
        
        # Aggregate final results
        result.data = self._aggregate_optimized_results(operation_results)
    
    async def _execute_single_operation(self, 
                                       db_type: str, 
                                       operation: Dict[str, Any], 
                                       previous_results: Dict[str, Any]) -> QueryResult:
        """Execute a single database operation."""
        
        # Inject data from previous operations if needed
        if 'depends_on' in operation:
            operation = self._inject_dependencies(operation, previous_results)
        
        if db_type == 'postgresql':
            return await self._execute_postgresql_operation(operation)
        elif db_type == 'vector':
            return await self._execute_vector_operation(operation)
        elif db_type == 'graph':
            return await self._execute_graph_operation(operation)
        else:
            raise ValueError(f"Unknown database type: {db_type}")
    
    def _inject_dependencies(self, 
                           operation: Dict[str, Any], 
                           previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Inject data from previous operations into current operation."""
        
        # This is a simplified implementation
        # In a real system, this would be more sophisticated
        
        depends_on = operation.get('depends_on')
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        for dependency in depends_on:
            if dependency in previous_results:
                dep_result = previous_results[dependency]
                if hasattr(dep_result, 'data') and dep_result.data:
                    # Inject data based on operation type
                    if operation.get('operation') == 'enrich_metadata':
                        # Extract entity IDs for join
                        if isinstance(dep_result.data, list):
                            entity_ids = [item.get('entity_id') for item in dep_result.data]
                            operation['entity_ids'] = entity_ids
                    
                    elif operation.get('operation') == 'get_entity_details':
                        # Extract node IDs from graph traversal
                        if isinstance(dep_result.data, list):
                            entity_ids = []
                            for item in dep_result.data:
                                if isinstance(item, dict) and 'nodes' in item:
                                    for node in item['nodes']:
                                        if 'entity_id' in node:
                                            entity_ids.append(node['entity_id'])
                            operation['entity_ids'] = entity_ids
        
        return operation
    
    async def _execute_postgresql_operation(self, operation: Dict[str, Any]) -> QueryResult:
        """Execute PostgreSQL operation."""
        op_type = operation.get('operation')
        
        if op_type == 'execute_query':
            return await self.unified_db.postgresql.execute_query(
                operation.get('query', ''),
                operation.get('params')
            )
        elif op_type == 'get_user_profile':
            return await self.unified_db.postgresql.read(
                'users',
                {'entity_id': operation.get('user_id')}
            )
        elif op_type == 'enrich_metadata':
            entity_ids = operation.get('entity_ids', [])
            if entity_ids:
                # Create IN clause for multiple IDs
                placeholders = ','.join([f"${i+1}" for i in range(len(entity_ids))])
                query = f"SELECT * FROM entities WHERE entity_id IN ({placeholders})"
                return await self.unified_db.postgresql.execute_query(query, entity_ids)
        elif op_type == 'get_entity_details':
            entity_ids = operation.get('entity_ids', [])
            if entity_ids:
                placeholders = ','.join([f"${i+1}" for i in range(len(entity_ids))])
                query = f"SELECT * FROM entities WHERE entity_id IN ({placeholders})"
                return await self.unified_db.postgresql.execute_query(query, entity_ids)
        
        # Default empty result
        return QueryResult(success=False, data=None, error="Unknown PostgreSQL operation")
    
    async def _execute_postgresql_operations(self, operations: List[Dict[str, Any]]) -> QueryResult:
        """Execute multiple PostgreSQL operations."""
        results = []
        for op in operations:
            result = await self._execute_postgresql_operation(op)
            results.append(result)
        
        # Combine results
        combined_data = []
        for result in results:
            if result.success and result.data:
                if isinstance(result.data, list):
                    combined_data.extend(result.data)
                else:
                    combined_data.append(result.data)
        
        return QueryResult(
            success=True,
            data=combined_data,
            source_database=DatabaseType.POSTGRESQL,
            affected_rows=len(combined_data)
        )
    
    async def _execute_vector_operation(self, operation: Dict[str, Any]) -> QueryResult:
        """Execute vector database operation."""
        op_type = operation.get('operation')
        
        if op_type == 'execute_query':
            return await self.unified_db.vector.execute_query(
                operation.get('query', ''),
                operation.get('params')
            )
        elif op_type == 'similarity_search':
            if operation.get('query_text'):
                return await self.unified_db.vector.similarity_search_by_text(
                    operation['query_text'],
                    operation.get('limit', 10),
                    operation.get('threshold', 0.8),
                    operation.get('entity_type')
                )
            elif operation.get('query_embedding'):
                return await self.unified_db.vector.similarity_search(
                    operation['query_embedding'],
                    operation.get('limit', 10),
                    operation.get('threshold', 0.8),
                    operation.get('entity_type')
                )
        elif op_type == 'get_user_embedding':
            return await self.unified_db.vector.get_embedding(operation.get('entity_id'))
        
        return QueryResult(success=False, data=None, error="Unknown vector operation")
    
    async def _execute_vector_operations(self, operations: List[Dict[str, Any]]) -> QueryResult:
        """Execute multiple vector operations."""
        results = []
        for op in operations:
            result = await self._execute_vector_operation(op)
            results.append(result)
        
        # Combine results
        combined_data = []
        for result in results:
            if result.success and result.data:
                if isinstance(result.data, list):
                    combined_data.extend(result.data)
                else:
                    combined_data.append(result.data)
        
        return QueryResult(
            success=True,
            data=combined_data,
            source_database=DatabaseType.VECTOR,
            affected_rows=len(combined_data)
        )
    
    async def _execute_graph_operation(self, operation: Dict[str, Any]) -> QueryResult:
        """Execute graph database operation."""
        op_type = operation.get('operation')
        
        if op_type == 'execute_cypher':
            return await self.unified_db.graph.execute_cypher(
                operation.get('query', ''),
                operation.get('return_columns')
            )
        elif op_type == 'graph_traversal':
            return await self.unified_db.graph.graph_traversal(
                operation.get('start_node_id'),
                operation.get('max_depth', 3),
                operation.get('filters')
            )
        elif op_type == 'find_similar_users':
            return await self.unified_db.graph.find_relationships(
                operation.get('user_id'),
                operation.get('relationship_types'),
                "both"
            )
        
        return QueryResult(success=False, data=None, error="Unknown graph operation")
    
    async def _execute_graph_operations(self, operations: List[Dict[str, Any]]) -> QueryResult:
        """Execute multiple graph operations."""
        results = []
        for op in operations:
            result = await self._execute_graph_operation(op)
            results.append(result)
        
        # Combine results
        combined_data = []
        for result in results:
            if result.success and result.data:
                if isinstance(result.data, list):
                    combined_data.extend(result.data)
                else:
                    combined_data.append(result.data)
        
        return QueryResult(
            success=True,
            data=combined_data,
            source_database=DatabaseType.GRAPH,
            affected_rows=len(combined_data)
        )
    
    def _combine_results(self, result: UnifiedQueryResult) -> Dict[str, Any]:
        """Combine results from multiple databases."""
        combined = {
            'query_id': result.query_id,
            'databases_queried': [],
            'total_rows': 0,
            'results': {}
        }
        
        if result.postgresql_result and result.postgresql_result.success:
            combined['databases_queried'].append('postgresql')
            combined['results']['postgresql'] = result.postgresql_result.data
            combined['total_rows'] += result.postgresql_result.affected_rows
        
        if result.vector_result and result.vector_result.success:
            combined['databases_queried'].append('vector')
            combined['results']['vector'] = result.vector_result.data
            combined['total_rows'] += result.vector_result.affected_rows
        
        if result.graph_result and result.graph_result.success:
            combined['databases_queried'].append('graph')
            combined['results']['graph'] = result.graph_result.data
            combined['total_rows'] += result.graph_result.affected_rows
        
        return combined
    
    def _aggregate_optimized_results(self, operation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from optimized execution."""
        aggregated = {
            'operations_executed': len(operation_results),
            'results': operation_results,
            'final_data': None
        }
        
        # Extract final result (usually the last operation)
        if operation_results:
            last_result = list(operation_results.values())[-1]
            if hasattr(last_result, 'data'):
                aggregated['final_data'] = last_result.data
        
        return aggregated


class UnifiedQueryInterface:
    """High-level interface for unified database queries."""
    
    def __init__(self, unified_db: UnifiedDatabase):
        self.unified_db = unified_db
        self.query_planner = QueryPlanner()
        self.query_executor = QueryExecutor(unified_db)
    
    async def search_similar_content(self, 
                                   query_text: str, 
                                   content_type: Optional[str] = None,
                                   limit: int = 10,
                                   include_metadata: bool = True) -> UnifiedQueryResult:
        """Search for similar content across the ecosystem."""
        
        query_spec = {
            'query_text': query_text,
            'entity_type': content_type,
            'limit': limit,
            'threshold': 0.7,
            'include_metadata': include_metadata
        }
        
        plan = self.query_planner.create_plan(query_spec, QueryType.SIMILARITY_SEARCH)
        return await self.query_executor.execute_query(plan)
    
    async def get_recommendations(self, 
                                user_id: str, 
                                recommendation_type: str = 'content',
                                limit: int = 20) -> UnifiedQueryResult:
        """Get AI-powered recommendations for a user."""
        
        query_spec = {
            'user_id': user_id,
            'type': recommendation_type,
            'limit': limit
        }
        
        plan = self.query_planner.create_plan(query_spec, QueryType.RECOMMENDATION)
        return await self.query_executor.execute_query(plan)
    
    async def analyze_relationships(self, 
                                  entity_id: str, 
                                  max_depth: int = 3,
                                  include_similarity: bool = False) -> UnifiedQueryResult:
        """Analyze entity relationships with data enrichment."""
        
        query_spec = {
            'start_node_id': entity_id,
            'max_depth': max_depth,
            'include_similarity': include_similarity,
            'similarity_limit': 5
        }
        
        plan = self.query_planner.create_plan(query_spec, QueryType.GRAPH_TRAVERSAL)
        return await self.query_executor.execute_query(plan)
    
    async def federated_join(self, 
                           postgresql_query: Optional[str] = None,
                           vector_query: Optional[str] = None,
                           graph_query: Optional[str] = None,
                           **params) -> UnifiedQueryResult:
        """Execute federated join across multiple databases."""
        
        query_spec = {
            'postgresql_query': postgresql_query,
            'vector_query': vector_query,
            'graph_query': graph_query,
            **params
        }
        
        plan = self.query_planner.create_plan(query_spec, QueryType.FEDERATED_JOIN)
        return await self.query_executor.execute_query(plan)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_query_system():
        print("🔍 Testing 3db Query Coordination System")
        
        # Create a query planner
        planner = QueryPlanner()
        
        # Test similarity search plan
        similarity_spec = {
            'query_text': 'machine learning algorithms',
            'entity_type': 'document',
            'limit': 10,
            'threshold': 0.8
        }
        
        plan = planner.create_plan(similarity_spec, QueryType.SIMILARITY_SEARCH)
        
        print(f"✅ Created similarity search plan: {plan.query_id}")
        print(f"Strategy: {plan.strategy.value}")
        print(f"Estimated cost: {plan.estimated_cost}")
        print(f"Vector operations: {len(plan.vector_ops)}")
        print(f"PostgreSQL operations: {len(plan.postgresql_ops)}")
        print(f"Dependencies: {plan.dependencies}")
        
        # Test recommendation plan
        rec_spec = {
            'user_id': 'user_123',
            'type': 'content',
            'limit': 20
        }
        
        rec_plan = planner.create_plan(rec_spec, QueryType.RECOMMENDATION)
        
        print(f"\n✅ Created recommendation plan: {rec_plan.query_id}")
        print(f"Strategy: {rec_plan.strategy.value}")
        print(f"Estimated cost: {rec_plan.estimated_cost}")
        print(f"Complex dependencies: {len(rec_plan.dependencies)}")
    
    # Run test
    asyncio.run(test_query_system())
