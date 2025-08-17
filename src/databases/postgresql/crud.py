"""
3db Unified Database Ecosystem - PostgreSQL CRUD Implementation

This module provides the PostgreSQL implementation for CRUD operations,
serving as the "memory storage" component of the unified database brain.
"""

import asyncio
import asyncpg
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
from contextlib import asynccontextmanager

from ...core.base import (
    BaseDatabase, CRUDOperations, DatabaseType, QueryResult, EntityMetadata
)
from ...core.config import DatabaseConfig
from ..utils.logging import get_logger, monitor_performance, CircuitBreaker

logger = get_logger("postgresql")


class PostgreSQLDatabase(BaseDatabase, CRUDOperations):
    """
    PostgreSQL implementation for CRUD operations in the unified ecosystem.
    Handles traditional relational data storage with full ACID compliance.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config.dict())
        self.config = config
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)
        
        # PostgreSQL-specific settings
        self.statement_timeout = config.statement_timeout if hasattr(config, 'statement_timeout') else 30000
        self.idle_timeout = config.idle_in_transaction_timeout if hasattr(config, 'idle_in_transaction_timeout') else 60000
    
    def _get_database_type(self) -> DatabaseType:
        """Return PostgreSQL database type."""
        return DatabaseType.POSTGRESQL
    
    @monitor_performance("postgresql_connect", "postgresql")
    async def connect(self) -> bool:
        """Establish PostgreSQL connection pool."""
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
                'server_settings': {
                    'statement_timeout': str(self.statement_timeout),
                    'idle_in_transaction_session_timeout': str(self.idle_timeout)
                }
            }
            
            self.connection_pool = await asyncpg.create_pool(**pool_config)
            
            # Test connection
            async with self.connection_pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            self.is_connected = True
            self._metrics['connections_opened'] += 1
            
            logger.info(
                "PostgreSQL connection pool established",
                host=self.config.host,
                database=self.config.database,
                pool_size=f"{self.config.pool_min_size}-{self.config.pool_max_size}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            self.connection_pool = None
            self.is_connected = False
            self._metrics['connections_closed'] += 1
            logger.info("PostgreSQL connection pool closed")
    
    @monitor_performance("postgresql_health_check", "postgresql")
    async def health_check(self) -> bool:
        """Perform PostgreSQL health check."""
        if not self.connection_pool:
            return False
        
        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchval('SELECT 1')
                return result == 1
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with proper error handling."""
        if not self.connection_pool:
            raise Exception("Database not connected")
        
        conn = None
        try:
            conn = await self.connection_pool.acquire()
            yield conn
        finally:
            if conn:
                await self.connection_pool.release(conn)
    
    @monitor_performance("postgresql_execute", "postgresql")
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a PostgreSQL query and return standardized result."""
        start_time = datetime.utcnow()
        
        try:
            async def _execute():
                async with self.get_connection() as conn:
                    if params:
                        # Convert named parameters to positional for asyncpg
                        param_values = list(params.values())
                        # Replace named placeholders with $1, $2, etc.
                        formatted_query = query
                        for i, key in enumerate(params.keys(), 1):
                            formatted_query = formatted_query.replace(f":{key}", f"${i}")
                        
                        result = await conn.fetch(formatted_query, *param_values)
                    else:
                        result = await conn.fetch(query)
                    
                    return result
            
            # Execute with circuit breaker protection
            raw_result = await self.circuit_breaker.call(_execute)
            
            # Convert asyncpg records to dictionaries
            data = [dict(record) for record in raw_result] if raw_result else []
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(execution_time, success=True)
            
            logger.debug(
                "PostgreSQL query executed",
                query=query[:100] + "..." if len(query) > 100 else query,
                params=params,
                rows_returned=len(data),
                execution_time_ms=execution_time * 1000
            )
            
            return QueryResult(
                success=True,
                data=data,
                metadata={
                    'rows_returned': len(data),
                    'execution_time_ms': execution_time * 1000
                },
                execution_time=execution_time,
                source_database=DatabaseType.POSTGRESQL,
                affected_rows=len(data)
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(execution_time, success=False)
            
            error_msg = f"PostgreSQL query failed: {e}"
            logger.error(error_msg, query=query, params=params)
            
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=execution_time,
                source_database=DatabaseType.POSTGRESQL
            )
    
    @asynccontextmanager
    async def transaction(self):
        """PostgreSQL transaction context manager."""
        async with self.get_connection() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                yield conn
                await tr.commit()
                logger.debug("PostgreSQL transaction committed")
            except Exception as e:
                await tr.rollback()
                logger.error(f"PostgreSQL transaction rolled back: {e}")
                raise
    
    # CRUD Operations Implementation
    
    @monitor_performance("postgresql_create", "postgresql")
    async def create(self, table: str, data: Dict[str, Any]) -> QueryResult:
        """Create a new record in PostgreSQL."""
        try:
            # Prepare insert statement
            columns = list(data.keys())
            placeholders = [f"${i+1}" for i in range(len(columns))]
            values = list(data.values())
            
            query = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                RETURNING *
            """
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(query, *values)
                
            return QueryResult(
                success=True,
                data=dict(result) if result else None,
                metadata={'operation': 'create', 'table': table},
                source_database=DatabaseType.POSTGRESQL,
                affected_rows=1 if result else 0
            )
            
        except Exception as e:
            error_msg = f"Failed to create record in {table}: {e}"
            logger.error(error_msg, table=table, data=data)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.POSTGRESQL
            )
    
    @monitor_performance("postgresql_read", "postgresql")
    async def read(self, table: str, filters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Read records from PostgreSQL with optional filters."""
        try:
            query = f"SELECT * FROM {table}"
            params = []
            
            if filters:
                conditions = []
                for i, (key, value) in enumerate(filters.items(), 1):
                    conditions.append(f"{key} = ${i}")
                    params.append(value)
                
                query += f" WHERE {' AND '.join(conditions)}"
            
            async with self.get_connection() as conn:
                results = await conn.fetch(query, *params)
            
            data = [dict(record) for record in results]
            
            return QueryResult(
                success=True,
                data=data,
                metadata={'operation': 'read', 'table': table, 'filters': filters},
                source_database=DatabaseType.POSTGRESQL,
                affected_rows=len(data)
            )
            
        except Exception as e:
            error_msg = f"Failed to read from {table}: {e}"
            logger.error(error_msg, table=table, filters=filters)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.POSTGRESQL
            )
    
    @monitor_performance("postgresql_update", "postgresql")
    async def update(self, table: str, data: Dict[str, Any], filters: Dict[str, Any]) -> QueryResult:
        """Update records in PostgreSQL matching filters."""
        try:
            # Prepare SET clause
            set_clauses = []
            params = []
            param_counter = 1
            
            for key, value in data.items():
                set_clauses.append(f"{key} = ${param_counter}")
                params.append(value)
                param_counter += 1
            
            # Prepare WHERE clause
            where_clauses = []
            for key, value in filters.items():
                where_clauses.append(f"{key} = ${param_counter}")
                params.append(value)
                param_counter += 1
            
            query = f"""
                UPDATE {table}
                SET {', '.join(set_clauses)}
                WHERE {' AND '.join(where_clauses)}
                RETURNING *
            """
            
            async with self.get_connection() as conn:
                results = await conn.fetch(query, *params)
            
            data_result = [dict(record) for record in results]
            
            return QueryResult(
                success=True,
                data=data_result,
                metadata={'operation': 'update', 'table': table, 'filters': filters},
                source_database=DatabaseType.POSTGRESQL,
                affected_rows=len(data_result)
            )
            
        except Exception as e:
            error_msg = f"Failed to update {table}: {e}"
            logger.error(error_msg, table=table, data=data, filters=filters)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.POSTGRESQL
            )
    
    @monitor_performance("postgresql_delete", "postgresql")
    async def delete(self, table: str, filters: Dict[str, Any]) -> QueryResult:
        """Delete records from PostgreSQL matching filters."""
        try:
            # Prepare WHERE clause
            where_clauses = []
            params = []
            
            for i, (key, value) in enumerate(filters.items(), 1):
                where_clauses.append(f"{key} = ${i}")
                params.append(value)
            
            query = f"""
                DELETE FROM {table}
                WHERE {' AND '.join(where_clauses)}
                RETURNING *
            """
            
            async with self.get_connection() as conn:
                results = await conn.fetch(query, *params)
            
            data = [dict(record) for record in results]
            
            return QueryResult(
                success=True,
                data=data,
                metadata={'operation': 'delete', 'table': table, 'filters': filters},
                source_database=DatabaseType.POSTGRESQL,
                affected_rows=len(data)
            )
            
        except Exception as e:
            error_msg = f"Failed to delete from {table}: {e}"
            logger.error(error_msg, table=table, filters=filters)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.POSTGRESQL
            )
    
    @monitor_performance("postgresql_upsert", "postgresql")
    async def upsert(self, table: str, data: Dict[str, Any], conflict_columns: List[str]) -> QueryResult:
        """Insert or update record in PostgreSQL based on conflict columns."""
        try:
            # Prepare insert statement
            columns = list(data.keys())
            placeholders = [f"${i+1}" for i in range(len(columns))]
            values = list(data.values())
            
            # Prepare ON CONFLICT clause
            conflict_cols = ', '.join(conflict_columns)
            update_clauses = []
            
            for col in columns:
                if col not in conflict_columns:
                    update_clauses.append(f"{col} = EXCLUDED.{col}")
            
            query = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT ({conflict_cols})
                DO UPDATE SET {', '.join(update_clauses)}
                RETURNING *
            """
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(query, *values)
            
            return QueryResult(
                success=True,
                data=dict(result) if result else None,
                metadata={'operation': 'upsert', 'table': table, 'conflict_columns': conflict_columns},
                source_database=DatabaseType.POSTGRESQL,
                affected_rows=1 if result else 0
            )
            
        except Exception as e:
            error_msg = f"Failed to upsert in {table}: {e}"
            logger.error(error_msg, table=table, data=data, conflict_columns=conflict_columns)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.POSTGRESQL
            )
    
    # Advanced PostgreSQL operations
    
    async def execute_raw_sql(self, sql: str, params: Optional[List[Any]] = None) -> QueryResult:
        """Execute raw SQL with parameters."""
        try:
            async with self.get_connection() as conn:
                if params:
                    results = await conn.fetch(sql, *params)
                else:
                    results = await conn.fetch(sql)
            
            data = [dict(record) for record in results]
            
            return QueryResult(
                success=True,
                data=data,
                metadata={'operation': 'raw_sql'},
                source_database=DatabaseType.POSTGRESQL,
                affected_rows=len(data)
            )
            
        except Exception as e:
            error_msg = f"Failed to execute raw SQL: {e}"
            logger.error(error_msg, sql=sql[:100])
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.POSTGRESQL
            )
    
    async def bulk_insert(self, table: str, records: List[Dict[str, Any]]) -> QueryResult:
        """Perform bulk insert for multiple records."""
        if not records:
            return QueryResult(
                success=True,
                data=[],
                metadata={'operation': 'bulk_insert', 'table': table},
                source_database=DatabaseType.POSTGRESQL,
                affected_rows=0
            )
        
        try:
            # Get columns from first record
            columns = list(records[0].keys())
            
            # Prepare values for bulk insert
            values_list = []
            for record in records:
                values_list.extend([record.get(col) for col in columns])
            
            # Create placeholders for all values
            rows_count = len(records)
            cols_count = len(columns)
            
            placeholders = []
            for i in range(rows_count):
                row_placeholders = []
                for j in range(cols_count):
                    row_placeholders.append(f"${i * cols_count + j + 1}")
                placeholders.append(f"({', '.join(row_placeholders)})")
            
            query = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES {', '.join(placeholders)}
                RETURNING *
            """
            
            async with self.get_connection() as conn:
                results = await conn.fetch(query, *values_list)
            
            data = [dict(record) for record in results]
            
            logger.info(f"Bulk inserted {len(data)} records into {table}")
            
            return QueryResult(
                success=True,
                data=data,
                metadata={'operation': 'bulk_insert', 'table': table},
                source_database=DatabaseType.POSTGRESQL,
                affected_rows=len(data)
            )
            
        except Exception as e:
            error_msg = f"Failed to bulk insert into {table}: {e}"
            logger.error(error_msg, table=table, record_count=len(records))
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.POSTGRESQL
            )
    
    async def get_table_schema(self, table: str) -> QueryResult:
        """Get schema information for a table."""
        query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """
        
        return await self.execute_query(query, [table])
    
    async def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = $1
            )
        """
        
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval(query, table)
                return bool(result)
        except Exception:
            return False


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ..core.config import DatabaseConfig
    
    async def test_postgresql():
        # Example configuration
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_3db",
            username="postgres",
            password="password"
        )
        
        db = PostgreSQLDatabase(config)
        
        # Test connection
        if await db.connect():
            print("✅ PostgreSQL connection successful")
            
            # Test health check
            if await db.health_check():
                print("✅ PostgreSQL health check passed")
            
            await db.disconnect()
        else:
            print("❌ PostgreSQL connection failed")
    
    # Run test
    asyncio.run(test_postgresql())
