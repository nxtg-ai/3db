"""
3db Unified Database Ecosystem - Vector Database Implementation (pgvector)

This module provides the pgvector implementation for vector embeddings and similarity search,
serving as the "intuitive recall" component of the unified database brain.
"""

import asyncio
import asyncpg
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

from ...core.base import (
    BaseDatabase, VectorOperations, DatabaseType, QueryResult, EntityMetadata
)
from ...core.config import DatabaseConfig
from ..utils.logging import get_logger, monitor_performance, CircuitBreaker

logger = get_logger("vector_db")


class VectorDatabase(BaseDatabase, VectorOperations):
    """
    pgvector implementation for vector embeddings and similarity search.
    Handles semantic search, recommendations, and intelligent content retrieval.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config.dict())
        self.config = config
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)
        
        # Vector-specific settings
        self.embedding_dimension = getattr(config, 'embedding_dimension', 384)
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.8)
        self.embedding_model_name = getattr(config, 'embedding_model', 'all-MiniLM-L6-v2')
        
        # Initialize embedding model
        self.embedding_model: Optional[SentenceTransformer] = None
        self._model_loaded = False
    
    def _get_database_type(self) -> DatabaseType:
        """Return Vector database type."""
        return DatabaseType.VECTOR
    
    async def _load_embedding_model(self) -> bool:
        """Load the sentence transformer model for embeddings."""
        if self._model_loaded:
            return True
        
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self._model_loaded = True
            logger.info("Embedding model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    @monitor_performance("vector_connect", "vector")
    async def connect(self) -> bool:
        """Establish pgvector connection pool and setup extensions."""
        try:
            # Load embedding model first
            if not await self._load_embedding_model():
                return False
            
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
            
            # Setup pgvector extension
            async with self.connection_pool.acquire() as conn:
                # Enable vector extension
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
                
                # Test vector functionality
                await conn.execute('SELECT 1')
            
            # Initialize vector tables if they don't exist
            await self._initialize_vector_tables()
            
            self.is_connected = True
            self._metrics['connections_opened'] += 1
            
            logger.info(
                "pgvector connection pool established",
                host=self.config.host,
                database=self.config.database,
                embedding_dimension=self.embedding_dimension,
                model=self.embedding_model_name
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to pgvector: {e}")
            self.is_connected = False
            return False
    
    async def _initialize_vector_tables(self) -> None:
        """Initialize required vector tables and indexes."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Create embeddings table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        entity_id VARCHAR(255) UNIQUE NOT NULL,
                        entity_type VARCHAR(100) NOT NULL,
                        embedding vector({self.embedding_dimension}) NOT NULL,
                        metadata JSONB DEFAULT '{{}}',
                        text_content TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for similarity search (using cosine distance)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS embeddings_cosine_idx 
                    ON embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                
                # Create index for L2 distance
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS embeddings_l2_idx 
                    ON embeddings USING ivfflat (embedding vector_l2_ops)
                    WITH (lists = 100)
                """)
                
                # Create indexes for efficient filtering
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS embeddings_entity_id_idx 
                    ON embeddings (entity_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS embeddings_entity_type_idx 
                    ON embeddings (entity_type)
                """)
                
                # Create trigger for updating updated_at
                await conn.execute("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql'
                """)
                
                await conn.execute("""
                    DROP TRIGGER IF EXISTS update_embeddings_updated_at ON embeddings
                """)
                
                await conn.execute("""
                    CREATE TRIGGER update_embeddings_updated_at 
                    BEFORE UPDATE ON embeddings 
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
                """)
                
                logger.info("Vector tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector tables: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close pgvector connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            self.connection_pool = None
            self.is_connected = False
            self._metrics['connections_closed'] += 1
            logger.info("pgvector connection pool closed")
    
    @monitor_performance("vector_health_check", "vector")
    async def health_check(self) -> bool:
        """Perform pgvector health check."""
        if not self.connection_pool or not self._model_loaded:
            return False
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Test basic vector operation
                test_vector = [0.1] * self.embedding_dimension
                result = await conn.fetchval(
                    'SELECT $1::vector <-> $1::vector',
                    test_vector
                )
                return result == 0.0
        except Exception as e:
            logger.error(f"pgvector health check failed: {e}")
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
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a pgvector query and return standardized result."""
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
                source_database=DatabaseType.VECTOR,
                affected_rows=len(data)
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(execution_time, success=False)
            
            error_msg = f"pgvector query failed: {e}"
            logger.error(error_msg, query=query[:100])
            
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=execution_time,
                source_database=DatabaseType.VECTOR
            )
    
    # Vector Operations Implementation
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector from text using the loaded model."""
        if not self._model_loaded or not self.embedding_model:
            raise ValueError("Embedding model not loaded")
        
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    @monitor_performance("vector_insert", "vector")
    async def insert_embedding(self, 
                              entity_id: str, 
                              embedding: List[float], 
                              metadata: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Insert a vector embedding."""
        try:
            if len(embedding) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
            
            metadata = metadata or {}
            
            query = """
                INSERT INTO embeddings (entity_id, entity_type, embedding, metadata, text_content)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (entity_id) 
                DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    text_content = EXCLUDED.text_content,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
            """
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(
                    query,
                    entity_id,
                    metadata.get('entity_type', 'unknown'),
                    embedding,
                    json.dumps(metadata),
                    metadata.get('text_content', '')
                )
            
            return QueryResult(
                success=True,
                data=dict(result) if result else None,
                metadata={'operation': 'insert_embedding', 'entity_id': entity_id},
                source_database=DatabaseType.VECTOR,
                affected_rows=1 if result else 0
            )
            
        except Exception as e:
            error_msg = f"Failed to insert embedding for {entity_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.VECTOR
            )
    
    async def insert_text_embedding(self, 
                                   entity_id: str, 
                                   text: str, 
                                   metadata: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Generate and insert embedding from text."""
        try:
            embedding = self.generate_embedding(text)
            
            if metadata is None:
                metadata = {}
            metadata['text_content'] = text
            
            return await self.insert_embedding(entity_id, embedding, metadata)
            
        except Exception as e:
            error_msg = f"Failed to insert text embedding for {entity_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.VECTOR
            )
    
    @monitor_performance("vector_similarity_search", "vector")
    async def similarity_search(self, 
                               query_embedding: List[float], 
                               limit: int = 10, 
                               threshold: float = 0.8,
                               entity_type: Optional[str] = None) -> QueryResult:
        """Perform similarity search on embeddings."""
        try:
            if len(query_embedding) != self.embedding_dimension:
                raise ValueError(f"Query embedding dimension mismatch: expected {self.embedding_dimension}, got {len(query_embedding)}")
            
            # Build query with optional entity type filter
            base_query = """
                SELECT entity_id, entity_type, metadata, text_content,
                       1 - (embedding <=> $1) as similarity,
                       embedding <-> $1 as distance
                FROM embeddings
            """
            
            conditions = ["1 - (embedding <=> $1) >= $2"]
            params = [query_embedding, threshold]
            
            if entity_type:
                conditions.append("entity_type = $3")
                params.append(entity_type)
            
            query = f"""
                {base_query}
                WHERE {' AND '.join(conditions)}
                ORDER BY embedding <=> $1
                LIMIT ${len(params) + 1}
            """
            params.append(limit)
            
            async with self.get_connection() as conn:
                results = await conn.fetch(query, *params)
            
            data = []
            for result in results:
                record = dict(result)
                # Parse metadata JSON
                if record['metadata']:
                    record['metadata'] = json.loads(record['metadata']) if isinstance(record['metadata'], str) else record['metadata']
                data.append(record)
            
            logger.debug(
                f"Similarity search completed",
                results_count=len(data),
                threshold=threshold,
                entity_type=entity_type
            )
            
            return QueryResult(
                success=True,
                data=data,
                metadata={
                    'operation': 'similarity_search',
                    'threshold': threshold,
                    'entity_type': entity_type,
                    'limit': limit
                },
                source_database=DatabaseType.VECTOR,
                affected_rows=len(data)
            )
            
        except Exception as e:
            error_msg = f"Similarity search failed: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.VECTOR
            )
    
    async def similarity_search_by_text(self, 
                                       query_text: str, 
                                       limit: int = 10, 
                                       threshold: float = 0.8,
                                       entity_type: Optional[str] = None) -> QueryResult:
        """Perform similarity search using text query."""
        try:
            query_embedding = self.generate_embedding(query_text)
            return await self.similarity_search(query_embedding, limit, threshold, entity_type)
        except Exception as e:
            error_msg = f"Text similarity search failed: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.VECTOR
            )
    
    @monitor_performance("vector_update", "vector")
    async def update_embedding(self, 
                              entity_id: str, 
                              embedding: List[float], 
                              metadata: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Update an existing embedding."""
        try:
            if len(embedding) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
            
            metadata = metadata or {}
            
            query = """
                UPDATE embeddings 
                SET embedding = $2, 
                    metadata = $3,
                    text_content = $4,
                    updated_at = CURRENT_TIMESTAMP
                WHERE entity_id = $1
                RETURNING *
            """
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(
                    query,
                    entity_id,
                    embedding,
                    json.dumps(metadata),
                    metadata.get('text_content', '')
                )
            
            if not result:
                return QueryResult(
                    success=False,
                    data=None,
                    error=f"No embedding found for entity_id: {entity_id}",
                    source_database=DatabaseType.VECTOR
                )
            
            return QueryResult(
                success=True,
                data=dict(result),
                metadata={'operation': 'update_embedding', 'entity_id': entity_id},
                source_database=DatabaseType.VECTOR,
                affected_rows=1
            )
            
        except Exception as e:
            error_msg = f"Failed to update embedding for {entity_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.VECTOR
            )
    
    @monitor_performance("vector_delete", "vector")
    async def delete_embedding(self, entity_id: str) -> QueryResult:
        """Delete an embedding."""
        try:
            query = "DELETE FROM embeddings WHERE entity_id = $1 RETURNING *"
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(query, entity_id)
            
            if not result:
                return QueryResult(
                    success=False,
                    data=None,
                    error=f"No embedding found for entity_id: {entity_id}",
                    source_database=DatabaseType.VECTOR
                )
            
            return QueryResult(
                success=True,
                data=dict(result),
                metadata={'operation': 'delete_embedding', 'entity_id': entity_id},
                source_database=DatabaseType.VECTOR,
                affected_rows=1
            )
            
        except Exception as e:
            error_msg = f"Failed to delete embedding for {entity_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.VECTOR
            )
    
    @monitor_performance("vector_get", "vector")
    async def get_embedding(self, entity_id: str) -> QueryResult:
        """Retrieve a specific embedding."""
        try:
            query = "SELECT * FROM embeddings WHERE entity_id = $1"
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(query, entity_id)
            
            if not result:
                return QueryResult(
                    success=False,
                    data=None,
                    error=f"No embedding found for entity_id: {entity_id}",
                    source_database=DatabaseType.VECTOR
                )
            
            data = dict(result)
            # Parse metadata JSON
            if data['metadata']:
                data['metadata'] = json.loads(data['metadata']) if isinstance(data['metadata'], str) else data['metadata']
            
            return QueryResult(
                success=True,
                data=data,
                metadata={'operation': 'get_embedding', 'entity_id': entity_id},
                source_database=DatabaseType.VECTOR,
                affected_rows=1
            )
            
        except Exception as e:
            error_msg = f"Failed to get embedding for {entity_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.VECTOR
            )
    
    # Advanced vector operations
    
    async def batch_similarity_search(self, 
                                     query_embeddings: List[List[float]], 
                                     limit: int = 10, 
                                     threshold: float = 0.8) -> List[QueryResult]:
        """Perform batch similarity searches."""
        results = []
        for i, embedding in enumerate(query_embeddings):
            result = await self.similarity_search(embedding, limit, threshold)
            results.append(result)
        return results
    
    async def get_embedding_stats(self) -> QueryResult:
        """Get statistics about the embeddings table."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_embeddings,
                    COUNT(DISTINCT entity_type) as unique_entity_types,
                    MIN(created_at) as oldest_embedding,
                    MAX(created_at) as newest_embedding,
                    AVG(ARRAY_LENGTH(embedding, 1)) as avg_dimension
                FROM embeddings
            """
            
            async with self.get_connection() as conn:
                result = await conn.fetchrow(query)
            
            return QueryResult(
                success=True,
                data=dict(result) if result else {},
                metadata={'operation': 'get_stats'},
                source_database=DatabaseType.VECTOR,
                affected_rows=1
            )
            
        except Exception as e:
            error_msg = f"Failed to get embedding stats: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.VECTOR
            )
    
    async def find_similar_entities(self, 
                                   entity_id: str, 
                                   limit: int = 10,
                                   threshold: float = 0.8) -> QueryResult:
        """Find entities similar to a given entity."""
        try:
            # First get the embedding for the target entity
            target_result = await self.get_embedding(entity_id)
            if not target_result.success:
                return target_result
            
            target_embedding = target_result.data['embedding']
            
            # Perform similarity search excluding the target entity
            query = """
                SELECT entity_id, entity_type, metadata, text_content,
                       1 - (embedding <=> $1) as similarity
                FROM embeddings
                WHERE entity_id != $2 
                AND 1 - (embedding <=> $1) >= $3
                ORDER BY embedding <=> $1
                LIMIT $4
            """
            
            async with self.get_connection() as conn:
                results = await conn.fetch(query, target_embedding, entity_id, threshold, limit)
            
            data = []
            for result in results:
                record = dict(result)
                if record['metadata']:
                    record['metadata'] = json.loads(record['metadata']) if isinstance(record['metadata'], str) else record['metadata']
                data.append(record)
            
            return QueryResult(
                success=True,
                data=data,
                metadata={
                    'operation': 'find_similar_entities',
                    'target_entity_id': entity_id,
                    'threshold': threshold
                },
                source_database=DatabaseType.VECTOR,
                affected_rows=len(data)
            )
            
        except Exception as e:
            error_msg = f"Failed to find similar entities for {entity_id}: {e}"
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=None,
                error=error_msg,
                source_database=DatabaseType.VECTOR
            )


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ..core.config import DatabaseConfig
    
    async def test_vector_db():
        # Example configuration
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_3db_vector",
            username="postgres",
            password="password",
            embedding_dimension=384,
            similarity_threshold=0.8,
            embedding_model="all-MiniLM-L6-v2"
        )
        
        db = VectorDatabase(config)
        
        # Test connection
        if await db.connect():
            print("✅ pgvector connection successful")
            
            # Test health check
            if await db.health_check():
                print("✅ pgvector health check passed")
            
            # Test embedding generation
            try:
                embedding = db.generate_embedding("This is a test sentence.")
                print(f"✅ Embedding generated: dimension={len(embedding)}")
            except Exception as e:
                print(f"❌ Embedding generation failed: {e}")
            
            await db.disconnect()
        else:
            print("❌ pgvector connection failed")
    
    # Run test
    asyncio.run(test_vector_db())
