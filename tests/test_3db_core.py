"""
3db Unified Database Ecosystem - Core System Tests

Comprehensive tests for the main 3db system functionality including
entity creation, similarity search, relationship analysis, and recommendations.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from conftest import PerformanceTest


class TestDatabase3D:
    """Test the main Database3D class functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_3db_system):
        """Test 3db system initialization."""
        db3d = mock_3db_system
        
        # Test that system is initialized
        assert db3d.is_initialized
        assert db3d.is_running
        
        # Test database components are present
        assert db3d.postgresql_db is not None
        assert db3d.vector_db is not None
        assert db3d.graph_db is not None
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_3db_system):
        """Test system health check functionality."""
        db3d = mock_3db_system
        
        # Mock health check responses
        db3d.postgresql_db.health_check.return_value = True
        db3d.vector_db.health_check.return_value = True
        db3d.graph_db.health_check.return_value = True
        
        health_status = await db3d.health_check()
        
        assert health_status is not None
        assert "overall_health" in health_status
        assert "components" in health_status
        assert "timestamp" in health_status
        
        # Verify all components are checked
        components = health_status["components"]
        assert "postgresql" in components
        assert "vector" in components
        assert "graph" in components
    
    @pytest.mark.asyncio
    async def test_system_metrics(self, mock_3db_system):
        """Test system metrics collection."""
        db3d = mock_3db_system
        
        # Mock metrics from individual databases
        db3d.postgresql_db.get_metrics.return_value = {
            "queries_executed": 100,
            "queries_failed": 2,
            "avg_execution_time": 0.05
        }
        
        db3d.vector_db.get_metrics.return_value = {
            "queries_executed": 50,
            "queries_failed": 0,
            "avg_execution_time": 0.12
        }
        
        db3d.graph_db.get_metrics.return_value = {
            "queries_executed": 25,
            "queries_failed": 1,
            "avg_execution_time": 0.08
        }
        
        metrics = await db3d.get_system_metrics()
        
        assert metrics is not None
        assert "timestamp" in metrics
        assert "databases" in metrics
        assert "synchronization" in metrics
        assert "system" in metrics
        
        # Verify database metrics are included
        db_metrics = metrics["databases"]
        assert "postgresql" in db_metrics
        assert "vector" in db_metrics
        assert "graph" in db_metrics


class TestEntityManagement:
    """Test entity creation and management functionality."""
    
    @pytest.mark.asyncio
    async def test_create_user_entity(self, mock_3db_system, sample_entity_data):
        """Test creating a user entity."""
        db3d = mock_3db_system
        user_data = sample_entity_data["user"]
        
        result = await db3d.create_entity("user", user_data, sync_immediately=True)
        
        assert result["success"] is True
        assert "entity_id" in result
        assert result["entity_id"].startswith("user_")
        
        # Verify PostgreSQL create was called
        db3d.postgresql_db.create.assert_called_once()
        
        # Verify vector embedding creation for user with bio/interests
        db3d.vector_db.insert_text_embedding.assert_called_once()
        
        # Verify graph node creation
        db3d.graph_db.create_node.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_document_entity(self, mock_3db_system, sample_entity_data):
        """Test creating a document entity."""
        db3d = mock_3db_system
        document_data = sample_entity_data["document"]
        
        result = await db3d.create_entity("document", document_data, sync_immediately=True)
        
        assert result["success"] is True
        assert "entity_id" in result
        assert result["entity_id"].startswith("document_")
        
        # Verify all database operations were triggered
        db3d.postgresql_db.create.assert_called_once()
        db3d.vector_db.insert_text_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_product_entity(self, mock_3db_system, sample_entity_data):
        """Test creating a product entity."""
        db3d = mock_3db_system
        product_data = sample_entity_data["product"]
        
        result = await db3d.create_entity("product", product_data, sync_immediately=True)
        
        assert result["success"] is True
        assert "entity_id" in result
        assert result["entity_id"].startswith("product_")
        
        # Verify operations based on sync rules
        db3d.postgresql_db.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_entity_with_custom_id(self, mock_3db_system):
        """Test creating entity with custom entity_id."""
        db3d = mock_3db_system
        
        custom_data = {
            "entity_id": "custom_user_123",
            "name": "Custom User",
            "email": "custom@example.com"
        }
        
        result = await db3d.create_entity("user", custom_data)
        
        assert result["success"] is True
        assert result["entity_id"] == "custom_user_123"
    
    @pytest.mark.asyncio
    async def test_create_entity_failure_handling(self, mock_3db_system):
        """Test entity creation failure handling."""
        db3d = mock_3db_system
        
        # Mock PostgreSQL failure
        from src.core.base import QueryResult
        db3d.postgresql_db.create.return_value = QueryResult(
            success=False,
            data=None,
            error="Database connection failed"
        )
        
        result = await db3d.create_entity("user", {"name": "Test User"})
        
        assert result["success"] is False
        assert "error" in result or any(not success for success in result.get("results", {}).values())


class TestSimilaritySearch:
    """Test vector similarity search functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_similarity_search(self, mock_3db_system):
        """Test basic similarity search."""
        db3d = mock_3db_system
        
        result = await db3d.search_similar(
            query_text="machine learning and artificial intelligence",
            entity_type="document",
            limit=5
        )
        
        assert result["success"] is True
        assert "results" in result
        assert "execution_time" in result
        assert result["query"] == "machine learning and artificial intelligence"
        
        # Verify query interface was used
        assert db3d.query_interface is not None
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_relationships(self, mock_3db_system):
        """Test similarity search with relationship analysis."""
        db3d = mock_3db_system
        
        result = await db3d.search_similar(
            query_text="database optimization",
            include_relationships=True,
            limit=3
        )
        
        assert result["success"] is True
        assert "results" in result
        # Relationships should be included when requested
        if result["success"]:
            assert "relationships" in result or result.get("relationships") is not None
    
    @pytest.mark.asyncio
    async def test_similarity_search_empty_query(self, mock_3db_system):
        """Test similarity search with empty query."""
        db3d = mock_3db_system
        
        result = await db3d.search_similar(
            query_text="",
            limit=5
        )
        
        # Should handle empty query gracefully
        assert "success" in result
        assert "error" in result or result["success"] is False
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_similarity_search_performance(self, mock_3db_system):
        """Test similarity search performance."""
        db3d = mock_3db_system
        perf_test = PerformanceTest()
        
        perf_test.start_timer("similarity_search")
        
        result = await db3d.search_similar(
            query_text="artificial intelligence and machine learning systems",
            limit=10
        )
        
        perf_test.end_timer("similarity_search")
        
        # Should complete within reasonable time (even for mocked operations)
        perf_test.assert_performance("similarity_search", 1.0)  # 1 second max
        
        assert result is not None


class TestRecommendations:
    """Test AI-powered recommendation functionality."""
    
    @pytest.mark.asyncio
    async def test_get_content_recommendations(self, mock_3db_system):
        """Test getting content recommendations for a user."""
        db3d = mock_3db_system
        
        result = await db3d.get_recommendations(
            user_id="user_123",
            recommendation_type="content",
            limit=10
        )
        
        assert result["success"] is True
        assert result["user_id"] == "user_123"
        assert result["recommendation_type"] == "content"
        assert "recommendations" in result
        assert "execution_time" in result
    
    @pytest.mark.asyncio
    async def test_get_product_recommendations(self, mock_3db_system):
        """Test getting product recommendations for a user."""
        db3d = mock_3db_system
        
        result = await db3d.get_recommendations(
            user_id="user_456",
            recommendation_type="product",
            limit=5
        )
        
        assert result["success"] is True
        assert result["user_id"] == "user_456"
        assert result["recommendation_type"] == "product"
    
    @pytest.mark.asyncio
    async def test_recommendations_nonexistent_user(self, mock_3db_system):
        """Test recommendations for non-existent user."""
        db3d = mock_3db_system
        
        # Mock query interface to return failure for non-existent user
        mock_query_result = MagicMock()
        mock_query_result.success = False
        mock_query_result.error = "User not found"
        
        with patch.object(db3d.query_interface, 'get_recommendations', return_value=mock_query_result):
            result = await db3d.get_recommendations(
                user_id="nonexistent_user",
                recommendation_type="content",
                limit=10
            )
            
            assert result["success"] is False
            assert "error" in result


class TestNetworkAnalysis:
    """Test graph network analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_network_analysis(self, mock_3db_system):
        """Test basic network analysis for an entity."""
        db3d = mock_3db_system
        
        result = await db3d.analyze_entity_network(
            entity_id="user_123",
            max_depth=3
        )
        
        assert result["success"] is True
        assert result["entity_id"] == "user_123"
        assert result["max_depth"] == 3
        assert "network_analysis" in result
        assert "execution_time" in result
    
    @pytest.mark.asyncio
    async def test_network_analysis_max_depth_limit(self, mock_3db_system):
        """Test network analysis with depth limit enforcement."""
        db3d = mock_3db_system
        
        # Test with very high depth
        result = await db3d.analyze_entity_network(
            entity_id="user_123",
            max_depth=20  # Should be limited by system max
        )
        
        assert result["success"] is True
        # Depth should be limited to reasonable maximum
        assert result["max_depth"] <= 10  # System should enforce limits
    
    @pytest.mark.asyncio
    async def test_network_analysis_single_depth(self, mock_3db_system):
        """Test network analysis with depth 1 (immediate connections)."""
        db3d = mock_3db_system
        
        result = await db3d.analyze_entity_network(
            entity_id="user_123",
            max_depth=1
        )
        
        assert result["success"] is True
        assert result["max_depth"] == 1


class TestSynchronization:
    """Test data synchronization functionality."""
    
    @pytest.mark.asyncio
    async def test_sync_status_check(self, mock_3db_system):
        """Test checking synchronization status."""
        db3d = mock_3db_system
        
        # Mock sync handler to return status
        mock_status = {
            "entity_id": "test_entity_123",
            "entity_type": "user",
            "in_postgresql": True,
            "in_vector": True,
            "in_graph": False,
            "fully_synced": False
        }
        
        db3d.sync_handler.get_sync_status.return_value = mock_status
        
        status = await db3d.sync_handler.get_sync_status("test_entity_123")
        
        assert status["entity_id"] == "test_entity_123"
        assert status["in_postgresql"] is True
        assert status["in_vector"] is True
        assert status["in_graph"] is False
        assert status["fully_synced"] is False
    
    @pytest.mark.asyncio
    async def test_manual_sync_trigger(self, mock_3db_system):
        """Test manually triggering synchronization."""
        db3d = mock_3db_system
        
        # Mock metadata manager
        mock_metadata = MagicMock()
        mock_metadata.entity_id = "test_entity_123"
        mock_metadata.entity_type = "user"
        db3d.metadata_manager.get_metadata.return_value = mock_metadata
        
        # Mock sync handler
        db3d.sync_handler.sync_entity.return_value = True
        
        # This would be called via API in real usage
        success = await db3d.sync_handler.sync_entity(
            mock_metadata,
            source_db="postgresql",
            target_dbs=["vector", "graph"]
        )
        
        assert success is True
        db3d.sync_handler.sync_entity.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, mock_config):
        """Test handling of database connection failures."""
        db3d = Database3D(mock_config)
        
        # Mock database connection failures
        with patch.object(db3d.postgresql_db, 'connect', return_value=False):
            success = await db3d.initialize()
            assert success is False
    
    @pytest.mark.asyncio
    async def test_invalid_entity_type(self, mock_3db_system):
        """Test handling of invalid entity types."""
        db3d = mock_3db_system
        
        # Mock sync handler to not have rules for invalid type
        db3d.sync_handler.sync_rules = {"user": {}, "document": {}, "product": {}}
        
        result = await db3d.create_entity("invalid_type", {"name": "Test"})
        
        # Should handle gracefully, even if sync rules don't exist
        assert "success" in result
    
    @pytest.mark.asyncio
    async def test_partial_system_failure(self, mock_3db_system):
        """Test system behavior with partial component failures."""
        db3d = mock_3db_system
        
        # Mock vector database failure
        db3d.vector_db.health_check.return_value = False
        
        health_status = await db3d.health_check()
        
        # System should report degraded but not completely failed
        assert health_status["overall_health"] is False
        assert health_status["components"]["vector"]["healthy"] is False


class TestConcurrency:
    """Test concurrent operations and thread safety."""
    
    @pytest.mark.asyncio
    async def test_concurrent_entity_creation(self, mock_3db_system, sample_entity_data):
        """Test creating multiple entities concurrently."""
        db3d = mock_3db_system
        
        # Create multiple entities concurrently
        tasks = []
        for i in range(5):
            entity_data = sample_entity_data["user"].copy()
            entity_data["name"] = f"User {i}"
            entity_data["email"] = f"user{i}@example.com"
            
            task = db3d.create_entity("user", entity_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_searches(self, mock_3db_system):
        """Test performing multiple searches concurrently."""
        db3d = mock_3db_system
        
        search_queries = [
            "artificial intelligence",
            "machine learning",
            "database systems",
            "distributed computing",
            "data science"
        ]
        
        # Perform searches concurrently
        tasks = [
            db3d.search_similar(query, limit=3)
            for query in search_queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All searches should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
