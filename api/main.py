"""
3db Unified Database Ecosystem - FastAPI REST API

This module provides a comprehensive REST API interface for the 3db system,
enabling HTTP access to all unified database capabilities including entity
management, similarity search, relationship analysis, and recommendations.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
import structlog

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unified import Database3D
from core.config import get_config
from utils.logging import get_logger

# Initialize logger
logger = get_logger("3db_api")

# Global 3db instance
db3d: Optional[Database3D] = None


# =====================================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# =====================================================================================

class EntityCreateRequest(BaseModel):
    """Request model for entity creation."""
    entity_type: str = Field(..., description="Type of entity (e.g., 'user', 'document', 'product')")
    data: Dict[str, Any] = Field(..., description="Entity data as key-value pairs")
    sync_immediately: bool = Field(True, description="Whether to synchronize immediately")
    
    class Config:
        schema_extra = {
            "example": {
                "entity_type": "user",
                "data": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "bio": "Software engineer with expertise in databases",
                    "interests": "databases, AI, distributed systems"
                },
                "sync_immediately": True
            }
        }


class EntityResponse(BaseModel):
    """Response model for entity operations."""
    entity_id: str
    success: bool
    results: Optional[Dict[str, bool]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SimilaritySearchRequest(BaseModel):
    """Request model for similarity search."""
    query_text: str = Field(..., description="Text query for similarity search")
    entity_type: Optional[str] = Field(None, description="Optional entity type filter")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    include_relationships: bool = Field(False, description="Include relationship analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "query_text": "machine learning and artificial intelligence",
                "entity_type": "document",
                "limit": 10,
                "threshold": 0.8,
                "include_relationships": False
            }
        }


class SimilaritySearchResponse(BaseModel):
    """Response model for similarity search."""
    query: str
    success: bool
    results: Optional[Dict[str, Any]] = None
    relationships: Optional[Dict[str, Any]] = None
    execution_time: float
    error: Optional[str] = None


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_id: str = Field(..., description="User entity ID for recommendations")
    recommendation_type: str = Field("content", description="Type of recommendations")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123456",
                "recommendation_type": "content",
                "limit": 20
            }
        }


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: str
    recommendation_type: str
    success: bool
    recommendations: Optional[Dict[str, Any]] = None
    execution_time: float
    error: Optional[str] = None


class NetworkAnalysisRequest(BaseModel):
    """Request model for network analysis."""
    entity_id: str = Field(..., description="Entity ID for network analysis")
    max_depth: int = Field(3, ge=1, le=10, description="Maximum traversal depth")
    
    class Config:
        schema_extra = {
            "example": {
                "entity_id": "user_123456",
                "max_depth": 3
            }
        }


class NetworkAnalysisResponse(BaseModel):
    """Response model for network analysis."""
    entity_id: str
    max_depth: int
    success: bool
    network_analysis: Optional[Dict[str, Any]] = None
    execution_time: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    components: Dict[str, Dict[str, Any]]
    overall_health: bool


class MetricsResponse(BaseModel):
    """Response model for system metrics."""
    timestamp: datetime
    system: Dict[str, Any]
    databases: Dict[str, Any]
    synchronization: Dict[str, Any]


# =====================================================================================
# APPLICATION LIFECYCLE
# =====================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global db3d
    
    # Startup
    logger.info("🚀 Starting 3db API server")
    try:
        db3d = Database3D()
        success = await db3d.initialize()
        
        if success:
            logger.info("✅ 3db system initialized successfully")
        else:
            logger.error("❌ Failed to initialize 3db system")
            raise Exception("3db initialization failed")
    
    except Exception as e:
        logger.error(f"💥 Failed to start 3db API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down 3db API server")
    if db3d:
        await db3d.shutdown()
    logger.info("👋 3db API server shutdown complete")


# =====================================================================================
# FASTAPI APPLICATION SETUP
# =====================================================================================

# Create FastAPI app
app = FastAPI(
    title="3db Unified Database Ecosystem API",
    description="""
    🧠 **3db Unified Database Ecosystem REST API**
    
    A comprehensive API for the intelligent database system that combines:
    - **PostgreSQL** for CRUD operations
    - **pgvector** for similarity search
    - **Apache AGE** for graph relationships
    
    ## Features
    
    - 🔧 **Entity Management**: Create, read, update entities across all databases
    - 🔍 **Similarity Search**: Vector-based semantic search with embeddings
    - 🕸️ **Relationship Analysis**: Graph traversal and network analysis
    - 🎯 **AI Recommendations**: Intelligent content and user recommendations
    - 📊 **System Monitoring**: Real-time metrics and health monitoring
    - 🔄 **Automatic Synchronization**: Seamless data consistency across databases
    
    ## Getting Started
    
    1. Create entities using the `/entities/` endpoint
    2. Perform similarity searches with `/search/similar`
    3. Analyze relationships via `/analysis/network`
    4. Get recommendations through `/recommendations/`
    5. Monitor system health at `/health`
    """,
    version="1.0.0",
    contact={
        "name": "3db Development Team",
        "email": "support@3db.ai",
        "url": "https://github.com/your-org/3db"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# =====================================================================================
# DEPENDENCY INJECTION
# =====================================================================================

def get_db3d() -> Database3D:
    """Dependency to get the 3db instance."""
    if db3d is None:
        raise HTTPException(
            status_code=503,
            detail="3db system not initialized or unavailable"
        )
    return db3d


# =====================================================================================
# API ENDPOINTS
# =====================================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🧠 3db Unified Database Ecosystem API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check(db: Database3D = Depends(get_db3d)):
    """
    Comprehensive health check of all system components.
    
    Returns the health status of:
    - PostgreSQL database
    - Vector database (pgvector)
    - Graph database (Apache AGE)
    - Event broker (Redis)
    - Synchronization system
    """
    try:
        health_data = await db.health_check()
        
        return HealthResponse(
            status="healthy" if health_data["overall_health"] else "degraded",
            timestamp=datetime.utcnow(),
            components=health_data["components"],
            overall_health=health_data["overall_health"]
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(db: Database3D = Depends(get_db3d)):
    """
    Get comprehensive system metrics including:
    - Database performance metrics
    - Synchronization statistics
    - Queue status
    - System resource usage
    """
    try:
        metrics_data = await db.get_system_metrics()
        
        return MetricsResponse(
            timestamp=datetime.utcnow(),
            system=metrics_data.get("system", {}),
            databases=metrics_data.get("databases", {}),
            synchronization=metrics_data.get("synchronization", {})
        )
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")


# =====================================================================================
# ENTITY MANAGEMENT ENDPOINTS
# =====================================================================================

@app.post("/entities/", response_model=EntityResponse, tags=["Entity Management"])
async def create_entity(
    request: EntityCreateRequest,
    background_tasks: BackgroundTasks,
    db: Database3D = Depends(get_db3d)
):
    """
    Create a new entity across all relevant databases.
    
    The entity will be automatically:
    - Stored in PostgreSQL for structured data
    - Vectorized for similarity search (if applicable)
    - Added to graph for relationship tracking (if applicable)
    
    Synchronization happens automatically based on entity type.
    """
    try:
        result = await db.create_entity(
            entity_type=request.entity_type,
            entity_data=request.data,
            sync_immediately=request.sync_immediately
        )
        
        return EntityResponse(**result)
    
    except Exception as e:
        logger.error(f"Failed to create entity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create entity: {e}")


@app.get("/entities/{entity_id}", tags=["Entity Management"])
async def get_entity(
    entity_id: str = Path(..., description="Entity ID to retrieve"),
    db: Database3D = Depends(get_db3d)
):
    """
    Retrieve an entity by ID with data from all databases.
    
    Returns unified view including:
    - PostgreSQL structured data
    - Vector embedding information
    - Graph relationship data
    """
    try:
        # Get entity data from PostgreSQL
        pg_result = await db.postgresql_db.read("entities", {"entity_id": entity_id})
        
        if not pg_result.success or not pg_result.data:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        entity_data = pg_result.data[0]
        
        # Get vector embedding if available
        vector_result = await db.vector_db.get_embedding(entity_id)
        if vector_result.success:
            entity_data["embedding_info"] = {
                "has_embedding": True,
                "created_at": vector_result.data.get("created_at")
            }
        
        # Get sync status
        sync_status = await db.sync_handler.get_sync_status(entity_id)
        entity_data["sync_status"] = sync_status
        
        return {"entity": entity_data}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entity: {e}")


# =====================================================================================
# SEARCH ENDPOINTS
# =====================================================================================

@app.post("/search/similar", response_model=SimilaritySearchResponse, tags=["Search"])
async def similarity_search(
    request: SimilaritySearchRequest,
    db: Database3D = Depends(get_db3d)
):
    """
    Perform semantic similarity search using vector embeddings.
    
    Finds entities similar to the query text based on:
    - Semantic meaning (not just keyword matching)
    - Vector embeddings and cosine similarity
    - Optional entity type filtering
    - Configurable similarity threshold
    """
    try:
        result = await db.search_similar(
            query_text=request.query_text,
            entity_type=request.entity_type,
            limit=request.limit,
            include_relationships=request.include_relationships
        )
        
        return SimilaritySearchResponse(**result)
    
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")


@app.get("/search/entities", tags=["Search"])
async def search_entities(
    q: str = Query(..., description="Search query"),
    entity_type: Optional[str] = Query(None, description="Entity type filter"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    db: Database3D = Depends(get_db3d)
):
    """
    General entity search across PostgreSQL with optional vector similarity.
    
    Supports both traditional text search and semantic similarity.
    """
    try:
        # Basic PostgreSQL search
        if entity_type:
            table_name = f"{entity_type}s"
        else:
            table_name = "entities"
        
        # Simple text search (can be enhanced with full-text search)
        query = f"""
            SELECT * FROM {table_name} 
            WHERE name ILIKE $1 OR description ILIKE $1
            LIMIT $2
        """
        
        search_term = f"%{q}%"
        result = await db.postgresql_db.execute_query(query, [search_term, limit])
        
        if result.success:
            return {"query": q, "results": result.data, "count": len(result.data)}
        else:
            raise HTTPException(status_code=500, detail="Search failed")
    
    except Exception as e:
        logger.error(f"Entity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity search failed: {e}")


# =====================================================================================
# RECOMMENDATION ENDPOINTS
# =====================================================================================

@app.post("/recommendations/", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    db: Database3D = Depends(get_db3d)
):
    """
    Get AI-powered recommendations for a user.
    
    Uses a combination of:
    - User profile and preferences (PostgreSQL)
    - Content similarity (vector embeddings)
    - Social relationships (graph analysis)
    - Collaborative filtering
    """
    try:
        result = await db.get_recommendations(
            user_id=request.user_id,
            recommendation_type=request.recommendation_type,
            limit=request.limit
        )
        
        return RecommendationResponse(**result)
    
    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {e}")


# =====================================================================================
# ANALYSIS ENDPOINTS
# =====================================================================================

@app.post("/analysis/network", response_model=NetworkAnalysisResponse, tags=["Analysis"])
async def network_analysis(
    request: NetworkAnalysisRequest,
    db: Database3D = Depends(get_db3d)
):
    """
    Analyze the network of relationships around an entity.
    
    Performs graph traversal to discover:
    - Direct relationships
    - Multi-hop connections
    - Influential entities in the network
    - Community structures
    """
    try:
        result = await db.analyze_entity_network(
            entity_id=request.entity_id,
            max_depth=request.max_depth
        )
        
        return NetworkAnalysisResponse(**result)
    
    except Exception as e:
        logger.error(f"Network analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Network analysis failed: {e}")


@app.get("/analysis/influential", tags=["Analysis"])
async def get_influential_entities(
    entity_type: Optional[str] = Query(None, description="Entity type filter"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    db: Database3D = Depends(get_db3d)
):
    """
    Find the most influential entities based on graph centrality measures.
    
    Identifies entities with:
    - Highest degree centrality (most connections)
    - Betweenness centrality (bridge entities)
    - PageRank scores
    """
    try:
        result = await db.graph_db.find_influential_nodes(limit=limit)
        
        if result.success:
            return {"influential_entities": result.data, "limit": limit}
        else:
            raise HTTPException(status_code=500, detail="Failed to find influential entities")
    
    except Exception as e:
        logger.error(f"Influential entities analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# =====================================================================================
# ADMIN ENDPOINTS
# =====================================================================================

@app.post("/admin/sync/{entity_id}", tags=["Administration"])
async def trigger_sync(
    entity_id: str = Path(..., description="Entity ID to synchronize"),
    force: bool = Query(False, description="Force synchronization even if up-to-date"),
    db: Database3D = Depends(get_db3d)
):
    """
    Manually trigger synchronization for a specific entity.
    
    Useful for:
    - Fixing sync inconsistencies
    - Updating stale data
    - Testing sync functionality
    """
    try:
        # Get entity metadata
        metadata = await db.metadata_manager.get_metadata(entity_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        # Trigger sync across all databases
        from src.core.base import DatabaseType
        
        success = await db.sync_handler.sync_entity(
            metadata,
            source_db=DatabaseType.POSTGRESQL,
            target_dbs=[DatabaseType.VECTOR, DatabaseType.GRAPH]
        )
        
        return {
            "entity_id": entity_id,
            "sync_triggered": True,
            "success": success,
            "timestamp": datetime.utcnow()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger sync for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Sync trigger failed: {e}")


@app.get("/admin/sync/status", tags=["Administration"])
async def get_sync_status(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    db: Database3D = Depends(get_db3d)
):
    """
    Get synchronization status overview for all entities.
    
    Shows:
    - Entities in each database
    - Sync consistency status
    - Recent sync errors
    - Performance metrics
    """
    try:
        stats = await db.metadata_manager.get_sync_statistics()
        
        # Filter by entity type if specified
        if entity_type:
            entities = await db.metadata_manager.list_entities_by_type(entity_type)
            type_stats = {
                "entity_type": entity_type,
                "total_entities": len(entities),
                "entities": [
                    {
                        "entity_id": e.entity_id,
                        "in_postgresql": e.in_postgresql,
                        "in_vector": e.in_vector,
                        "in_graph": e.in_graph,
                        "has_errors": e.has_sync_errors()
                    }
                    for e in entities
                ]
            }
            return type_stats
        
        return stats
    
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sync status: {e}")


# =====================================================================================
# ERROR HANDLERS
# =====================================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =====================================================================================
# CUSTOM OPENAPI SCHEMA
# =====================================================================================

def custom_openapi():
    """Custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="3db Unified Database Ecosystem API",
        version="1.0.0",
        description="Intelligent database system combining PostgreSQL, pgvector, and Apache AGE",
        routes=app.routes,
    )
    
    # Add custom schema extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://your-domain.com/logo.png"
    }
    
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.3db.ai", "description": "Production server"}
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# =====================================================================================
# DEVELOPMENT SERVER
# =====================================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
