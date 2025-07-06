# 🧠 3db Unified Database Ecosystem

**The Next Generation Intelligent Database System**

3db is a revolutionary unified database ecosystem that intelligently combines **PostgreSQL** (CRUD operations), **pgvector** (similarity search), and **Apache AGE** (graph relationships) into a cohesive, brain-like database system. It provides automatic synchronization, federated queries, and AI-powered insights across all database types.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green.svg)](https://fastapi.tiangolo.com/)

## 🌟 Key Features

### 🔧 **Unified Database Architecture**
- **PostgreSQL** as the primary CRUD database for structured data
- **pgvector** for semantic similarity search and AI embeddings
- **Apache AGE** for graph relationships and network analysis
- **Automatic synchronization** between all database types

### 🚀 **Advanced Capabilities**
- **Semantic Search**: Vector-based similarity search with AI embeddings
- **Graph Analytics**: Relationship analysis and network traversal
- **AI Recommendations**: Intelligent content and user recommendations
- **Federated Queries**: Cross-database operations with unified results
- **Real-time Sync**: Event-driven and change data capture synchronization

### 🛠️ **Developer Experience**
- **REST API**: Comprehensive FastAPI-based REST interface
- **Python SDK**: Native Python library for direct integration
- **Docker Ready**: Complete containerized deployment
- **Monitoring**: Built-in metrics and health monitoring
- **Testing**: Comprehensive test suite with performance benchmarks

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    3db Unified Database Ecosystem               │
├─────────────────────────────────────────────────────────────────┤
│                        🌐 REST API Layer                        │
│                     (FastAPI + OpenAPI)                         │
├─────────────────────────────────────────────────────────────────┤
│                     🧠 Query Coordinator                        │
│              (Federated Queries + Smart Routing)               │
├─────────────────────────────────────────────────────────────────┤
│                      🔄 Sync Engine                             │
│               (EDA + CDC + Conflict Resolution)                │
├─────────────────────────────────────────────────────────────────┤
│  💾 PostgreSQL    │  🔍 pgvector     │  🕸️ Apache AGE        │
│  (CRUD + ACID)    │  (Embeddings)    │  (Graph + Cypher)      │
│  Structured Data  │  Similarity      │  Relationships         │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **PostgreSQL 15+** (with pgvector and Apache AGE extensions)
- **Redis** (for event streaming)

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/3db-unified-database-ecosystem.git
cd 3db-unified-database-ecosystem
```

### 2. Environment Setup

```bash
# Copy environment configuration
cp config/.env.example .env

# Edit configuration (set your database passwords, etc.)
nano .env
```

### 3. Quick Start with Docker

```bash
# Start the complete system
docker-compose up -d

# Or start with monitoring and admin tools
docker-compose --profile monitoring --profile admin-tools up -d
```

### 4. Verify Installation

```bash
# Check system health
curl http://localhost:8000/health

# Access the API documentation
open http://localhost:8000/docs

# Check system metrics
curl http://localhost:8000/metrics
```

## 💡 Usage Examples

### Python SDK Usage

```python
import asyncio
from src.unified import Database3D

async def main():
    # Initialize 3db system
    db3d = Database3D()
    await db3d.initialize()
    
    # Create a user entity (automatically synced across all databases)
    user_result = await db3d.create_entity('user', {
        'name': 'Alice Johnson',
        'email': 'alice@example.com',
        'bio': 'AI researcher specializing in neural networks',
        'interests': 'machine learning, computer vision, deep learning'
    })
    
    print(f"Created user: {user_result['entity_id']}")
    
    # Perform semantic similarity search
    search_result = await db3d.search_similar(
        query_text='artificial intelligence and machine learning',
        entity_type='user',
        limit=5
    )
    
    print(f"Found {len(search_result['results'])} similar users")
    
    # Get AI-powered recommendations
    recommendations = await db3d.get_recommendations(
        user_id=user_result['entity_id'],
        recommendation_type='content',
        limit=10
    )
    
    print(f"Generated {len(recommendations.get('recommendations', []))} recommendations")
    
    # Analyze entity network
    network_analysis = await db3d.analyze_entity_network(
        entity_id=user_result['entity_id'],
        max_depth=3
    )
    
    print(f"Network analysis completed in {network_analysis['execution_time']:.3f}s")
    
    # Cleanup
    await db3d.shutdown()

asyncio.run(main())
```

### REST API Usage

```bash
# Create a document entity
curl -X POST "http://localhost:8000/entities/" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "document",
    "data": {
      "title": "Introduction to Vector Databases",
      "content": "Vector databases are revolutionizing how we store and query high-dimensional data...",
      "category": "technology",
      "tags": ["databases", "vectors", "AI"]
    }
  }'

# Perform similarity search
curl -X POST "http://localhost:8000/search/similar" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning and artificial intelligence",
    "entity_type": "document",
    "limit": 10,
    "threshold": 0.8
  }'

# Get recommendations for a user
curl -X POST "http://localhost:8000/recommendations/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123456",
    "recommendation_type": "content",
    "limit": 20
  }'

# Analyze entity relationships
curl -X POST "http://localhost:8000/analysis/network" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "user_123456",
    "max_depth": 3
  }'
```

## 🔧 Configuration

### Database Configuration

Edit `.env` file to configure your databases:

```env
# PostgreSQL CRUD Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=unified_3db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here

# Vector Database (can be same as PostgreSQL)
VECTOR_HOST=localhost
VECTOR_PORT=5432
VECTOR_DB=unified_3db_vector
VECTOR_USER=postgres
VECTOR_PASSWORD=your_password_here

# Graph Database (can be same as PostgreSQL)
GRAPH_HOST=localhost
GRAPH_PORT=5432
GRAPH_DB=unified_3db_graph
GRAPH_USER=postgres
GRAPH_PASSWORD=your_password_here

# Event Streaming
REDIS_HOST=localhost
REDIS_PORT=6379
EDA_ENABLED=true
CDC_ENABLED=true

# Vector Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
VECTOR_SIMILARITY_THRESHOLD=0.8
```

### Synchronization Rules

Configure which entity types are synchronized to which databases:

```python
sync_rules = {
    'user': {
        'postgresql': True,      # Always store user data
        'vector': True,          # Generate embeddings for user profiles
        'graph': True,           # Track user relationships
        'embedding_fields': ['name', 'bio', 'interests'],
        'graph_relationships': ['follows', 'friends', 'collaborates']
    },
    'document': {
        'postgresql': True,      # Store document metadata
        'vector': True,          # Enable content similarity search
        'graph': False,          # Documents don't need graph by default
        'embedding_fields': ['title', 'content', 'tags']
    }
}
```

## 🐳 Docker Deployment

### Single Instance Deployment

```bash
# Basic setup (all databases in one PostgreSQL instance)
docker-compose up -d postgresql-main redis
```

### Separate Databases for Scale

```bash
# Separate database instances for better performance
docker-compose --profile separate-databases up -d
```

### Complete Production Setup

```bash
# Full setup with monitoring and admin tools
docker-compose \
  --profile separate-databases \
  --profile monitoring \
  --profile admin-tools \
  --profile with-app \
  up -d
```

### Available Services

| Service | Port | Description |
|---------|------|-------------|
| 3db API | 8000 | Main REST API interface |
| PostgreSQL | 5432 | Primary PostgreSQL database |
| Vector DB | 5433 | Dedicated pgvector instance |
| Graph DB | 5434 | Dedicated Apache AGE instance |
| Redis | 6379 | Event streaming and caching |
| Grafana | 3000 | Monitoring dashboards |
| Prometheus | 9090 | Metrics collection |
| pgAdmin | 5050 | Database administration |
| Redis Commander | 8081 | Redis administration |
| Jupyter | 8888 | Development notebooks |

## 📊 Monitoring & Observability

### Health Monitoring

```bash
# Check overall system health
curl http://localhost:8000/health

# Get detailed metrics
curl http://localhost:8000/metrics
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin123) for:

- **Database Performance**: Query execution times, connection pools
- **Synchronization Status**: Sync success rates, queue lengths
- **Vector Operations**: Embedding generation, similarity searches
- **Graph Analytics**: Traversal performance, centrality metrics
- **System Resources**: CPU, memory, disk usage

### Prometheus Metrics

Key metrics available at `http://localhost:9090`:

- `3db_queries_total` - Total queries executed by database type
- `3db_query_duration_seconds` - Query execution time distribution
- `3db_sync_operations_total` - Synchronization operations
- `3db_entities_total` - Total entities by type and sync status
- `3db_vector_similarity_searches_total` - Vector search operations

## 🧪 Testing

### Run Test Suite

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
pytest -m performance  # Only performance tests
```

### Load Testing

```bash
# Start load testing with Locust
docker-compose --profile load-testing up -d

# Access Locust UI at http://localhost:8089
```

## 🔒 Security Considerations

### Production Security Checklist

- [ ] **Change default passwords** in all configuration files
- [ ] **Enable SSL/TLS** for all database connections
- [ ] **Configure firewall rules** to restrict database access
- [ ] **Use secrets management** for sensitive configuration
- [ ] **Enable API authentication** in production
- [ ] **Regular security updates** for all components
- [ ] **Database backup strategy** implementation
- [ ] **Audit logging** configuration

### Authentication & Authorization

```python
# Configure API authentication
ENABLE_API_AUTH=true
API_SECRET_KEY=your_secure_secret_key_here

# Database connection security
POSTGRES_SSL_MODE=require
REDIS_PASSWORD=your_redis_password
```

## 📈 Performance Optimization

### Database Tuning

1. **PostgreSQL Optimization**:
   - Adjust `shared_buffers`, `work_mem`, `maintenance_work_mem`
   - Configure connection pooling with pgbouncer
   - Create appropriate indexes for query patterns

2. **Vector Search Optimization**:
   - Tune `ivfflat` index parameters for your data size
   - Optimize embedding model for your use case
   - Consider quantization for large-scale deployments

3. **Graph Query Optimization**:
   - Limit traversal depth for performance
   - Use selective graph patterns
   - Consider graph partitioning for large datasets

### Scaling Strategies

- **Horizontal Scaling**: Separate database instances with read replicas
- **Caching**: Redis caching for frequently accessed data
- **Connection Pooling**: Optimize database connection management
- **Async Processing**: Use background tasks for heavy operations

## 🛠️ Development

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start development environment
docker-compose --profile development up -d

# Run the application in development mode
python api/main.py
```

### Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with tests
4. **Run** the test suite (`pytest`)
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Code Quality

```bash
# Format code
black src/ api/ tests/

# Lint code
flake8 src/ api/ tests/

# Type checking
mypy src/ api/

# Security check
bandit -r src/ api/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## 📚 Documentation

### API Documentation

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Additional Resources

- [Architecture Deep Dive](docs/architecture.md)
- [Synchronization Guide](docs/synchronization.md)
- [Performance Tuning](docs/performance.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api-reference.md)

## 🤝 Support & Community

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community Q&A and general discussions
- **Documentation**: Comprehensive guides and tutorials
- **Examples**: Real-world usage examples and patterns

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- **PostgreSQL Team** for the robust relational database
- **pgvector** for vector similarity search capabilities
- **Apache AGE** for graph database functionality
- **FastAPI** for the excellent API framework
- **Sentence Transformers** for embedding models

---

## 🎯 Roadmap

### Version 1.x (Current)
- ✅ Core unified database architecture
- ✅ Automatic synchronization system
- ✅ REST API interface
- ✅ Vector similarity search
- ✅ Graph relationship analysis
- ✅ AI-powered recommendations

### Version 2.x (Planned)
- 🔄 **Multi-tenant Support**: Isolated data for different organizations
- 🔄 **Advanced Analytics**: Built-in OLAP capabilities
- 🔄 **ML Pipeline Integration**: Automated model training and deployment
- 🔄 **Real-time Streaming**: Event-driven real-time data processing
- 🔄 **Advanced Graph Algorithms**: PageRank, community detection, centrality measures

### Version 3.x (Future)
- 🔮 **AI-First Design**: Natural language database queries
- 🔮 **Autonomous Optimization**: Self-tuning performance optimization
- 🔮 **Distributed Architecture**: Multi-region deployment support
- 🔮 **Advanced Security**: Zero-trust security model
- 🔮 **Blockchain Integration**: Immutable audit trails

---

**Built with ❤️ by the 3db Development Team**

*Transforming how applications interact with data through intelligent, unified database ecosystems.*
