"""
3db Unified Database Ecosystem - Complete Usage Example

This script demonstrates the full capabilities of the 3db system,
showing how to create entities, perform searches, analyze relationships,
and get recommendations across PostgreSQL, pgvector, and Apache AGE.
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Any

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unified import Database3D, create_3db_system
from core.config import get_config
from utils.logging import get_logger

logger = get_logger("3db_example")


class Demo3db:
    """Comprehensive demonstration of 3db capabilities."""
    
    def __init__(self):
        self.db3d: Database3D = None
        self.demo_entities = {}
    
    async def initialize(self):
        """Initialize the 3db system."""
        print("🚀 Initializing 3db Unified Database Ecosystem")
        print("=" * 60)
        
        try:
            self.db3d = Database3D()
            success = await self.db3d.initialize()
            
            if success:
                print("✅ 3db system initialized successfully!")
                
                # Display system health
                health = await self.db3d.health_check()
                print(f"🏥 System Health: {'✅ All systems healthy' if health['overall_health'] else '⚠️ Some issues detected'}")
                
                # Display component status
                for component, status in health['components'].items():
                    indicator = "✅" if status['healthy'] else "❌"
                    print(f"   {indicator} {component.title().replace('_', ' ')}")
                
                return True
            else:
                print("❌ Failed to initialize 3db system")
                return False
                
        except Exception as e:
            print(f"💥 Initialization error: {e}")
            return False
    
    async def demo_entity_creation(self):
        """Demonstrate entity creation across all databases."""
        print("\n📝 DEMO: Entity Creation")
        print("-" * 40)
        
        # Create sample users
        users = [
            {
                'name': 'Dr. Sarah Chen',
                'email': 'sarah.chen@techcorp.com',
                'bio': 'Machine learning researcher specializing in neural networks and deep learning architectures',
                'interests': 'artificial intelligence, machine learning, neural networks, computer vision',
                'role': 'Senior AI Researcher',
                'department': 'R&D'
            },
            {
                'name': 'Marcus Rodriguez',
                'email': 'marcus.r@innovate.io',
                'bio': 'Full-stack developer with expertise in distributed systems and database optimization',
                'interests': 'distributed systems, databases, microservices, cloud computing',
                'role': 'Lead Backend Engineer',
                'department': 'Engineering'
            },
            {
                'name': 'Elena Kowalski',
                'email': 'elena.k@dataflow.ai',
                'bio': 'Data scientist focused on recommendation systems and user behavior analysis',
                'interests': 'data science, recommendation systems, statistics, user analytics',
                'role': 'Senior Data Scientist',
                'department': 'Analytics'
            }
        ]
        
        print("Creating user entities...")
        for user_data in users:
            result = await self.db3d.create_entity('user', user_data)
            if result['success']:
                self.demo_entities[user_data['name']] = result['entity_id']
                print(f"✅ Created user: {user_data['name']} (ID: {result['entity_id']})")
            else:
                print(f"❌ Failed to create user: {user_data['name']}")
        
        # Create sample documents
        documents = [
            {
                'title': 'Advanced Neural Network Architectures for Computer Vision',
                'content': 'This comprehensive guide explores state-of-the-art neural network architectures specifically designed for computer vision tasks. We cover convolutional neural networks, transformer-based vision models, and attention mechanisms.',
                'author_id': self.demo_entities.get('Dr. Sarah Chen'),
                'category': 'artificial-intelligence',
                'tags': ['neural networks', 'computer vision', 'deep learning', 'AI'],
                'document_type': 'research-paper',
                'status': 'published'
            },
            {
                'title': 'Building Scalable Database Systems: A Practical Guide',
                'content': 'Learn how to design and implement database systems that can handle massive scale. This guide covers sharding strategies, replication patterns, consistency models, and performance optimization techniques.',
                'author_id': self.demo_entities.get('Marcus Rodriguez'),
                'category': 'database-systems',
                'tags': ['databases', 'scalability', 'distributed systems', 'performance'],
                'document_type': 'technical-guide',
                'status': 'published'
            },
            {
                'title': 'Modern Recommendation Systems: From Collaborative Filtering to Deep Learning',
                'content': 'Explore the evolution of recommendation systems from traditional collaborative filtering to modern deep learning approaches. Includes case studies and implementation examples.',
                'author_id': self.demo_entities.get('Elena Kowalski'),
                'category': 'data-science',
                'tags': ['recommendation systems', 'machine learning', 'data science', 'algorithms'],
                'document_type': 'tutorial',
                'status': 'published'
            }
        ]
        
        print("\nCreating document entities...")
        for doc_data in documents:
            result = await self.db3d.create_entity('document', doc_data)
            if result['success']:
                self.demo_entities[doc_data['title']] = result['entity_id']
                print(f"✅ Created document: {doc_data['title'][:50]}...")
            else:
                print(f"❌ Failed to create document: {doc_data['title'][:50]}...")
        
        # Create sample products
        products = [
            {
                'name': 'AI Development Platform Pro',
                'description': 'Comprehensive platform for building, training, and deploying machine learning models with built-in MLOps capabilities',
                'category': 'ai-tools',
                'price': 999.99,
                'currency': 'USD',
                'features': ['AutoML', 'Model Deployment', 'Data Pipeline', 'Monitoring'],
                'target_audience': 'data scientists, ML engineers',
                'availability_status': 'available'
            },
            {
                'name': 'Database Performance Optimizer',
                'description': 'Advanced database optimization tool that automatically identifies performance bottlenecks and suggests improvements',
                'category': 'database-tools',
                'price': 599.99,
                'currency': 'USD',
                'features': ['Query Optimization', 'Index Tuning', 'Performance Monitoring', 'Automated Scaling'],
                'target_audience': 'database administrators, backend engineers',
                'availability_status': 'available'
            },
            {
                'name': 'Recommendation Engine SDK',
                'description': 'Easy-to-integrate recommendation engine with pre-built algorithms and real-time personalization capabilities',
                'category': 'analytics-tools',
                'price': 449.99,
                'currency': 'USD',
                'features': ['Real-time Recommendations', 'A/B Testing', 'Analytics Dashboard', 'API Integration'],
                'target_audience': 'product managers, data scientists',
                'availability_status': 'available'
            }
        ]
        
        print("\nCreating product entities...")
        for product_data in products:
            result = await self.db3d.create_entity('product', product_data)
            if result['success']:
                self.demo_entities[product_data['name']] = result['entity_id']
                print(f"✅ Created product: {product_data['name']}")
            else:
                print(f"❌ Failed to create product: {product_data['name']}")
        
        print(f"\n📊 Created {len(self.demo_entities)} entities across all databases")
    
    async def demo_similarity_search(self):
        """Demonstrate vector similarity search capabilities."""
        print("\n🔍 DEMO: Similarity Search")
        print("-" * 40)
        
        # Test various similarity searches
        search_queries = [
            "machine learning and artificial intelligence",
            "database optimization and performance",
            "recommendation systems and user analytics",
            "distributed systems and scalability",
            "computer vision and deep learning"
        ]
        
        for query in search_queries:
            print(f"\n🔎 Searching for: '{query}'")
            
            result = await self.db3d.search_similar(
                query_text=query,
                limit=3,
                include_relationships=False
            )
            
            if result['success']:
                if isinstance(result['results'], dict) and 'results' in result['results']:
                    vector_results = result['results']['results'].get('vector', [])
                    
                    if vector_results:
                        print("   📋 Top matches:")
                        for i, match in enumerate(vector_results[:3], 1):
                            similarity = match.get('similarity', 0)
                            entity_id = match.get('entity_id', 'unknown')
                            print(f"   {i}. Entity: {entity_id} (Similarity: {similarity:.3f})")
                    else:
                        print("   📭 No matches found")
                else:
                    print("   ⚠️ Unexpected result format")
            else:
                print(f"   ❌ Search failed: {result.get('error', 'Unknown error')}")
            
            # Small delay between searches
            await asyncio.sleep(0.5)
    
    async def demo_relationship_analysis(self):
        """Demonstrate graph relationship analysis."""
        print("\n🕸️ DEMO: Relationship Analysis")
        print("-" * 40)
        
        # Analyze relationships for each created user
        for user_name, entity_id in self.demo_entities.items():
            if 'Dr.' in user_name or 'Marcus' in user_name or 'Elena' in user_name:
                print(f"\n👤 Analyzing relationships for: {user_name}")
                
                result = await self.db3d.analyze_entity_network(
                    entity_id=entity_id,
                    max_depth=2
                )
                
                if result['success']:
                    network_data = result.get('network_analysis')
                    if network_data:
                        print(f"   🔗 Network analysis completed")
                        print(f"   ⏱️  Execution time: {result['execution_time']:.3f}s")
                        
                        # Display network statistics if available
                        if isinstance(network_data, dict):
                            if 'final_data' in network_data:
                                print(f"   📊 Operations executed: {network_data.get('operations_executed', 0)}")
                    else:
                        print("   📭 No network data found")
                else:
                    print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    async def demo_recommendations(self):
        """Demonstrate AI-powered recommendations."""
        print("\n🎯 DEMO: AI-Powered Recommendations")
        print("-" * 40)
        
        # Get recommendations for each user
        for user_name, entity_id in self.demo_entities.items():
            if 'Dr.' in user_name or 'Marcus' in user_name or 'Elena' in user_name:
                print(f"\n🤖 Getting recommendations for: {user_name}")
                
                result = await self.db3d.get_recommendations(
                    user_id=entity_id,
                    recommendation_type='content',
                    limit=5
                )
                
                if result['success']:
                    recommendations = result.get('recommendations')
                    if recommendations:
                        print(f"   ✅ Recommendations generated")
                        print(f"   ⏱️  Execution time: {result['execution_time']:.3f}s")
                        
                        # Display recommendation details if available
                        if isinstance(recommendations, dict):
                            final_data = recommendations.get('final_data')
                            if final_data:
                                print(f"   📈 Recommendation data available")
                    else:
                        print("   📭 No recommendations generated")
                else:
                    print(f"   ❌ Recommendation failed: {result.get('error', 'Unknown error')}")
    
    async def demo_system_metrics(self):
        """Demonstrate system monitoring and metrics."""
        print("\n📊 DEMO: System Metrics & Monitoring")
        print("-" * 40)
        
        # Get comprehensive system metrics
        metrics = await self.db3d.get_system_metrics()
        
        print("🔧 System Status:")
        system_info = metrics.get('system', {})
        print(f"   Initialized: {'✅' if system_info.get('initialized') else '❌'}")
        print(f"   Running: {'✅' if system_info.get('running') else '❌'}")
        print(f"   Active sync tasks: {system_info.get('active_sync_tasks', 0)}")
        
        print("\n💾 Database Metrics:")
        db_metrics = metrics.get('databases', {})
        
        for db_name, db_data in db_metrics.items():
            if isinstance(db_data, dict):
                print(f"   📊 {db_name.upper()}:")
                print(f"      Queries executed: {db_data.get('queries_executed', 0)}")
                print(f"      Queries failed: {db_data.get('queries_failed', 0)}")
                print(f"      Avg execution time: {db_data.get('avg_execution_time', 0):.3f}s")
                
                # Vector-specific metrics
                if db_name == 'vector' and 'embedding_stats' in db_data:
                    embedding_stats = db_data['embedding_stats']
                    if isinstance(embedding_stats, dict):
                        print(f"      Total embeddings: {embedding_stats.get('total_embeddings', 0)}")
                        print(f"      Entity types: {embedding_stats.get('unique_entity_types', 0)}")
                
                # Graph-specific metrics
                if db_name == 'graph' and 'graph_stats' in db_data:
                    graph_stats = db_data['graph_stats']
                    if isinstance(graph_stats, dict):
                        print(f"      Nodes: {graph_stats.get('node_count', 0)}")
                        print(f"      Edges: {graph_stats.get('edge_count', 0)}")
        
        print("\n🔄 Synchronization Metrics:")
        sync_metrics = metrics.get('synchronization', {})
        if sync_metrics:
            print(f"   Total entities: {sync_metrics.get('total_entities', 0)}")
            print(f"   In PostgreSQL: {sync_metrics.get('in_postgresql', 0)}")
            print(f"   In Vector DB: {sync_metrics.get('in_vector', 0)}")
            print(f"   In Graph DB: {sync_metrics.get('in_graph', 0)}")
            print(f"   Fully synced: {sync_metrics.get('fully_synced', 0)}")
        
        # Queue statistics if available
        queue_stats = sync_metrics.get('queue_stats', {})
        if queue_stats:
            print("\n📥 Queue Statistics:")
            for queue_name, count in queue_stats.items():
                print(f"   {queue_name.replace('_', ' ').title()}: {count}")
    
    async def demo_federated_queries(self):
        """Demonstrate complex federated queries."""
        print("\n🌐 DEMO: Federated Queries")
        print("-" * 40)
        
        print("🔗 Executing federated join query...")
        
        try:
            # Example federated query combining PostgreSQL and vector data
            result = await self.db3d.query_interface.federated_join(
                postgresql_query="SELECT entity_id, name, email FROM users WHERE status = 'active'",
                vector_query=None,  # Would include vector similarity if needed
                graph_query=None   # Would include graph traversal if needed
            )
            
            if result.success:
                print("   ✅ Federated query completed successfully")
                print(f"   ⏱️  Total execution time: {result.total_execution_time:.3f}s")
                
                # Display execution times by database
                for db_name, exec_time in result.database_execution_times.items():
                    print(f"   📊 {db_name}: {exec_time:.3f}s")
                
                # Display data summary
                if result.data:
                    total_rows = result.data.get('total_rows', 0)
                    databases_queried = result.data.get('databases_queried', [])
                    print(f"   📈 Total rows processed: {total_rows}")
                    print(f"   🎯 Databases queried: {', '.join(databases_queried)}")
            else:
                print(f"   ❌ Federated query failed: {result.error}")
                
        except Exception as e:
            print(f"   💥 Federated query error: {e}")
    
    async def run_complete_demo(self):
        """Run the complete 3db demonstration."""
        try:
            # Initialize system
            if not await self.initialize():
                return False
            
            # Run all demonstrations
            await self.demo_entity_creation()
            await self.demo_similarity_search()
            await self.demo_relationship_analysis()
            await self.demo_recommendations()
            await self.demo_federated_queries()
            await self.demo_system_metrics()
            
            print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("The 3db Unified Database Ecosystem demonstration showcased:")
            print("✅ Multi-database entity creation and synchronization")
            print("✅ Vector similarity search with semantic understanding")
            print("✅ Graph relationship analysis and traversal")
            print("✅ AI-powered recommendation generation")
            print("✅ Federated queries across database types")
            print("✅ Real-time system monitoring and metrics")
            print("\n🚀 The 3db system is ready for production use!")
            
            return True
            
        except Exception as e:
            print(f"\n💥 Demo failed with error: {e}")
            return False
        
        finally:
            # Clean shutdown
            if self.db3d:
                await self.db3d.shutdown()
                print("\n👋 System shutdown complete")


async def main():
    """Main demonstration function."""
    
    print("🧠 3db Unified Database Ecosystem")
    print("  Complete Demonstration & Example Usage")
    print("=" * 60)
    print()
    
    demo = Demo3db()
    success = await demo.run_complete_demo()
    
    if success:
        print("\n✨ Thank you for exploring the 3db Unified Database Ecosystem!")
        print("Visit the documentation for more advanced usage examples.")
    else:
        print("\n⚠️ Demo encountered issues. Please check your configuration and try again.")
    
    return success


if __name__ == "__main__":
    # Run the complete demonstration
    try:
        success = asyncio.run(main())
        exit_code = 0 if success else 1
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit(1)
