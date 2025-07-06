#!/usr/bin/env python3
"""
3db Unified Database Ecosystem - Command Line Interface

A comprehensive CLI tool for managing, monitoring, and interacting with the 3db system.
Provides commands for entity management, system administration, and development tasks.
"""

import asyncio
import sys
import os
import json
import click
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import csv

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unified import Database3D
from core.config import get_config, reload_config
from utils.logging import get_logger

logger = get_logger("3db_cli")


class ClickFormatter:
    """Custom formatter for CLI output with colors."""
    
    @staticmethod
    def success(message: str) -> str:
        return click.style(message, fg='green', bold=True)
    
    @staticmethod
    def error(message: str) -> str:
        return click.style(message, fg='red', bold=True)
    
    @staticmethod
    def warning(message: str) -> str:
        return click.style(message, fg='yellow', bold=True)
    
    @staticmethod
    def info(message: str) -> str:
        return click.style(message, fg='blue', bold=True)
    
    @staticmethod
    def header(message: str) -> str:
        return click.style(message, fg='cyan', bold=True)


fmt = ClickFormatter()


class CLI3db:
    """Main CLI class for 3db operations."""
    
    def __init__(self):
        self.db3d: Optional[Database3D] = None
        self.config_loaded = False
    
    async def initialize(self, config_path: Optional[str] = None) -> bool:
        """Initialize the 3db system."""
        try:
            if config_path:
                reload_config(config_path)
            
            self.db3d = Database3D()
            success = await self.db3d.initialize()
            
            if success:
                self.config_loaded = True
                return True
            else:
                click.echo(fmt.error("Failed to initialize 3db system"))
                return False
                
        except Exception as e:
            click.echo(fmt.error(f"Initialization error: {e}"))
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        if self.db3d:
            await self.db3d.shutdown()


# Global CLI instance
cli_instance = CLI3db()


# =====================================================================================
# MAIN CLI GROUP
# =====================================================================================

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """
    🧠 3db Unified Database Ecosystem CLI
    
    A comprehensive command-line interface for managing and interacting
    with the 3db unified database system.
    
    Examples:
        3db status                          # Check system status
        3db entity create user.json        # Create entity from file
        3db search "machine learning"       # Search for similar content
        3db sync status                     # Check synchronization status
        3db metrics                         # Show system metrics
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        click.echo(fmt.info("🧠 3db CLI - Verbose mode enabled"))


# =====================================================================================
# SYSTEM MANAGEMENT COMMANDS
# =====================================================================================

@cli.command()
@click.pass_context
def status(ctx):
    """Check the health status of all 3db components."""
    
    async def _status():
        click.echo(fmt.header("🏥 3db System Health Check"))
        click.echo("=" * 50)
        
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            health_data = await cli_instance.db3d.health_check()
            
            # Overall status
            overall = "✅ Healthy" if health_data['overall_health'] else "❌ Degraded"
            click.echo(f"Overall Status: {overall}")
            click.echo(f"Timestamp: {health_data['timestamp']}")
            click.echo()
            
            # Component status
            click.echo("📊 Component Status:")
            for component, status in health_data['components'].items():
                indicator = "✅" if status['healthy'] else "❌"
                click.echo(f"  {indicator} {component.title().replace('_', ' ')}")
                
                if ctx.obj['verbose'] and isinstance(status, dict):
                    for key, value in status.items():
                        if key != 'healthy':
                            click.echo(f"      {key}: {value}")
            
        except Exception as e:
            click.echo(fmt.error(f"Health check failed: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_status())


@cli.command()
@click.pass_context
def metrics(ctx):
    """Display comprehensive system metrics."""
    
    async def _metrics():
        click.echo(fmt.header("📊 3db System Metrics"))
        click.echo("=" * 50)
        
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            metrics_data = await cli_instance.db3d.get_system_metrics()
            
            # System metrics
            system_info = metrics_data.get('system', {})
            click.echo("🔧 System Information:")
            click.echo(f"  Initialized: {'✅' if system_info.get('initialized') else '❌'}")
            click.echo(f"  Running: {'✅' if system_info.get('running') else '❌'}")
            click.echo(f"  Active sync tasks: {system_info.get('active_sync_tasks', 0)}")
            click.echo()
            
            # Database metrics
            db_metrics = metrics_data.get('databases', {})
            click.echo("💾 Database Performance:")
            
            for db_name, db_data in db_metrics.items():
                if isinstance(db_data, dict):
                    click.echo(f"  📊 {db_name.upper()}:")
                    click.echo(f"    Queries executed: {db_data.get('queries_executed', 0)}")
                    click.echo(f"    Queries failed: {db_data.get('queries_failed', 0)}")
                    click.echo(f"    Avg execution time: {db_data.get('avg_execution_time', 0):.3f}s")
                    
                    # Additional metrics for specific databases
                    if db_name == 'vector' and 'embedding_stats' in db_data:
                        embedding_stats = db_data['embedding_stats']
                        if isinstance(embedding_stats, dict):
                            click.echo(f"    Total embeddings: {embedding_stats.get('total_embeddings', 0)}")
                    
                    if db_name == 'graph' and 'graph_stats' in db_data:
                        graph_stats = db_data['graph_stats']
                        if isinstance(graph_stats, dict):
                            click.echo(f"    Nodes: {graph_stats.get('node_count', 0)}")
                            click.echo(f"    Edges: {graph_stats.get('edge_count', 0)}")
            
            # Synchronization metrics
            sync_metrics = metrics_data.get('synchronization', {})
            if sync_metrics:
                click.echo()
                click.echo("🔄 Synchronization Status:")
                click.echo(f"  Total entities: {sync_metrics.get('total_entities', 0)}")
                click.echo(f"  Fully synced: {sync_metrics.get('fully_synced', 0)}")
                click.echo(f"  With errors: {sync_metrics.get('entities_with_errors', 0)}")
        
        except Exception as e:
            click.echo(fmt.error(f"Failed to get metrics: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_metrics())


# =====================================================================================
# ENTITY MANAGEMENT COMMANDS
# =====================================================================================

@cli.group()
def entity():
    """Entity management operations."""
    pass


@entity.command()
@click.argument('entity_type')
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--sync/--no-sync', default=True, help='Enable/disable immediate synchronization')
@click.pass_context
def create(ctx, entity_type, data_file, sync):
    """Create an entity from a JSON file."""
    
    async def _create():
        click.echo(fmt.header(f"📝 Creating {entity_type} entity"))
        
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            # Load entity data from file
            with open(data_file, 'r') as f:
                if data_file.endswith('.json'):
                    entity_data = json.load(f)
                else:
                    click.echo(fmt.error("Only JSON files are supported"))
                    return
            
            # Create entity
            result = await cli_instance.db3d.create_entity(
                entity_type=entity_type,
                entity_data=entity_data,
                sync_immediately=sync
            )
            
            if result['success']:
                click.echo(fmt.success(f"✅ Entity created successfully!"))
                click.echo(f"Entity ID: {result['entity_id']}")
                
                if ctx.obj['verbose']:
                    click.echo("Database results:")
                    for db, success in result.get('results', {}).items():
                        indicator = "✅" if success else "❌"
                        click.echo(f"  {indicator} {db}")
            else:
                click.echo(fmt.error(f"❌ Failed to create entity"))
                if 'error' in result:
                    click.echo(f"Error: {result['error']}")
        
        except Exception as e:
            click.echo(fmt.error(f"Entity creation failed: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_create())


@entity.command()
@click.argument('entity_id')
@click.option('--format', type=click.Choice(['json', 'table']), default='json', help='Output format')
@click.pass_context
def get(ctx, entity_id, format):
    """Retrieve an entity by ID."""
    
    async def _get():
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            # Get entity from PostgreSQL
            result = await cli_instance.db3d.postgresql_db.read("entities", {"entity_id": entity_id})
            
            if result.success and result.data:
                entity = result.data[0]
                
                if format == 'json':
                    click.echo(json.dumps(entity, indent=2, default=str))
                else:
                    click.echo(fmt.header(f"Entity: {entity_id}"))
                    for key, value in entity.items():
                        click.echo(f"{key}: {value}")
            else:
                click.echo(fmt.error(f"Entity {entity_id} not found"))
        
        except Exception as e:
            click.echo(fmt.error(f"Failed to retrieve entity: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_get())


# =====================================================================================
# SEARCH COMMANDS
# =====================================================================================

@cli.command()
@click.argument('query_text')
@click.option('--entity-type', help='Filter by entity type')
@click.option('--limit', default=10, help='Maximum number of results')
@click.option('--threshold', default=0.7, help='Similarity threshold')
@click.option('--format', type=click.Choice(['json', 'table']), default='table', help='Output format')
@click.pass_context
def search(ctx, query_text, entity_type, limit, threshold, format):
    """Perform semantic similarity search."""
    
    async def _search():
        click.echo(fmt.header(f"🔍 Searching for: '{query_text}'"))
        
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            result = await cli_instance.db3d.search_similar(
                query_text=query_text,
                entity_type=entity_type,
                limit=limit
            )
            
            if result['success']:
                if format == 'json':
                    click.echo(json.dumps(result, indent=2, default=str))
                else:
                    # Table format
                    if isinstance(result['results'], dict) and 'results' in result['results']:
                        vector_results = result['results']['results'].get('vector', [])
                        
                        if vector_results:
                            click.echo(f"\n📋 Found {len(vector_results)} matches:")
                            click.echo("-" * 80)
                            
                            for i, match in enumerate(vector_results, 1):
                                similarity = match.get('similarity', 0)
                                entity_id = match.get('entity_id', 'unknown')
                                click.echo(f"{i:2d}. {entity_id:<30} Similarity: {similarity:.3f}")
                        else:
                            click.echo("📭 No matches found")
                    
                    click.echo(f"\n⏱️ Execution time: {result['execution_time']:.3f}s")
            else:
                click.echo(fmt.error(f"Search failed: {result.get('error', 'Unknown error')}"))
        
        except Exception as e:
            click.echo(fmt.error(f"Search failed: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_search())


# =====================================================================================
# SYNCHRONIZATION COMMANDS
# =====================================================================================

@cli.group()
def sync():
    """Synchronization management commands."""
    pass


@sync.command()
@click.option('--entity-type', help='Filter by entity type')
@click.option('--format', type=click.Choice(['json', 'table']), default='table', help='Output format')
@click.pass_context
def status(ctx, entity_type, format):
    """Check synchronization status."""
    
    async def _sync_status():
        click.echo(fmt.header("🔄 Synchronization Status"))
        
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            stats = await cli_instance.db3d.metadata_manager.get_sync_statistics()
            
            if format == 'json':
                click.echo(json.dumps(stats, indent=2, default=str))
            else:
                # Table format
                click.echo("📊 Overall Statistics:")
                click.echo(f"  Total entities: {stats.get('total_entities', 0)}")
                click.echo(f"  In PostgreSQL: {stats.get('in_postgresql', 0)}")
                click.echo(f"  In Vector DB: {stats.get('in_vector', 0)}")
                click.echo(f"  In Graph DB: {stats.get('in_graph', 0)}")
                click.echo(f"  Fully synced: {stats.get('fully_synced', 0)}")
                click.echo(f"  With errors: {stats.get('entities_with_errors', 0)}")
        
        except Exception as e:
            click.echo(fmt.error(f"Failed to get sync status: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_sync_status())


@sync.command()
@click.argument('entity_id')
@click.option('--force', is_flag=True, help='Force synchronization even if up-to-date')
@click.pass_context
def trigger(ctx, entity_id, force):
    """Manually trigger synchronization for an entity."""
    
    async def _trigger_sync():
        click.echo(fmt.header(f"🔄 Triggering sync for: {entity_id}"))
        
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            # Get entity metadata
            metadata = await cli_instance.db3d.metadata_manager.get_metadata(entity_id)
            
            if not metadata:
                click.echo(fmt.error(f"Entity {entity_id} not found"))
                return
            
            # Trigger sync
            from src.core.base import DatabaseType
            
            success = await cli_instance.db3d.sync_handler.sync_entity(
                metadata,
                source_db=DatabaseType.POSTGRESQL,
                target_dbs=[DatabaseType.VECTOR, DatabaseType.GRAPH]
            )
            
            if success:
                click.echo(fmt.success("✅ Synchronization completed successfully"))
            else:
                click.echo(fmt.error("❌ Synchronization failed"))
        
        except Exception as e:
            click.echo(fmt.error(f"Sync trigger failed: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_trigger_sync())


# =====================================================================================
# UTILITY COMMANDS
# =====================================================================================

@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json', help='Export format')
@click.pass_context
def export(ctx, output, format):
    """Export system data and configuration."""
    
    async def _export():
        click.echo(fmt.header("📤 Exporting 3db data"))
        
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            # Get all metrics and status
            metrics = await cli_instance.db3d.get_system_metrics()
            health = await cli_instance.db3d.health_check()
            
            export_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'health_status': health,
                'system_metrics': metrics,
                'version': '1.0.0'
            }
            
            if output:
                output_path = Path(output)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"3db_export_{timestamp}.{format}")
            
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                # CSV format (simplified)
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Component', 'Status', 'Metric', 'Value'])
                    
                    # Health data
                    for component, status in health['components'].items():
                        writer.writerow([component, 'health', 'healthy', status.get('healthy', False)])
                    
                    # Metrics data
                    for db_name, db_data in metrics.get('databases', {}).items():
                        if isinstance(db_data, dict):
                            for metric, value in db_data.items():
                                if isinstance(value, (int, float, str)):
                                    writer.writerow([db_name, 'metrics', metric, value])
            
            click.echo(fmt.success(f"✅ Data exported to: {output_path}"))
        
        except Exception as e:
            click.echo(fmt.error(f"Export failed: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_export())


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--entity-type', required=True, help='Entity type for bulk import')
@click.option('--batch-size', default=100, help='Batch size for processing')
@click.pass_context
def import_data(ctx, file_path, entity_type, batch_size):
    """Bulk import entities from a JSON file."""
    
    async def _import():
        click.echo(fmt.header(f"📥 Importing {entity_type} entities from {file_path}"))
        
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        entities = data
                    else:
                        entities = [data]
                else:
                    click.echo(fmt.error("Only JSON files are supported"))
                    return
            
            successful = 0
            failed = 0
            
            # Process in batches
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                click.echo(f"Processing batch {i//batch_size + 1}...")
                
                for entity_data in batch:
                    try:
                        result = await cli_instance.db3d.create_entity(
                            entity_type=entity_type,
                            entity_data=entity_data,
                            sync_immediately=True
                        )
                        
                        if result['success']:
                            successful += 1
                        else:
                            failed += 1
                            if ctx.obj['verbose']:
                                click.echo(f"  Failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        failed += 1
                        if ctx.obj['verbose']:
                            click.echo(f"  Error: {e}")
            
            click.echo(fmt.success(f"✅ Import completed!"))
            click.echo(f"  Successful: {successful}")
            click.echo(f"  Failed: {failed}")
        
        except Exception as e:
            click.echo(fmt.error(f"Import failed: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_import())


# =====================================================================================
# DEVELOPMENT COMMANDS
# =====================================================================================

@cli.group()
def dev():
    """Development and debugging commands."""
    pass


@dev.command()
@click.option('--iterations', default=10, help='Number of test iterations')
@click.pass_context
def benchmark(ctx, iterations):
    """Run performance benchmarks."""
    
    async def _benchmark():
        click.echo(fmt.header(f"🏃 Running performance benchmarks ({iterations} iterations)"))
        
        if not await cli_instance.initialize(ctx.obj['config']):
            return
        
        try:
            # Test entity creation
            create_times = []
            search_times = []
            
            for i in range(iterations):
                # Entity creation benchmark
                start_time = time.time()
                
                result = await cli_instance.db3d.create_entity(
                    entity_type="user",
                    entity_data={
                        "name": f"Benchmark User {i}",
                        "email": f"benchmark{i}@example.com",
                        "bio": "This is a benchmark user for performance testing"
                    }
                )
                
                create_time = (time.time() - start_time) * 1000
                create_times.append(create_time)
                
                # Search benchmark
                start_time = time.time()
                
                search_result = await cli_instance.db3d.search_similar(
                    query_text="benchmark user",
                    limit=5
                )
                
                search_time = (time.time() - start_time) * 1000
                search_times.append(search_time)
                
                if i % 10 == 0:
                    click.echo(f"  Completed {i + 1}/{iterations} iterations...")
            
            # Calculate statistics
            avg_create = sum(create_times) / len(create_times)
            avg_search = sum(search_times) / len(search_times)
            
            click.echo("\n📊 Benchmark Results:")
            click.echo(f"Entity Creation:")
            click.echo(f"  Average time: {avg_create:.2f}ms")
            click.echo(f"  Min time: {min(create_times):.2f}ms")
            click.echo(f"  Max time: {max(create_times):.2f}ms")
            
            click.echo(f"Similarity Search:")
            click.echo(f"  Average time: {avg_search:.2f}ms")
            click.echo(f"  Min time: {min(search_times):.2f}ms")
            click.echo(f"  Max time: {max(search_times):.2f}ms")
        
        except Exception as e:
            click.echo(fmt.error(f"Benchmark failed: {e}"))
        
        await cli_instance.cleanup()
    
    asyncio.run(_benchmark())


@dev.command()
@click.pass_context
def shell(ctx):
    """Start an interactive Python shell with 3db loaded."""
    
    async def _prepare_shell():
        if not await cli_instance.initialize(ctx.obj['config']):
            return None
        return cli_instance.db3d
    
    # Initialize 3db
    db3d = asyncio.run(_prepare_shell())
    
    if db3d:
        click.echo(fmt.header("🐍 3db Interactive Shell"))
        click.echo("Available objects:")
        click.echo("  db3d - Main 3db instance")
        click.echo("  postgresql_db - PostgreSQL database")
        click.echo("  vector_db - Vector database")
        click.echo("  graph_db - Graph database")
        click.echo()
        
        # Start IPython shell if available, otherwise use standard Python shell
        try:
            from IPython import embed
            embed(user_ns={
                'db3d': db3d,
                'postgresql_db': db3d.postgresql_db,
                'vector_db': db3d.vector_db,
                'graph_db': db3d.graph_db
            })
        except ImportError:
            import code
            code.interact(local={
                'db3d': db3d,
                'postgresql_db': db3d.postgresql_db,
                'vector_db': db3d.vector_db,
                'graph_db': db3d.graph_db
            })
        
        # Cleanup
        asyncio.run(cli_instance.cleanup())


if __name__ == '__main__':
    cli()
