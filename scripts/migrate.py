#!/usr/bin/env python3
"""
3db Unified Database Ecosystem - Database Migration System

Manages database schema changes, data migrations, and version control
across PostgreSQL, pgvector, and Apache AGE databases.
"""

import asyncio
import sys
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncpg
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.config import get_config, DatabaseConfig
from utils.logging import get_logger

logger = get_logger("migration")


@dataclass
class Migration:
    """Database migration definition."""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    checksum: str
    timestamp: datetime
    database_type: str  # postgresql, vector, graph, all


class MigrationManager:
    """Manages database migrations across all 3db components."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config()
        self.migrations_dir = Path(__file__).parent.parent / 'migrations'
        self.migrations_dir.mkdir(exist_ok=True)
        
        # Database connections
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.vector_pool: Optional[asyncpg.Pool] = None
        self.graph_pool: Optional[asyncpg.Pool] = None
        
        # Migration tracking
        self.applied_migrations: Dict[str, List[str]] = {}
    
    async def initialize(self) -> bool:
        """Initialize database connections and migration tracking."""
        try:
            logger.info("🔧 Initializing Migration Manager")
            
            # Connect to databases
            await self._connect_databases()
            
            # Setup migration tracking tables
            await self._setup_migration_tables()
            
            # Load applied migrations
            await self._load_applied_migrations()
            
            logger.info("✅ Migration Manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Migration Manager: {e}")
            return False
    
    async def _connect_databases(self):
        """Connect to all databases."""
        # PostgreSQL connection
        self.pg_pool = await asyncpg.create_pool(
            host=self.config.postgresql.host,
            port=self.config.postgresql.port,
            database=self.config.postgresql.database,
            user=self.config.postgresql.username,
            password=self.config.postgresql.password,
            min_size=1,
            max_size=5
        )
        
        # Vector database connection (might be same as PostgreSQL)
        if (self.config.vector.host != self.config.postgresql.host or 
            self.config.vector.database != self.config.postgresql.database):
            self.vector_pool = await asyncpg.create_pool(
                host=self.config.vector.host,
                port=self.config.vector.port,
                database=self.config.vector.database,
                user=self.config.vector.username,
                password=self.config.vector.password,
                min_size=1,
                max_size=5
            )
        else:
            self.vector_pool = self.pg_pool
        
        # Graph database connection (might be same as PostgreSQL)
        if (self.config.graph.host != self.config.postgresql.host or 
            self.config.graph.database != self.config.postgresql.database):
            self.graph_pool = await asyncpg.create_pool(
                host=self.config.graph.host,
                port=self.config.graph.port,
                database=self.config.graph.database,
                user=self.config.graph.username,
                password=self.config.graph.password,
                min_size=1,
                max_size=5
            )
        else:
            self.graph_pool = self.pg_pool
    
    async def _setup_migration_tables(self):
        """Create migration tracking tables."""
        migration_table_sql = """
            CREATE TABLE IF NOT EXISTS _3db_migrations (
                id SERIAL PRIMARY KEY,
                version VARCHAR(50) NOT NULL,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                checksum VARCHAR(64) NOT NULL,
                database_type VARCHAR(20) NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                applied_by VARCHAR(100) DEFAULT CURRENT_USER,
                execution_time_ms INTEGER,
                UNIQUE(version, database_type)
            )
        """
        
        # Create table in all databases
        for pool_name, pool in [
            ('postgresql', self.pg_pool),
            ('vector', self.vector_pool),
            ('graph', self.graph_pool)
        ]:
            if pool:
                async with pool.acquire() as conn:
                    await conn.execute(migration_table_sql)
                logger.debug(f"Migration table ready in {pool_name}")
    
    async def _load_applied_migrations(self):
        """Load list of applied migrations from each database."""
        query = "SELECT version FROM _3db_migrations ORDER BY applied_at"
        
        for db_type, pool in [
            ('postgresql', self.pg_pool),
            ('vector', self.vector_pool),
            ('graph', self.graph_pool)
        ]:
            if pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch(query)
                    self.applied_migrations[db_type] = [row['version'] for row in rows]
                    logger.debug(f"Loaded {len(rows)} applied migrations for {db_type}")
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA-256 checksum of migration content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _parse_migration_file(self, file_path: Path) -> Migration:
        """Parse a migration file and extract metadata."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract metadata from comments
        lines = content.split('\n')
        metadata = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('-- @'):
                key, value = line[4:].split(':', 1)
                metadata[key.strip()] = value.strip()
        
        # Extract version from filename (e.g., 001_initial_schema.sql)
        version = file_path.name.split('_')[0]
        
        # Split UP and DOWN sections
        up_section = []
        down_section = []
        current_section = 'up'
        
        for line in lines:
            if line.strip().upper() == '-- DOWN':
                current_section = 'down'
                continue
            elif line.strip().upper() == '-- UP':
                current_section = 'up'
                continue
            
            if current_section == 'up':
                up_section.append(line)
            else:
                down_section.append(line)
        
        up_sql = '\n'.join(up_section).strip()
        down_sql = '\n'.join(down_section).strip()
        
        return Migration(
            version=version,
            name=metadata.get('name', file_path.stem),
            description=metadata.get('description', ''),
            up_sql=up_sql,
            down_sql=down_sql,
            checksum=self._calculate_checksum(up_sql),
            timestamp=datetime.utcnow(),
            database_type=metadata.get('database', 'all')
        )
    
    def discover_migrations(self) -> List[Migration]:
        """Discover all migration files in the migrations directory."""
        migrations = []
        
        for file_path in sorted(self.migrations_dir.glob('*.sql')):
            try:
                migration = self._parse_migration_file(file_path)
                migrations.append(migration)
                logger.debug(f"Discovered migration: {migration.version} - {migration.name}")
            except Exception as e:
                logger.error(f"Failed to parse migration {file_path}: {e}")
        
        return migrations
    
    async def _apply_migration_to_database(self, migration: Migration, db_type: str, pool: asyncpg.Pool) -> bool:
        """Apply a migration to a specific database."""
        try:
            start_time = datetime.utcnow()
            
            async with pool.acquire() as conn:
                # Execute migration SQL
                await conn.execute(migration.up_sql)
                
                # Record migration in tracking table
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                await conn.execute("""
                    INSERT INTO _3db_migrations 
                    (version, name, description, checksum, database_type, execution_time_ms)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, migration.version, migration.name, migration.description, 
                     migration.checksum, db_type, int(execution_time))
            
            logger.info(f"✅ Applied migration {migration.version} to {db_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply migration {migration.version} to {db_type}: {e}")
            return False
    
    async def _rollback_migration_from_database(self, migration: Migration, db_type: str, pool: asyncpg.Pool) -> bool:
        """Rollback a migration from a specific database."""
        try:
            async with pool.acquire() as conn:
                # Execute rollback SQL
                if migration.down_sql:
                    await conn.execute(migration.down_sql)
                
                # Remove migration record
                await conn.execute("""
                    DELETE FROM _3db_migrations 
                    WHERE version = $1 AND database_type = $2
                """, migration.version, db_type)
            
            logger.info(f"✅ Rolled back migration {migration.version} from {db_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to rollback migration {migration.version} from {db_type}: {e}")
            return False
    
    async def apply_migrations(self, target_version: Optional[str] = None) -> bool:
        """Apply pending migrations up to target version."""
        logger.info("🚀 Starting migration process")
        
        # Discover all migrations
        migrations = self.discover_migrations()
        
        if not migrations:
            logger.info("📭 No migrations found")
            return True
        
        # Filter migrations to apply
        pending_migrations = []
        
        for migration in migrations:
            if target_version and migration.version > target_version:
                break
            
            # Check which databases need this migration
            databases_to_apply = []
            
            if migration.database_type == 'all':
                databases_to_apply = ['postgresql', 'vector', 'graph']
            else:
                databases_to_apply = [migration.database_type]
            
            for db_type in databases_to_apply:
                if migration.version not in self.applied_migrations.get(db_type, []):
                    pending_migrations.append((migration, db_type))
        
        if not pending_migrations:
            logger.info("✅ All migrations are already applied")
            return True
        
        logger.info(f"📋 Found {len(pending_migrations)} migrations to apply")
        
        # Apply migrations
        success_count = 0
        
        for migration, db_type in pending_migrations:
            pool = getattr(self, f'{db_type}_pool')
            if pool:
                success = await self._apply_migration_to_database(migration, db_type, pool)
                if success:
                    success_count += 1
                else:
                    logger.error(f"💥 Migration failed, stopping process")
                    return False
        
        logger.info(f"🎉 Successfully applied {success_count} migrations")
        
        # Reload applied migrations
        await self._load_applied_migrations()
        
        return True
    
    async def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration."""
        logger.info(f"⏪ Rolling back migration {version}")
        
        # Find the migration
        migrations = self.discover_migrations()
        migration = next((m for m in migrations if m.version == version), None)
        
        if not migration:
            logger.error(f"Migration {version} not found")
            return False
        
        # Determine which databases have this migration applied
        databases_to_rollback = []
        
        if migration.database_type == 'all':
            check_databases = ['postgresql', 'vector', 'graph']
        else:
            check_databases = [migration.database_type]
        
        for db_type in check_databases:
            if version in self.applied_migrations.get(db_type, []):
                databases_to_rollback.append(db_type)
        
        if not databases_to_rollback:
            logger.info(f"Migration {version} is not applied to any database")
            return True
        
        # Rollback from databases
        success_count = 0
        
        for db_type in databases_to_rollback:
            pool = getattr(self, f'{db_type}_pool')
            if pool:
                success = await self._rollback_migration_from_database(migration, db_type, pool)
                if success:
                    success_count += 1
        
        logger.info(f"✅ Rolled back migration from {success_count} databases")
        
        # Reload applied migrations
        await self._load_applied_migrations()
        
        return success_count == len(databases_to_rollback)
    
    def create_migration(self, name: str, database_type: str = 'all') -> str:
        """Create a new migration file."""
        # Get next version number
        existing_migrations = self.discover_migrations()
        if existing_migrations:
            last_version = max(int(m.version) for m in existing_migrations)
            next_version = f"{last_version + 1:03d}"
        else:
            next_version = "001"
        
        # Create filename
        filename = f"{next_version}_{name.lower().replace(' ', '_')}.sql"
        file_path = self.migrations_dir / filename
        
        # Create migration template
        template = f"""-- @name: {name}
-- @description: {name} migration
-- @database: {database_type}
-- @created: {datetime.utcnow().isoformat()}

-- UP
-- Add your schema changes here



-- DOWN
-- Add rollback statements here


"""
        
        with open(file_path, 'w') as f:
            f.write(template)
        
        logger.info(f"📝 Created migration file: {filename}")
        return str(file_path)
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get status of all migrations."""
        migrations = self.discover_migrations()
        
        status = {
            'total_migrations': len(migrations),
            'databases': {},
            'migrations': []
        }
        
        # Database status
        for db_type in ['postgresql', 'vector', 'graph']:
            applied = len(self.applied_migrations.get(db_type, []))
            status['databases'][db_type] = {
                'applied_migrations': applied,
                'pending_migrations': len(migrations) - applied
            }
        
        # Individual migration status
        for migration in migrations:
            migration_status = {
                'version': migration.version,
                'name': migration.name,
                'description': migration.description,
                'database_type': migration.database_type,
                'applied_to': []
            }
            
            # Check which databases have this migration
            check_databases = ['postgresql', 'vector', 'graph'] if migration.database_type == 'all' else [migration.database_type]
            
            for db_type in check_databases:
                if migration.version in self.applied_migrations.get(db_type, []):
                    migration_status['applied_to'].append(db_type)
            
            status['migrations'].append(migration_status)
        
        return status
    
    async def cleanup(self):
        """Close database connections."""
        for pool in [self.pg_pool, self.vector_pool, self.graph_pool]:
            if pool and pool != self.pg_pool:  # Avoid closing the same pool twice
                await pool.close()
        
        if self.pg_pool:
            await self.pg_pool.close()


async def main():
    """Main migration CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="3db Database Migration Tool")
    parser.add_argument('--config', '-c', help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Migration commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show migration status')
    
    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply pending migrations')
    apply_parser.add_argument('--target', help='Target version to migrate to')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback a migration')
    rollback_parser.add_argument('version', help='Migration version to rollback')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new migration')
    create_parser.add_argument('name', help='Migration name')
    create_parser.add_argument('--database', choices=['postgresql', 'vector', 'graph', 'all'], 
                              default='all', help='Target database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create migration manager
    manager = MigrationManager(args.config)
    
    try:
        # Initialize
        if not await manager.initialize():
            return 1
        
        # Execute command
        if args.command == 'status':
            status = await manager.get_migration_status()
            print("📊 Migration Status:")
            print("=" * 50)
            print(f"Total migrations: {status['total_migrations']}")
            print()
            
            print("Database Status:")
            for db_type, db_status in status['databases'].items():
                print(f"  {db_type}: {db_status['applied_migrations']} applied, {db_status['pending_migrations']} pending")
            print()
            
            print("Migrations:")
            for migration in status['migrations']:
                applied_to = ', '.join(migration['applied_to']) or 'none'
                print(f"  {migration['version']}: {migration['name']} -> {applied_to}")
        
        elif args.command == 'apply':
            success = await manager.apply_migrations(args.target)
            return 0 if success else 1
        
        elif args.command == 'rollback':
            success = await manager.rollback_migration(args.version)
            return 0 if success else 1
        
        elif args.command == 'create':
            file_path = manager.create_migration(args.name, args.database)
            print(f"Created migration: {file_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"💥 Migration command failed: {e}")
        return 1
    finally:
        await manager.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
