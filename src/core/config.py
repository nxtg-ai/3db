"""
3db Unified Database Ecosystem - Configuration Management

This module provides centralized configuration management for the unified database ecosystem,
handling PostgreSQL CRUD, pgvector, and Apache AGE graph database configurations.
"""

from typing import Optional
from pydantic import BaseSettings, Field, validator
import os
from pathlib import Path


class DatabaseConfig(BaseSettings):
    """Base database configuration with common connection parameters."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    
    # Connection pool settings
    pool_min_size: int = Field(default=5, description="Minimum connection pool size")
    pool_max_size: int = Field(default=20, description="Maximum connection pool size")
    pool_timeout: int = Field(default=30, description="Connection timeout in seconds")
    
    @property
    def connection_url(self) -> str:
        """Generate connection URL for the database."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_url(self) -> str:
        """Generate async connection URL for the database."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class PostgreSQLConfig(DatabaseConfig):
    """PostgreSQL CRUD database configuration."""
    
    # PostgreSQL-specific settings
    statement_timeout: int = Field(default=30000, description="Statement timeout in milliseconds")
    idle_in_transaction_timeout: int = Field(default=60000, description="Idle transaction timeout")
    

class VectorConfig(DatabaseConfig):
    """Vector database (pgvector) configuration."""
    
    # Vector-specific settings
    embedding_dimension: int = Field(default=384, description="Vector embedding dimension")
    similarity_threshold: float = Field(default=0.8, description="Similarity search threshold")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Default embedding model")
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v


class GraphConfig(DatabaseConfig):
    """Graph database (Apache AGE) configuration."""
    
    # Graph-specific settings
    graph_name: str = Field(default="unified_graph", description="Default graph name")
    max_traversal_depth: int = Field(default=10, description="Maximum graph traversal depth")


class SynchronizationConfig(BaseSettings):
    """Synchronization settings for EDA and CDC."""
    
    # Event-Driven Architecture settings
    eda_enabled: bool = Field(default=True, description="Enable event-driven architecture")
    eda_broker_url: str = Field(default="redis://localhost:6379/0", description="Message broker URL")
    eda_result_backend: str = Field(default="redis://localhost:6379/0", description="Result backend URL")
    
    # Change Data Capture settings
    cdc_enabled: bool = Field(default=True, description="Enable change data capture")
    cdc_batch_size: int = Field(default=1000, description="CDC batch processing size")
    cdc_sync_interval: int = Field(default=60, description="CDC synchronization interval in seconds")
    cdc_log_retention_days: int = Field(default=30, description="CDC log retention period")


class CacheConfig(BaseSettings):
    """Caching configuration using Redis."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=1, description="Redis database number")
    ttl_seconds: int = Field(default=3600, description="Default TTL for cached items")
    max_connections: int = Field(default=100, description="Maximum Redis connections")
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        return f"redis://{self.host}:{self.port}/{self.db}"


class SecurityConfig(BaseSettings):
    """Security and encryption configuration."""
    
    enable_encryption: bool = Field(default=False, description="Enable data encryption")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key")
    enable_api_auth: bool = Field(default=True, description="Enable API authentication")
    api_secret_key: str = Field(..., description="API secret key")
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v, values):
        if values.get('enable_encryption') and not v:
            raise ValueError('Encryption key is required when encryption is enabled')
        return v


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format")
    prometheus_port: int = Field(default=8000, description="Prometheus metrics port")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()


class UnifiedConfig(BaseSettings):
    """Main configuration class that aggregates all component configurations."""
    
    # Database configurations
    postgresql: PostgreSQLConfig
    vector: VectorConfig
    graph: GraphConfig
    
    # System configurations
    sync: SynchronizationConfig
    cache: CacheConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "_"
    
    @classmethod
    def load_from_env(cls, env_file: Optional[str] = None) -> "UnifiedConfig":
        """Load configuration from environment file."""
        if env_file and Path(env_file).exists():
            return cls(_env_file=env_file)
        
        # Try to find .env file in common locations
        possible_paths = [
            Path(".env"),
            Path("config/.env"),
            Path("../config/.env"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return cls(_env_file=str(path))
        
        # Return with default values if no .env file found
        return cls()
    
    def validate_configuration(self) -> bool:
        """Validate the entire configuration for consistency and completeness."""
        try:
            # Validate database connectivity parameters
            for db_name, db_config in [
                ("PostgreSQL", self.postgresql),
                ("Vector", self.vector),
                ("Graph", self.graph)
            ]:
                if not all([db_config.host, db_config.database, db_config.username]):
                    raise ValueError(f"{db_name} database configuration is incomplete")
            
            # Validate synchronization settings
            if self.sync.eda_enabled and not self.sync.eda_broker_url:
                raise ValueError("EDA broker URL is required when EDA is enabled")
            
            # Validate security settings
            if self.security.enable_encryption and not self.security.encryption_key:
                raise ValueError("Encryption key is required when encryption is enabled")
            
            return True
            
        except ValueError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def get_database_config(self, db_type: str) -> DatabaseConfig:
        """Get configuration for a specific database type."""
        configs = {
            "postgresql": self.postgresql,
            "vector": self.vector,
            "graph": self.graph
        }
        
        if db_type not in configs:
            raise ValueError(f"Unknown database type: {db_type}")
        
        return configs[db_type]


# Global configuration instance
config: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = UnifiedConfig.load_from_env()
        config.validate_configuration()
    return config


def reload_config(env_file: Optional[str] = None) -> UnifiedConfig:
    """Reload configuration from environment file."""
    global config
    config = UnifiedConfig.load_from_env(env_file)
    config.validate_configuration()
    return config


# Example usage and testing
if __name__ == "__main__":
    # Load configuration
    cfg = get_config()
    
    # Print configuration summary
    print("3db Unified Database Ecosystem Configuration Loaded:")
    print(f"PostgreSQL: {cfg.postgresql.host}:{cfg.postgresql.port}/{cfg.postgresql.database}")
    print(f"Vector DB: {cfg.vector.host}:{cfg.vector.port}/{cfg.vector.database}")
    print(f"Graph DB: {cfg.graph.host}:{cfg.graph.port}/{cfg.graph.database}")
    print(f"Cache: {cfg.cache.redis_url}")
    print(f"EDA Enabled: {cfg.sync.eda_enabled}")
    print(f"CDC Enabled: {cfg.sync.cdc_enabled}")
