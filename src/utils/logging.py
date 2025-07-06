"""
3db Unified Database Ecosystem - Logging and Utilities

This module provides centralized logging, monitoring, and utility functions
for the unified database ecosystem.
"""

import logging
import structlog
import sys
import time
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from functools import wraps
from contextlib import asynccontextmanager
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

# Configure structlog for consistent JSON logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class PerformanceMetrics:
    """Performance metrics for database operations."""
    
    operation_id: str
    operation_type: str
    database_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    rows_affected: int = 0
    memory_usage_mb: Optional[float] = None
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark the operation as finished."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class UnifiedLogger:
    """Centralized logger for the 3db ecosystem."""
    
    def __init__(self, name: str = "3db", level: LogLevel = LogLevel.INFO):
        self.logger = structlog.get_logger(name)
        self.name = name
        self.level = level
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, self.level.value)
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        self.info(
            "Performance metrics",
            operation_id=metrics.operation_id,
            operation_type=metrics.operation_type,
            database_type=metrics.database_type,
            duration_ms=metrics.duration_ms,
            success=metrics.success,
            rows_affected=metrics.rows_affected,
            error_message=metrics.error_message
        )
    
    def log_sync_event(self, event_type: str, entity_id: str, source_db: str, target_db: str, success: bool, **kwargs):
        """Log synchronization events."""
        self.info(
            "Synchronization event",
            event_type=event_type,
            entity_id=entity_id,
            source_database=source_db,
            target_database=target_db,
            success=success,
            **kwargs
        )
    
    def log_query_execution(self, query: str, params: Optional[Dict[str, Any]], result: Any, duration_ms: float):
        """Log query execution details."""
        self.info(
            "Query executed",
            query=query[:200] + "..." if len(query) > 200 else query,  # Truncate long queries
            params=params,
            duration_ms=duration_ms,
            has_result=result is not None
        )


# Global logger instance
logger = UnifiedLogger()


def get_logger(name: str = "3db") -> UnifiedLogger:
    """Get a logger instance."""
    return UnifiedLogger(name)


# Performance monitoring decorators
def monitor_performance(operation_type: str, database_type: str = "unknown"):
    """Decorator to monitor performance of async functions."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            operation_id = str(uuid.uuid4())
            metrics = PerformanceMetrics(
                operation_id=operation_id,
                operation_type=operation_type,
                database_type=database_type,
                start_time=datetime.utcnow()
            )
            
            try:
                result = await func(*args, **kwargs)
                metrics.finish(success=True)
                logger.log_performance(metrics)
                return result
                
            except Exception as e:
                metrics.finish(success=False, error_message=str(e))
                logger.log_performance(metrics)
                raise
                
        return wrapper
    return decorator


def monitor_sync_performance(func: Callable):
    """Decorator specifically for synchronization operations."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            logger.info(
                "Synchronization completed",
                function=func.__name__,
                duration_ms=duration,
                success=True
            )
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(
                "Synchronization failed",
                function=func.__name__,
                duration_ms=duration,
                error=str(e),
                success=False
            )
            raise
    return wrapper


# Utility functions
def generate_entity_id(entity_type: str, prefix: Optional[str] = None) -> str:
    """Generate a unique entity ID."""
    timestamp = int(time.time() * 1000)  # Millisecond precision
    unique_id = str(uuid.uuid4())[:8]
    
    if prefix:
        return f"{prefix}_{entity_type}_{timestamp}_{unique_id}"
    else:
        return f"{entity_type}_{timestamp}_{unique_id}"


def validate_entity_data(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate that entity data contains required fields."""
    for field in required_fields:
        if field not in data or data[field] is None:
            logger.error(f"Missing required field: {field}", data=data)
            return False
    return True


def sanitize_sql_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize SQL parameters to prevent injection attacks."""
    sanitized = {}
    for key, value in params.items():
        if isinstance(value, str):
            # Basic sanitization - remove potentially dangerous characters
            sanitized[key] = value.replace(';', '').replace('--', '').replace('/*', '').replace('*/', '')
        else:
            sanitized[key] = value
    return sanitized


def format_duration(duration_ms: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if duration_ms < 1000:
        return f"{duration_ms:.2f}ms"
    elif duration_ms < 60000:
        return f"{duration_ms/1000:.2f}s"
    else:
        minutes = int(duration_ms / 60000)
        seconds = (duration_ms % 60000) / 1000
        return f"{minutes}m {seconds:.2f}s"


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


async def retry_async(
    func: Callable,
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry an async function with exponential backoff.
    """
    last_exception = None
    delay = delay_seconds
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                    error=str(e),
                    function=func.__name__ if hasattr(func, '__name__') else 'unknown'
                )
                await asyncio.sleep(delay)
                delay *= backoff_multiplier
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed",
                    error=str(e),
                    function=func.__name__ if hasattr(func, '__name__') else 'unknown'
                )
    
    raise last_exception


@asynccontextmanager
async def timed_operation(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    logger.info(f"Starting operation: {operation_name}")
    
    try:
        yield
        duration = (time.time() - start_time) * 1000
        logger.info(
            f"Operation completed: {operation_name}",
            duration_ms=duration,
            duration_formatted=format_duration(duration)
        )
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(
            f"Operation failed: {operation_name}",
            duration_ms=duration,
            duration_formatted=format_duration(duration),
            error=str(e)
        )
        raise


class CircuitBreaker:
    """Circuit breaker pattern for handling database failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        
        elapsed = datetime.utcnow() - self.last_failure_time
        return elapsed.total_seconds() >= self.timeout_seconds
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )


# Configuration validation utilities
def validate_database_config(config: Dict[str, Any]) -> List[str]:
    """Validate database configuration and return list of errors."""
    errors = []
    
    required_fields = ['host', 'port', 'database', 'username']
    for field in required_fields:
        if field not in config or not config[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate port is a number
    if 'port' in config:
        try:
            port = int(config['port'])
            if not (1 <= port <= 65535):
                errors.append("Port must be between 1 and 65535")
        except (ValueError, TypeError):
            errors.append("Port must be a valid integer")
    
    return errors


def create_connection_pool_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create connection pool configuration from base config."""
    return {
        'host': base_config['host'],
        'port': base_config['port'],
        'database': base_config['database'],
        'user': base_config['username'],
        'password': base_config.get('password', ''),
        'min_size': base_config.get('pool_min_size', 5),
        'max_size': base_config.get('pool_max_size', 20),
        'command_timeout': base_config.get('pool_timeout', 30)
    }


# Example usage and testing
if __name__ == "__main__":
    # Example usage of logging utilities
    logger.info("3db Logging and Utilities System Initialized")
    
    # Test performance monitoring
    @monitor_performance("test_operation", "test_db")
    async def test_function():
        await asyncio.sleep(0.1)  # Simulate work
        return "success"
    
    # Test the function
    async def main():
        result = await test_function()
        logger.info("Test completed", result=result)
    
    # Run test
    asyncio.run(main())
