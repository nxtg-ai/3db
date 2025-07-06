#!/usr/bin/env python3
"""
3db Unified Database Ecosystem - Performance Monitor

Real-time performance monitoring and alerting system for 3db components.
Tracks database performance, synchronization health, and system resources.
"""

import asyncio
import time
import psutil
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unified import Database3D
from core.config import get_config
from utils.logging import get_logger

logger = get_logger("performance_monitor")


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    timestamp: datetime
    severity: str  # info, warning, critical
    component: str
    metric: str
    current_value: float
    threshold: float
    message: str


@dataclass
class SystemSnapshot:
    """System performance snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    database_metrics: Dict[str, Any]
    sync_metrics: Dict[str, Any]
    alerts: List[PerformanceAlert]


class PerformanceMonitor:
    """Real-time performance monitoring for 3db system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.db3d: Optional[Database3D] = None
        self.monitoring = False
        self.snapshots: List[SystemSnapshot] = []
        self.alerts: List[PerformanceAlert] = []
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'query_time_ms': 1000.0,
            'queue_length': 1000,
            'sync_error_rate': 0.1,
            'failed_queries_rate': 0.05
        }
        
        # Metrics history for trend analysis
        self.metrics_history = []
        self.max_history_size = 1000
    
    async def initialize(self) -> bool:
        """Initialize the monitoring system."""
        try:
            logger.info("🔍 Initializing Performance Monitor")
            
            self.db3d = Database3D()
            success = await self.db3d.initialize()
            
            if success:
                logger.info("✅ Performance Monitor initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize 3db system")
                return False
                
        except Exception as e:
            logger.error(f"💥 Performance Monitor initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the monitoring system."""
        self.monitoring = False
        if self.db3d:
            await self.db3d.shutdown()
        logger.info("👋 Performance Monitor shutdown complete")
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_usage_percent': disk_percent,
                'network_io': network_io,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return {}
    
    async def check_database_performance(self) -> Dict[str, Any]:
        """Check database performance metrics."""
        try:
            if not self.db3d:
                return {}
            
            # Get metrics from all databases
            metrics = await self.db3d.get_system_metrics()
            
            # Extract relevant performance data
            db_performance = {}
            
            for db_name, db_data in metrics.get('databases', {}).items():
                if isinstance(db_data, dict):
                    db_performance[db_name] = {
                        'queries_executed': db_data.get('queries_executed', 0),
                        'queries_failed': db_data.get('queries_failed', 0),
                        'avg_execution_time': db_data.get('avg_execution_time', 0),
                        'total_execution_time': db_data.get('total_execution_time', 0),
                        'is_connected': db_data.get('is_connected', False)
                    }
                    
                    # Calculate derived metrics
                    total_queries = db_performance[db_name]['queries_executed'] + db_performance[db_name]['queries_failed']
                    if total_queries > 0:
                        db_performance[db_name]['error_rate'] = db_performance[db_name]['queries_failed'] / total_queries
                    else:
                        db_performance[db_name]['error_rate'] = 0.0
            
            return db_performance
            
        except Exception as e:
            logger.error(f"Failed to get database performance: {e}")
            return {}
    
    async def check_sync_health(self) -> Dict[str, Any]:
        """Check synchronization system health."""
        try:
            if not self.db3d or not self.db3d.metadata_manager:
                return {}
            
            # Get synchronization statistics
            sync_stats = await self.db3d.metadata_manager.get_sync_statistics()
            
            # Calculate sync health metrics
            total_entities = sync_stats.get('total_entities', 0)
            fully_synced = sync_stats.get('fully_synced', 0)
            with_errors = sync_stats.get('entities_with_errors', 0)
            
            sync_health = {
                'total_entities': total_entities,
                'fully_synced': fully_synced,
                'with_errors': with_errors,
                'sync_rate': fully_synced / total_entities if total_entities > 0 else 1.0,
                'error_rate': with_errors / total_entities if total_entities > 0 else 0.0
            }
            
            # Get queue statistics if available
            if self.db3d.event_broker:
                try:
                    queue_stats = await self.db3d.event_broker.get_queue_stats()
                    sync_health['queue_stats'] = queue_stats
                    sync_health['total_queued'] = sum(queue_stats.values())
                except Exception:
                    sync_health['queue_stats'] = {}
                    sync_health['total_queued'] = 0
            
            return sync_health
            
        except Exception as e:
            logger.error(f"Failed to get sync health: {e}")
            return {}
    
    def check_thresholds(self, 
                        system_metrics: Dict[str, Any], 
                        db_metrics: Dict[str, Any], 
                        sync_metrics: Dict[str, Any]) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        timestamp = datetime.utcnow()
        
        # System resource alerts
        if system_metrics:
            if system_metrics.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
                alerts.append(PerformanceAlert(
                    timestamp=timestamp,
                    severity='warning',
                    component='system',
                    metric='cpu_percent',
                    current_value=system_metrics['cpu_percent'],
                    threshold=self.thresholds['cpu_percent'],
                    message=f"High CPU usage: {system_metrics['cpu_percent']:.1f}%"
                ))
            
            if system_metrics.get('memory_percent', 0) > self.thresholds['memory_percent']:
                alerts.append(PerformanceAlert(
                    timestamp=timestamp,
                    severity='warning',
                    component='system',
                    metric='memory_percent',
                    current_value=system_metrics['memory_percent'],
                    threshold=self.thresholds['memory_percent'],
                    message=f"High memory usage: {system_metrics['memory_percent']:.1f}%"
                ))
            
            if system_metrics.get('disk_usage_percent', 0) > self.thresholds['disk_usage_percent']:
                alerts.append(PerformanceAlert(
                    timestamp=timestamp,
                    severity='critical',
                    component='system',
                    metric='disk_usage_percent',
                    current_value=system_metrics['disk_usage_percent'],
                    threshold=self.thresholds['disk_usage_percent'],
                    message=f"High disk usage: {system_metrics['disk_usage_percent']:.1f}%"
                ))
        
        # Database performance alerts
        for db_name, db_data in db_metrics.items():
            if not db_data.get('is_connected', False):
                alerts.append(PerformanceAlert(
                    timestamp=timestamp,
                    severity='critical',
                    component=db_name,
                    metric='connection',
                    current_value=0,
                    threshold=1,
                    message=f"Database {db_name} is not connected"
                ))
            
            avg_time_ms = db_data.get('avg_execution_time', 0) * 1000
            if avg_time_ms > self.thresholds['query_time_ms']:
                alerts.append(PerformanceAlert(
                    timestamp=timestamp,
                    severity='warning',
                    component=db_name,
                    metric='avg_execution_time',
                    current_value=avg_time_ms,
                    threshold=self.thresholds['query_time_ms'],
                    message=f"Slow queries in {db_name}: {avg_time_ms:.1f}ms average"
                ))
            
            error_rate = db_data.get('error_rate', 0)
            if error_rate > self.thresholds['failed_queries_rate']:
                alerts.append(PerformanceAlert(
                    timestamp=timestamp,
                    severity='warning',
                    component=db_name,
                    metric='error_rate',
                    current_value=error_rate,
                    threshold=self.thresholds['failed_queries_rate'],
                    message=f"High error rate in {db_name}: {error_rate:.1%}"
                ))
        
        # Synchronization alerts
        if sync_metrics:
            error_rate = sync_metrics.get('error_rate', 0)
            if error_rate > self.thresholds['sync_error_rate']:
                alerts.append(PerformanceAlert(
                    timestamp=timestamp,
                    severity='warning',
                    component='synchronization',
                    metric='error_rate',
                    current_value=error_rate,
                    threshold=self.thresholds['sync_error_rate'],
                    message=f"High sync error rate: {error_rate:.1%}"
                ))
            
            total_queued = sync_metrics.get('total_queued', 0)
            if total_queued > self.thresholds['queue_length']:
                alerts.append(PerformanceAlert(
                    timestamp=timestamp,
                    severity='warning',
                    component='synchronization',
                    metric='queue_length',
                    current_value=total_queued,
                    threshold=self.thresholds['queue_length'],
                    message=f"High queue length: {total_queued} items"
                ))
        
        return alerts
    
    async def take_snapshot(self) -> SystemSnapshot:
        """Take a complete system performance snapshot."""
        timestamp = datetime.utcnow()
        
        # Gather all metrics
        system_metrics = self.check_system_resources()
        db_metrics = await self.check_database_performance()
        sync_metrics = await self.check_sync_health()
        
        # Check for alerts
        alerts = self.check_thresholds(system_metrics, db_metrics, sync_metrics)
        
        # Create snapshot
        snapshot = SystemSnapshot(
            timestamp=timestamp,
            cpu_percent=system_metrics.get('cpu_percent', 0),
            memory_percent=system_metrics.get('memory_percent', 0),
            disk_usage_percent=system_metrics.get('disk_usage_percent', 0),
            network_io=system_metrics.get('network_io', {}),
            database_metrics=db_metrics,
            sync_metrics=sync_metrics,
            alerts=alerts
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots
        if len(self.snapshots) > self.max_history_size:
            self.snapshots = self.snapshots[-self.max_history_size:]
        
        # Store alerts
        self.alerts.extend(alerts)
        
        # Log critical alerts
        for alert in alerts:
            if alert.severity == 'critical':
                logger.error(f"🚨 CRITICAL ALERT: {alert.message}")
            elif alert.severity == 'warning':
                logger.warning(f"⚠️ WARNING: {alert.message}")
        
        return snapshot
    
    async def monitor_continuously(self, interval_seconds: int = 30):
        """Run continuous monitoring."""
        logger.info(f"🔍 Starting continuous monitoring (interval: {interval_seconds}s)")
        self.monitoring = True
        
        try:
            while self.monitoring:
                snapshot = await self.take_snapshot()
                
                # Log summary every 10 snapshots
                if len(self.snapshots) % 10 == 0:
                    logger.info(
                        f"📊 Snapshot {len(self.snapshots)}: "
                        f"CPU: {snapshot.cpu_percent:.1f}%, "
                        f"Memory: {snapshot.memory_percent:.1f}%, "
                        f"Alerts: {len(snapshot.alerts)}"
                    )
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"💥 Monitoring error: {e}")
        finally:
            self.monitoring = False
    
    def export_metrics(self, output_file: str):
        """Export collected metrics to file."""
        try:
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'total_snapshots': len(self.snapshots),
                'total_alerts': len(self.alerts),
                'thresholds': self.thresholds,
                'snapshots': [asdict(snapshot) for snapshot in self.snapshots],
                'alerts': [asdict(alert) for alert in self.alerts]
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"📤 Metrics exported to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance over the monitoring period."""
        if not self.snapshots:
            return {}
        
        # Calculate averages and trends
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_percent for s in self.snapshots]
        disk_values = [s.disk_usage_percent for s in self.snapshots]
        
        summary = {
            'monitoring_period': {
                'start': self.snapshots[0].timestamp.isoformat(),
                'end': self.snapshots[-1].timestamp.isoformat(),
                'duration_minutes': (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds() / 60
            },
            'system_resources': {
                'cpu_avg': sum(cpu_values) / len(cpu_values),
                'cpu_max': max(cpu_values),
                'memory_avg': sum(memory_values) / len(memory_values),
                'memory_max': max(memory_values),
                'disk_avg': sum(disk_values) / len(disk_values),
                'disk_max': max(disk_values)
            },
            'alerts_summary': {
                'total_alerts': len(self.alerts),
                'critical_alerts': len([a for a in self.alerts if a.severity == 'critical']),
                'warning_alerts': len([a for a in self.alerts if a.severity == 'warning']),
                'info_alerts': len([a for a in self.alerts if a.severity == 'info'])
            },
            'database_health': {},
            'sync_health': {}
        }
        
        # Database health summary
        if self.snapshots[-1].database_metrics:
            for db_name, db_data in self.snapshots[-1].database_metrics.items():
                summary['database_health'][db_name] = {
                    'connected': db_data.get('is_connected', False),
                    'total_queries': db_data.get('queries_executed', 0) + db_data.get('queries_failed', 0),
                    'error_rate': db_data.get('error_rate', 0),
                    'avg_execution_time_ms': db_data.get('avg_execution_time', 0) * 1000
                }
        
        # Sync health summary
        if self.snapshots[-1].sync_metrics:
            sync_data = self.snapshots[-1].sync_metrics
            summary['sync_health'] = {
                'total_entities': sync_data.get('total_entities', 0),
                'sync_rate': sync_data.get('sync_rate', 0),
                'error_rate': sync_data.get('error_rate', 0),
                'total_queued': sync_data.get('total_queued', 0)
            }
        
        return summary


async def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="3db Performance Monitor")
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--interval', '-i', type=int, default=30, help='Monitoring interval in seconds')
    parser.add_argument('--duration', '-d', type=int, help='Monitoring duration in minutes (0 for continuous)')
    parser.add_argument('--export', '-e', help='Export file path for metrics')
    parser.add_argument('--summary', '-s', action='store_true', help='Show performance summary at end')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = PerformanceMonitor(args.config)
    
    try:
        # Initialize
        if not await monitor.initialize():
            return 1
        
        # Run monitoring
        if args.duration:
            # Run for specified duration
            end_time = time.time() + (args.duration * 60)
            
            while time.time() < end_time and monitor.monitoring:
                await monitor.take_snapshot()
                await asyncio.sleep(args.interval)
        else:
            # Run continuously
            await monitor.monitor_continuously(args.interval)
        
        # Export metrics if requested
        if args.export:
            monitor.export_metrics(args.export)
        
        # Show summary if requested
        if args.summary:
            summary = monitor.get_performance_summary()
            print("\n📊 Performance Summary:")
            print("=" * 50)
            print(json.dumps(summary, indent=2, default=str))
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        return 0
    except Exception as e:
        logger.error(f"💥 Monitor failed: {e}")
        return 1
    finally:
        await monitor.shutdown()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
