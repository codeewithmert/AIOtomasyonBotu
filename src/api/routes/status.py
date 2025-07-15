"""
Status Routes

Provides endpoints for system health, monitoring, and status information
in the AI Automation Bot.
"""

import psutil
import platform
from datetime import datetime
from typing import Dict, Any
from flask import Blueprint, request, jsonify, g

from ...core.logger import get_logger

logger = get_logger(__name__)

# Create status blueprint
status_bp = Blueprint('status', __name__)


@status_bp.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        - 200: System is healthy
        - 503: System is unhealthy
    """
    try:
        # Basic health checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'uptime': _get_uptime(),
            'checks': {
                'database': _check_database_health(),
                'memory': _check_memory_health(),
                'disk': _check_disk_health(),
                'cpu': _check_cpu_health()
            }
        }
        
        # Determine overall health
        all_healthy = all(check['status'] == 'healthy' for check in health_status['checks'].values())
        
        if not all_healthy:
            health_status['status'] = 'unhealthy'
            return jsonify(health_status), 503
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 503


def _get_uptime() -> Dict[str, Any]:
    """Get system uptime information."""
    try:
        uptime_seconds = psutil.boot_time()
        uptime = datetime.fromtimestamp(uptime_seconds)
        current_time = datetime.now()
        uptime_duration = current_time - uptime
        
        return {
            'boot_time': uptime.isoformat(),
            'uptime_seconds': int(uptime_duration.total_seconds()),
            'uptime_formatted': str(uptime_duration)
        }
    except Exception as e:
        logger.error(f"Uptime check error: {e}")
        return {'error': str(e)}


def _check_database_health() -> Dict[str, Any]:
    """Check database health."""
    try:
        # This would check actual database connectivity
        # For now, return a mock healthy status
        return {
            'status': 'healthy',
            'response_time_ms': 5.2,
            'connections': 10
        }
    except Exception as e:
        logger.error(f"Database health check error: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


def _check_memory_health() -> Dict[str, Any]:
    """Check memory health."""
    try:
        memory = psutil.virtual_memory()
        
        # Consider memory healthy if usage is below 90%
        is_healthy = memory.percent < 90
        
        return {
            'status': 'healthy' if is_healthy else 'warning',
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'usage_percent': memory.percent
        }
    except Exception as e:
        logger.error(f"Memory health check error: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


def _check_disk_health() -> Dict[str, Any]:
    """Check disk health."""
    try:
        disk = psutil.disk_usage('/')
        
        # Consider disk healthy if usage is below 90%
        is_healthy = disk.percent < 90
        
        return {
            'status': 'healthy' if is_healthy else 'warning',
            'total_gb': round(disk.total / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'usage_percent': disk.percent
        }
    except Exception as e:
        logger.error(f"Disk health check error: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


def _check_cpu_health() -> Dict[str, Any]:
    """Check CPU health."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Consider CPU healthy if usage is below 80%
        is_healthy = cpu_percent < 80
        
        return {
            'status': 'healthy' if is_healthy else 'warning',
            'usage_percent': cpu_percent,
            'core_count': psutil.cpu_count(),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    except Exception as e:
        logger.error(f"CPU health check error: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


@status_bp.route('/system', methods=['GET'])
def system_info():
    """
    Get detailed system information.
    
    Returns:
        - 200: System information
    """
    try:
        system_info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation(),
                'compiler': platform.python_compiler()
            },
            'memory': _get_memory_info(),
            'disk': _get_disk_info(),
            'cpu': _get_cpu_info(),
            'network': _get_network_info(),
            'processes': _get_process_info()
        }
        
        return jsonify(system_info), 200
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        return jsonify({'error': str(e)}), 500


def _get_memory_info() -> Dict[str, Any]:
    """Get detailed memory information."""
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'virtual_memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'free_gb': round(memory.free / (1024**3), 2),
                'usage_percent': memory.percent
            },
            'swap_memory': {
                'total_gb': round(swap.total / (1024**3), 2),
                'used_gb': round(swap.used / (1024**3), 2),
                'free_gb': round(swap.free / (1024**3), 2),
                'usage_percent': swap.percent
            }
        }
    except Exception as e:
        logger.error(f"Memory info error: {e}")
        return {'error': str(e)}


def _get_disk_info() -> Dict[str, Any]:
    """Get detailed disk information."""
    try:
        disk_partitions = psutil.disk_partitions()
        disk_info = {}
        
        for partition in disk_partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info[partition.device] = {
                    'mountpoint': partition.mountpoint,
                    'filesystem': partition.fstype,
                    'total_gb': round(usage.total / (1024**3), 2),
                    'used_gb': round(usage.used / (1024**3), 2),
                    'free_gb': round(usage.free / (1024**3), 2),
                    'usage_percent': usage.percent
                }
            except Exception:
                continue
        
        return disk_info
    except Exception as e:
        logger.error(f"Disk info error: {e}")
        return {'error': str(e)}


def _get_cpu_info() -> Dict[str, Any]:
    """Get detailed CPU information."""
    try:
        cpu_freq = psutil.cpu_freq()
        
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'usage_percent': psutil.cpu_percent(interval=1),
            'frequency_mhz': round(cpu_freq.current, 2) if cpu_freq else None,
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            'per_cpu_usage': psutil.cpu_percent(interval=1, percpu=True)
        }
    except Exception as e:
        logger.error(f"CPU info error: {e}")
        return {'error': str(e)}


def _get_network_info() -> Dict[str, Any]:
    """Get network information."""
    try:
        network_io = psutil.net_io_counters()
        network_interfaces = psutil.net_if_addrs()
        
        return {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv,
            'interfaces': {
                name: {
                    'addresses': [
                        {
                            'family': str(addr.family),
                            'address': addr.address,
                            'netmask': addr.netmask
                        }
                        for addr in addrs
                    ]
                }
                for name, addrs in network_interfaces.items()
            }
        }
    except Exception as e:
        logger.error(f"Network info error: {e}")
        return {'error': str(e)}


def _get_process_info() -> Dict[str, Any]:
    """Get process information."""
    try:
        current_process = psutil.Process()
        
        return {
            'pid': current_process.pid,
            'name': current_process.name(),
            'status': current_process.status(),
            'cpu_percent': current_process.cpu_percent(),
            'memory_percent': current_process.memory_percent(),
            'memory_info': {
                'rss_mb': round(current_process.memory_info().rss / (1024**2), 2),
                'vms_mb': round(current_process.memory_info().vms / (1024**2), 2)
            },
            'create_time': datetime.fromtimestamp(current_process.create_time()).isoformat(),
            'num_threads': current_process.num_threads()
        }
    except Exception as e:
        logger.error(f"Process info error: {e}")
        return {'error': str(e)}


@status_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get system metrics for monitoring.
    
    Returns:
        - 200: System metrics
    """
    try:
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=1),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            'memory': {
                'usage_percent': psutil.virtual_memory().percent,
                'available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
            },
            'disk': {
                'usage_percent': psutil.disk_usage('/').percent,
                'free_gb': round(psutil.disk_usage('/').free / (1024**3), 2)
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            },
            'processes': {
                'total': len(psutil.pids()),
                'running': len([p for p in psutil.process_iter() if p.status() == 'running'])
            }
        }
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({'error': str(e)}), 500


@status_bp.route('/version', methods=['GET'])
def get_version():
    """
    Get application version information.
    
    Returns:
        - 200: Version information
    """
    try:
        version_info = {
            'version': '1.0.0',
            'build_date': '2024-01-01T00:00:00Z',
            'git_commit': 'abc123def456',
            'python_version': platform.python_version(),
            'dependencies': {
                'flask': '2.3.0',
                'pandas': '2.0.0',
                'scikit-learn': '1.3.0',
                'psutil': '5.9.0'
            }
        }
        
        return jsonify(version_info), 200
        
    except Exception as e:
        logger.error(f"Version info error: {e}")
        return jsonify({'error': str(e)}), 500


@status_bp.route('/ping', methods=['GET'])
def ping():
    """
    Simple ping endpoint for connectivity testing.
    
    Returns:
        - 200: Pong response
    """
    return jsonify({
        'message': 'pong',
        'timestamp': datetime.utcnow().isoformat()
    }), 200 