"""
Automation Routes

Provides endpoints for task scheduling, execution, monitoring, and management
operations in the AI Automation Bot.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, g

from ...core.exceptions import ValidationError, AutomationError
from ...core.logger import get_logger
from ...automation.scheduler import AutomationScheduler

logger = get_logger(__name__)

# Create automation blueprint
automation_bp = Blueprint('automation', __name__)

# In-memory automation storage (replace with database in production)
tasks_db = {}
executions_db = {}


@automation_bp.route('/tasks', methods=['POST'])
def create_task():
    """
    Create a new automation task.
    
    Request body:
        - name: Task name
        - description: Task description
        - task_type: Type of task ('data_collection', 'ml_training', 'reporting', 'custom')
        - schedule: Schedule configuration
        - config: Task-specific configuration
        - enabled: Whether task is enabled
    
    Returns:
        - 201: Task created successfully
        - 400: Validation error
        - 500: Creation error
    """
    try:
        data = request.get_json()
        
        if not data:
            raise ValidationError("Request body is required")
        
        name = data.get('name')
        description = data.get('description', '')
        task_type = data.get('task_type')
        schedule = data.get('schedule')
        config = data.get('config', {})
        enabled = data.get('enabled', True)
        
        # Validate required fields
        if not name or not task_type or not schedule:
            raise ValidationError("name, task_type, and schedule are required")
        
        # Validate task type
        if task_type not in ['data_collection', 'ml_training', 'reporting', 'custom']:
            raise ValidationError("Invalid task type")
        
        # Validate schedule
        if not _validate_schedule(schedule):
            raise ValidationError("Invalid schedule configuration")
        
        # Create task
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'user_id': g.user_id,
            'name': name,
            'description': description,
            'task_type': task_type,
            'schedule': schedule,
            'config': config,
            'enabled': enabled,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'last_execution': None,
            'next_execution': _calculate_next_execution(schedule),
            'execution_count': 0,
            'success_count': 0,
            'failure_count': 0
        }
        
        tasks_db[task_id] = task
        
        # Schedule task if enabled
        if enabled:
            _schedule_task(task)
        
        logger.info(f"Automation task created: {name}")
        
        return jsonify({
            'message': 'Task created successfully',
            'task_id': task_id,
            'task': task
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Create task error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to create task'}), 500


def _validate_schedule(schedule: Dict[str, Any]) -> bool:
    """Validate schedule configuration."""
    try:
        schedule_type = schedule.get('type')
        
        if schedule_type == 'interval':
            interval = schedule.get('interval')
            return interval and isinstance(interval, int) and interval > 0
        
        elif schedule_type == 'cron':
            cron_expression = schedule.get('cron')
            return cron_expression and isinstance(cron_expression, str)
        
        elif schedule_type == 'daily':
            time = schedule.get('time')
            return time and isinstance(time, str)
        
        elif schedule_type == 'weekly':
            day = schedule.get('day')
            time = schedule.get('time')
            return day and time and isinstance(day, str) and isinstance(time, str)
        
        else:
            return False
            
    except Exception:
        return False


def _calculate_next_execution(schedule: Dict[str, Any]) -> Optional[datetime]:
    """Calculate next execution time based on schedule."""
    try:
        schedule_type = schedule.get('type')
        now = datetime.utcnow()
        
        if schedule_type == 'interval':
            interval = schedule.get('interval', 3600)  # Default 1 hour
            return now + timedelta(seconds=interval)
        
        elif schedule_type == 'daily':
            time_str = schedule.get('time', '00:00')
            hour, minute = map(int, time_str.split(':'))
            next_exec = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if next_exec <= now:
                next_exec += timedelta(days=1)
            
            return next_exec
        
        elif schedule_type == 'weekly':
            day = schedule.get('day', 'monday')
            time_str = schedule.get('time', '00:00')
            hour, minute = map(int, time_str.split(':'))
            
            # Map day names to numbers (0=Monday, 6=Sunday)
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            
            target_day = day_map.get(day.lower(), 0)
            current_day = now.weekday()
            
            days_ahead = target_day - current_day
            if days_ahead <= 0:
                days_ahead += 7
            
            next_exec = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            next_exec += timedelta(days=days_ahead)
            
            return next_exec
        
        else:
            return None
            
    except Exception:
        return None


def _schedule_task(task: Dict[str, Any]):
    """Schedule task for execution."""
    try:
        # This would integrate with the actual scheduler
        # For now, just log the scheduling
        logger.info(f"Task scheduled: {task['name']} - Next execution: {task['next_execution']}")
        
    except Exception as e:
        logger.error(f"Task scheduling error: {e}")


@automation_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """
    List all automation tasks for the current user.
    
    Query parameters:
        - limit: Maximum number of tasks to return
        - offset: Number of tasks to skip
        - task_type: Filter by task type
        - enabled: Filter by enabled status
    
    Returns:
        - 200: List of tasks
        - 400: Validation error
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        task_type = request.args.get('task_type')
        enabled = request.args.get('enabled')
        
        # Filter tasks by user
        user_tasks = [
            task for task in tasks_db.values()
            if task['user_id'] == g.user_id
        ]
        
        # Apply filters
        if task_type:
            user_tasks = [t for t in user_tasks if t['task_type'] == task_type]
        
        if enabled is not None:
            enabled_bool = enabled.lower() == 'true'
            user_tasks = [t for t in user_tasks if t['enabled'] == enabled_bool]
        
        # Sort by creation date (newest first)
        user_tasks.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Apply pagination
        total_count = len(user_tasks)
        tasks = user_tasks[offset:offset + limit]
        
        return jsonify({
            'tasks': tasks,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }), 200
        
    except ValueError as e:
        return jsonify({'error': 'Validation Error', 'message': 'Invalid limit or offset'}), 400
    except Exception as e:
        logger.error(f"List tasks error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to list tasks'}), 500


@automation_bp.route('/tasks/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """
    Get specific automation task.
    
    Args:
        task_id: Task identifier
    
    Returns:
        - 200: Task details
        - 404: Task not found
        - 403: Access denied
    """
    try:
        task = tasks_db.get(task_id)
        
        if not task:
            return jsonify({'error': 'Not Found', 'message': 'Task not found'}), 404
        
        # Check access permission
        if task['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        return jsonify({
            'task': task
        }), 200
        
    except Exception as e:
        logger.error(f"Get task error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to get task'}), 500


@automation_bp.route('/tasks/<task_id>', methods=['PUT'])
def update_task(task_id: str):
    """
    Update automation task.
    
    Args:
        task_id: Task identifier
    
    Request body:
        - name: Task name
        - description: Task description
        - schedule: Schedule configuration
        - config: Task-specific configuration
        - enabled: Whether task is enabled
    
    Returns:
        - 200: Task updated successfully
        - 400: Validation error
        - 404: Task not found
        - 403: Access denied
    """
    try:
        data = request.get_json()
        
        task = tasks_db.get(task_id)
        
        if not task:
            return jsonify({'error': 'Not Found', 'message': 'Task not found'}), 404
        
        # Check access permission
        if task['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        # Update fields
        if 'name' in data:
            task['name'] = data['name']
        
        if 'description' in data:
            task['description'] = data['description']
        
        if 'schedule' in data:
            if not _validate_schedule(data['schedule']):
                raise ValidationError("Invalid schedule configuration")
            task['schedule'] = data['schedule']
            task['next_execution'] = _calculate_next_execution(data['schedule'])
        
        if 'config' in data:
            task['config'] = data['config']
        
        if 'enabled' in data:
            task['enabled'] = data['enabled']
            if data['enabled']:
                _schedule_task(task)
        
        task['updated_at'] = datetime.utcnow()
        
        logger.info(f"Task updated: {task['name']}")
        
        return jsonify({
            'message': 'Task updated successfully',
            'task': task
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Update task error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to update task'}), 500


@automation_bp.route('/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id: str):
    """
    Delete automation task.
    
    Args:
        task_id: Task identifier
    
    Returns:
        - 200: Task deleted successfully
        - 404: Task not found
        - 403: Access denied
    """
    try:
        task = tasks_db.get(task_id)
        
        if not task:
            return jsonify({'error': 'Not Found', 'message': 'Task not found'}), 404
        
        # Check access permission
        if task['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        # Delete task
        del tasks_db[task_id]
        
        logger.info(f"Task deleted: {task['name']}")
        
        return jsonify({'message': 'Task deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Delete task error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to delete task'}), 500


@automation_bp.route('/tasks/<task_id>/execute', methods=['POST'])
def execute_task(task_id: str):
    """
    Execute task immediately.
    
    Args:
        task_id: Task identifier
    
    Returns:
        - 200: Task executed successfully
        - 404: Task not found
        - 403: Access denied
        - 500: Execution error
    """
    try:
        task = tasks_db.get(task_id)
        
        if not task:
            return jsonify({'error': 'Not Found', 'message': 'Task not found'}), 404
        
        # Check access permission
        if task['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        # Execute task
        execution_id = str(uuid.uuid4())
        execution = {
            'id': execution_id,
            'task_id': task_id,
            'user_id': g.user_id,
            'started_at': datetime.utcnow(),
            'status': 'running',
            'result': None,
            'error': None
        }
        
        executions_db[execution_id] = execution
        
        # Execute task based on type
        result = _execute_task(task)
        
        # Update execution
        execution['status'] = 'completed'
        execution['completed_at'] = datetime.utcnow()
        execution['result'] = result
        
        # Update task statistics
        task['last_execution'] = datetime.utcnow()
        task['execution_count'] += 1
        task['success_count'] += 1
        
        logger.info(f"Task executed: {task['name']}")
        
        return jsonify({
            'message': 'Task executed successfully',
            'execution_id': execution_id,
            'result': result
        }), 200
        
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        
        # Update execution with error
        if 'execution' in locals():
            execution['status'] = 'failed'
            execution['completed_at'] = datetime.utcnow()
            execution['error'] = str(e)
            
            # Update task statistics
            task['last_execution'] = datetime.utcnow()
            task['execution_count'] += 1
            task['failure_count'] += 1
        
        return jsonify({'error': 'Internal Server Error', 'message': 'Task execution failed'}), 500


def _execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute task based on its type."""
    try:
        task_type = task['task_type']
        config = task['config']
        
        if task_type == 'data_collection':
            return _execute_data_collection_task(config)
        
        elif task_type == 'ml_training':
            return _execute_ml_training_task(config)
        
        elif task_type == 'reporting':
            return _execute_reporting_task(config)
        
        elif task_type == 'custom':
            return _execute_custom_task(config)
        
        else:
            raise AutomationError(f"Unknown task type: {task_type}")
        
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise AutomationError(f"Task execution failed: {str(e)}")


def _execute_data_collection_task(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute data collection task."""
    # This would integrate with the data collection system
    return {
        'type': 'data_collection',
        'status': 'completed',
        'data_collected': 100,
        'timestamp': datetime.utcnow().isoformat()
    }


def _execute_ml_training_task(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute ML training task."""
    # This would integrate with the ML system
    return {
        'type': 'ml_training',
        'status': 'completed',
        'model_trained': True,
        'accuracy': 0.85,
        'timestamp': datetime.utcnow().isoformat()
    }


def _execute_reporting_task(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute reporting task."""
    # This would integrate with the reporting system
    return {
        'type': 'reporting',
        'status': 'completed',
        'report_generated': True,
        'report_url': '/reports/latest',
        'timestamp': datetime.utcnow().isoformat()
    }


def _execute_custom_task(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute custom task."""
    # This would execute custom user-defined logic
    return {
        'type': 'custom',
        'status': 'completed',
        'custom_result': config.get('result', 'Task completed'),
        'timestamp': datetime.utcnow().isoformat()
    }


@automation_bp.route('/executions', methods=['GET'])
def list_executions():
    """
    List task executions for the current user.
    
    Query parameters:
        - limit: Maximum number of executions to return
        - offset: Number of executions to skip
        - task_id: Filter by task ID
        - status: Filter by execution status
    
    Returns:
        - 200: List of executions
        - 400: Validation error
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        task_id = request.args.get('task_id')
        status = request.args.get('status')
        
        # Filter executions by user
        user_executions = [
            execution for execution in executions_db.values()
            if execution['user_id'] == g.user_id
        ]
        
        # Apply filters
        if task_id:
            user_executions = [e for e in user_executions if e['task_id'] == task_id]
        
        if status:
            user_executions = [e for e in user_executions if e['status'] == status]
        
        # Sort by start time (newest first)
        user_executions.sort(key=lambda x: x['started_at'], reverse=True)
        
        # Apply pagination
        total_count = len(user_executions)
        executions = user_executions[offset:offset + limit]
        
        return jsonify({
            'executions': executions,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }), 200
        
    except ValueError as e:
        return jsonify({'error': 'Validation Error', 'message': 'Invalid limit or offset'}), 400
    except Exception as e:
        logger.error(f"List executions error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to list executions'}), 500


@automation_bp.route('/executions/<execution_id>', methods=['GET'])
def get_execution(execution_id: str):
    """
    Get specific task execution.
    
    Args:
        execution_id: Execution identifier
    
    Returns:
        - 200: Execution details
        - 404: Execution not found
        - 403: Access denied
    """
    try:
        execution = executions_db.get(execution_id)
        
        if not execution:
            return jsonify({'error': 'Not Found', 'message': 'Execution not found'}), 404
        
        # Check access permission
        if execution['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        return jsonify({
            'execution': execution
        }), 200
        
    except Exception as e:
        logger.error(f"Get execution error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to get execution'}), 500 