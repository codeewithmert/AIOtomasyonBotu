"""
Data Management Routes

Provides endpoints for data collection, storage, processing, and retrieval
operations in the AI Automation Bot.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, g
from werkzeug.utils import secure_filename
import os

from ...core.exceptions import ValidationError, DataProcessingError
from ...core.logger import get_logger
from ...data.collectors.api_collector import APICollector
from ...data.collectors.web_scraper import WebScraper

logger = get_logger(__name__)

# Create data blueprint
data_bp = Blueprint('data', __name__)

# In-memory data storage (replace with database in production)
data_store = {}
collection_jobs = {}


@data_bp.route('/collect', methods=['POST'])
def collect_data():
    """
    Collect data from various sources.
    
    Request body:
        - source_type: Type of data source ('api', 'web', 'file')
        - source_config: Configuration for data source
        - collection_name: Name for this collection
        - schedule: Optional schedule configuration
    
    Returns:
        - 200: Data collection started successfully
        - 400: Validation error
        - 500: Collection error
    """
    try:
        data = request.get_json()
        
        if not data:
            raise ValidationError("Request body is required")
        
        source_type = data.get('source_type')
        source_config = data.get('source_config')
        collection_name = data.get('collection_name')
        schedule = data.get('schedule')
        
        # Validate required fields
        if not source_type or not source_config or not collection_name:
            raise ValidationError("source_type, source_config, and collection_name are required")
        
        # Validate source type
        if source_type not in ['api', 'web', 'file']:
            raise ValidationError("source_type must be 'api', 'web', or 'file'")
        
        # Create collection job
        job_id = f"collection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{g.user_id}"
        
        collection_job = {
            'id': job_id,
            'user_id': g.user_id,
            'source_type': source_type,
            'source_config': source_config,
            'collection_name': collection_name,
            'schedule': schedule,
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'started_at': None,
            'completed_at': None,
            'data_count': 0,
            'error': None
        }
        
        collection_jobs[job_id] = collection_job
        
        # Start collection based on type
        if source_type == 'api':
            result = _collect_from_api(source_config, collection_name)
        elif source_type == 'web':
            result = _collect_from_web(source_config, collection_name)
        elif source_type == 'file':
            result = _collect_from_file(source_config, collection_name)
        
        # Update job status
        collection_job['status'] = 'completed'
        collection_job['started_at'] = datetime.utcnow()
        collection_job['completed_at'] = datetime.utcnow()
        collection_job['data_count'] = result.get('count', 0)
        
        logger.info(f"Data collection completed: {collection_name}")
        
        return jsonify({
            'message': 'Data collection completed successfully',
            'job_id': job_id,
            'collection_name': collection_name,
            'data_count': result.get('count', 0),
            'result': result
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Data collection error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Data collection failed'}), 500


def _collect_from_api(source_config: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
    """Collect data from API source."""
    try:
        collector = APICollector()
        
        # Configure collector
        url = source_config.get('url')
        method = source_config.get('method', 'GET')
        headers = source_config.get('headers', {})
        params = source_config.get('params', {})
        
        if not url:
            raise ValidationError("API URL is required")
        
        # Collect data
        if method.upper() == 'GET':
            data = collector.get_data(url, headers=headers, params=params)
        else:
            data = collector.post_data(url, headers=headers, data=source_config.get('data', {}))
        
        # Store collected data
        collection_key = f"{collection_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        data_store[collection_key] = {
            'source_type': 'api',
            'source_config': source_config,
            'collection_name': collection_name,
            'data': data,
            'collected_at': datetime.utcnow(),
            'user_id': g.user_id
        }
        
        return {
            'count': len(data) if isinstance(data, list) else 1,
            'collection_key': collection_key
        }
        
    except Exception as e:
        logger.error(f"API collection error: {e}")
        raise DataProcessingError(f"Failed to collect data from API: {str(e)}")


def _collect_from_web(source_config: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
    """Collect data from web source."""
    try:
        scraper = WebScraper()
        
        # Configure scraper
        url = source_config.get('url')
        selectors = source_config.get('selectors', {})
        wait_time = source_config.get('wait_time', 5)
        
        if not url:
            raise ValidationError("Web URL is required")
        
        # Collect data
        data = scraper.scrape_page(url, selectors=selectors, wait_time=wait_time)
        
        # Store collected data
        collection_key = f"{collection_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        data_store[collection_key] = {
            'source_type': 'web',
            'source_config': source_config,
            'collection_name': collection_name,
            'data': data,
            'collected_at': datetime.utcnow(),
            'user_id': g.user_id
        }
        
        return {
            'count': len(data) if isinstance(data, list) else 1,
            'collection_key': collection_key
        }
        
    except Exception as e:
        logger.error(f"Web collection error: {e}")
        raise DataProcessingError(f"Failed to collect data from web: {str(e)}")


def _collect_from_file(source_config: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
    """Collect data from file source."""
    try:
        file_path = source_config.get('file_path')
        file_type = source_config.get('file_type', 'json')
        
        if not file_path:
            raise ValidationError("File path is required")
        
        # Read file based on type
        if file_type == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_type == 'csv':
            import pandas as pd
            data = pd.read_csv(file_path).to_dict('records')
        else:
            raise ValidationError("Unsupported file type")
        
        # Store collected data
        collection_key = f"{collection_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        data_store[collection_key] = {
            'source_type': 'file',
            'source_config': source_config,
            'collection_name': collection_name,
            'data': data,
            'collected_at': datetime.utcnow(),
            'user_id': g.user_id
        }
        
        return {
            'count': len(data) if isinstance(data, list) else 1,
            'collection_key': collection_key
        }
        
    except Exception as e:
        logger.error(f"File collection error: {e}")
        raise DataProcessingError(f"Failed to collect data from file: {str(e)}")


@data_bp.route('/collections', methods=['GET'])
def list_collections():
    """
    List all data collections for the current user.
    
    Query parameters:
        - limit: Maximum number of collections to return
        - offset: Number of collections to skip
        - source_type: Filter by source type
    
    Returns:
        - 200: List of collections
        - 400: Validation error
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        source_type = request.args.get('source_type')
        
        # Filter collections by user
        user_collections = [
            {'key': key, **value}
            for key, value in data_store.items()
            if value['user_id'] == g.user_id
        ]
        
        # Filter by source type if specified
        if source_type:
            user_collections = [
                c for c in user_collections
                if c['source_type'] == source_type
            ]
        
        # Sort by collection date (newest first)
        user_collections.sort(key=lambda x: x['collected_at'], reverse=True)
        
        # Apply pagination
        total_count = len(user_collections)
        collections = user_collections[offset:offset + limit]
        
        return jsonify({
            'collections': collections,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }), 200
        
    except ValueError as e:
        return jsonify({'error': 'Validation Error', 'message': 'Invalid limit or offset'}), 400
    except Exception as e:
        logger.error(f"List collections error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to list collections'}), 500


@data_bp.route('/collections/<collection_key>', methods=['GET'])
def get_collection(collection_key: str):
    """
    Get specific data collection.
    
    Args:
        collection_key: Collection identifier
    
    Returns:
        - 200: Collection data
        - 404: Collection not found
        - 403: Access denied
    """
    try:
        collection = data_store.get(collection_key)
        
        if not collection:
            return jsonify({'error': 'Not Found', 'message': 'Collection not found'}), 404
        
        # Check access permission
        if collection['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        return jsonify({
            'collection': collection
        }), 200
        
    except Exception as e:
        logger.error(f"Get collection error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to get collection'}), 500


@data_bp.route('/collections/<collection_key>', methods=['DELETE'])
def delete_collection(collection_key: str):
    """
    Delete specific data collection.
    
    Args:
        collection_key: Collection identifier
    
    Returns:
        - 200: Collection deleted successfully
        - 404: Collection not found
        - 403: Access denied
    """
    try:
        collection = data_store.get(collection_key)
        
        if not collection:
            return jsonify({'error': 'Not Found', 'message': 'Collection not found'}), 404
        
        # Check access permission
        if collection['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        # Delete collection
        del data_store[collection_key]
        
        logger.info(f"Collection deleted: {collection_key}")
        
        return jsonify({'message': 'Collection deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Delete collection error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to delete collection'}), 500


@data_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """
    List all collection jobs for the current user.
    
    Query parameters:
        - limit: Maximum number of jobs to return
        - offset: Number of jobs to skip
        - status: Filter by job status
    
    Returns:
        - 200: List of jobs
        - 400: Validation error
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        status = request.args.get('status')
        
        # Filter jobs by user
        user_jobs = [
            job for job in collection_jobs.values()
            if job['user_id'] == g.user_id
        ]
        
        # Filter by status if specified
        if status:
            user_jobs = [job for job in user_jobs if job['status'] == status]
        
        # Sort by creation date (newest first)
        user_jobs.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Apply pagination
        total_count = len(user_jobs)
        jobs = user_jobs[offset:offset + limit]
        
        return jsonify({
            'jobs': jobs,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }), 200
        
    except ValueError as e:
        return jsonify({'error': 'Validation Error', 'message': 'Invalid limit or offset'}), 400
    except Exception as e:
        logger.error(f"List jobs error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to list jobs'}), 500


@data_bp.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id: str):
    """
    Get specific collection job.
    
    Args:
        job_id: Job identifier
    
    Returns:
        - 200: Job details
        - 404: Job not found
        - 403: Access denied
    """
    try:
        job = collection_jobs.get(job_id)
        
        if not job:
            return jsonify({'error': 'Not Found', 'message': 'Job not found'}), 404
        
        # Check access permission
        if job['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        return jsonify({
            'job': job
        }), 200
        
    except Exception as e:
        logger.error(f"Get job error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to get job'}), 500


@data_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload file for data collection.
    
    Returns:
        - 200: File uploaded successfully
        - 400: Validation error
        - 500: Upload error
    """
    try:
        if 'file' not in request.files:
            raise ValidationError("No file provided")
        
        file = request.files['file']
        if file.filename == '':
            raise ValidationError("No file selected")
        
        # Validate file type
        allowed_extensions = {'json', 'csv', 'txt', 'xlsx'}
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise ValidationError(f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}")
        
        # Save file
        filename = secure_filename(file.filename)
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {filename}")
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'file_path': file_path,
            'file_size': os.path.getsize(file_path)
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'File upload failed'}), 500 