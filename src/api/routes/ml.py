"""
Machine Learning Routes

Provides endpoints for ML model training, prediction, evaluation, and management
operations in the AI Automation Bot.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, g
import pandas as pd

from ...core.exceptions import ValidationError, MLModelError
from ...core.logger import get_logger
from ...ml.models.random_forest_model import RandomForestModel
from ...ml.pipeline.ml_pipeline import MLPipeline

logger = get_logger(__name__)

# Create ML blueprint
ml_bp = Blueprint('ml', __name__)

# In-memory ML storage (replace with database in production)
models_db = {}
training_jobs = {}
predictions_db = {}


@ml_bp.route('/train', methods=['POST'])
def train_model():
    """
    Train a new ML model.
    
    Request body:
        - model_type: Type of model ('random_forest', 'linear_regression', etc.)
        - task_type: Type of task ('classification', 'regression')
        - data_config: Configuration for training data
        - hyperparameters: Model hyperparameters (optional)
        - model_name: Name for the model
    
    Returns:
        - 200: Model training started successfully
        - 400: Validation error
        - 500: Training error
    """
    try:
        data = request.get_json()
        
        if not data:
            raise ValidationError("Request body is required")
        
        model_type = data.get('model_type')
        task_type = data.get('task_type')
        data_config = data.get('data_config')
        hyperparameters = data.get('hyperparameters', {})
        model_name = data.get('model_name')
        
        # Validate required fields
        if not model_type or not task_type or not data_config or not model_name:
            raise ValidationError("model_type, task_type, data_config, and model_name are required")
        
        # Validate model type
        if model_type not in ['random_forest', 'linear_regression', 'logistic_regression']:
            raise ValidationError("Unsupported model type")
        
        # Validate task type
        if task_type not in ['classification', 'regression']:
            raise ValidationError("task_type must be 'classification' or 'regression'")
        
        # Create training job
        job_id = f"training_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{g.user_id}"
        
        training_job = {
            'id': job_id,
            'user_id': g.user_id,
            'model_type': model_type,
            'task_type': task_type,
            'data_config': data_config,
            'hyperparameters': hyperparameters,
            'model_name': model_name,
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'started_at': None,
            'completed_at': None,
            'metrics': {},
            'error': None
        }
        
        training_jobs[job_id] = training_job
        
        # Start training
        result = _train_model_job(training_job)
        
        # Update job status
        training_job['status'] = 'completed'
        training_job['started_at'] = datetime.utcnow()
        training_job['completed_at'] = datetime.utcnow()
        training_job['metrics'] = result.get('metrics', {})
        
        # Create model record
        model_id = str(uuid.uuid4())
        model_record = {
            'id': model_id,
            'user_id': g.user_id,
            'job_id': job_id,
            'model_type': model_type,
            'task_type': task_type,
            'model_name': model_name,
            'hyperparameters': hyperparameters,
            'metrics': result.get('metrics', {}),
            'model_path': result.get('model_path'),
            'created_at': datetime.utcnow(),
            'is_active': True
        }
        
        models_db[model_id] = model_record
        
        logger.info(f"Model training completed: {model_name}")
        
        return jsonify({
            'message': 'Model training completed successfully',
            'job_id': job_id,
            'model_id': model_id,
            'model_name': model_name,
            'metrics': result.get('metrics', {})
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Model training error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Model training failed'}), 500


def _train_model_job(training_job: Dict[str, Any]) -> Dict[str, Any]:
    """Execute model training job."""
    try:
        model_type = training_job['model_type']
        task_type = training_job['task_type']
        data_config = training_job['data_config']
        hyperparameters = training_job['hyperparameters']
        
        # Load training data
        data = _load_training_data(data_config)
        
        # Create and train model
        if model_type == 'random_forest':
            model = RandomForestModel()
            
            # Configure model
            if task_type == 'classification':
                model.configure_classification(**hyperparameters)
            else:
                model.configure_regression(**hyperparameters)
            
            # Train model
            model.train(data['X_train'], data['y_train'])
            
            # Evaluate model
            metrics = model.evaluate(data['X_test'], data['y_test'])
            
            # Save model
            model_path = f"models/{training_job['model_name']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            model.save_model(model_path)
            
            return {
                'metrics': metrics,
                'model_path': model_path
            }
        
        else:
            raise MLModelError(f"Model type {model_type} not implemented")
        
    except Exception as e:
        logger.error(f"Training job error: {e}")
        raise MLModelError(f"Training failed: {str(e)}")


def _load_training_data(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load training data based on configuration."""
    try:
        data_source = data_config.get('source')
        target_column = data_config.get('target_column')
        test_size = data_config.get('test_size', 0.2)
        
        if not data_source or not target_column:
            raise ValidationError("data_source and target_column are required")
        
        # Load data from source
        if data_source.startswith('collection:'):
            collection_key = data_source.split(':', 1)[1]
            collection = data_store.get(collection_key)
            
            if not collection:
                raise ValidationError(f"Collection not found: {collection_key}")
            
            # Convert to DataFrame
            df = pd.DataFrame(collection['data'])
            
        elif data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
        elif data_source.endswith('.json'):
            df = pd.read_json(data_source)
        else:
            raise ValidationError(f"Unsupported data source: {data_source}")
        
        # Prepare features and target
        if target_column not in df.columns:
            raise ValidationError(f"Target column not found: {target_column}")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        raise MLModelError(f"Failed to load training data: {str(e)}")


@ml_bp.route('/models', methods=['GET'])
def list_models():
    """
    List all ML models for the current user.
    
    Query parameters:
        - limit: Maximum number of models to return
        - offset: Number of models to skip
        - model_type: Filter by model type
        - task_type: Filter by task type
    
    Returns:
        - 200: List of models
        - 400: Validation error
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        model_type = request.args.get('model_type')
        task_type = request.args.get('task_type')
        
        # Filter models by user
        user_models = [
            model for model in models_db.values()
            if model['user_id'] == g.user_id
        ]
        
        # Apply filters
        if model_type:
            user_models = [m for m in user_models if m['model_type'] == model_type]
        
        if task_type:
            user_models = [m for m in user_models if m['task_type'] == task_type]
        
        # Sort by creation date (newest first)
        user_models.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Apply pagination
        total_count = len(user_models)
        models = user_models[offset:offset + limit]
        
        return jsonify({
            'models': models,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }), 200
        
    except ValueError as e:
        return jsonify({'error': 'Validation Error', 'message': 'Invalid limit or offset'}), 400
    except Exception as e:
        logger.error(f"List models error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to list models'}), 500


@ml_bp.route('/models/<model_id>', methods=['GET'])
def get_model(model_id: str):
    """
    Get specific ML model.
    
    Args:
        model_id: Model identifier
    
    Returns:
        - 200: Model details
        - 404: Model not found
        - 403: Access denied
    """
    try:
        model = models_db.get(model_id)
        
        if not model:
            return jsonify({'error': 'Not Found', 'message': 'Model not found'}), 404
        
        # Check access permission
        if model['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        return jsonify({
            'model': model
        }), 200
        
    except Exception as e:
        logger.error(f"Get model error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to get model'}), 500


@ml_bp.route('/models/<model_id>/predict', methods=['POST'])
def predict(model_id: str):
    """
    Make predictions using a trained model.
    
    Args:
        model_id: Model identifier
    
    Request body:
        - data: Input data for prediction
        - format: Data format ('json', 'csv')
    
    Returns:
        - 200: Predictions
        - 400: Validation error
        - 404: Model not found
        - 403: Access denied
    """
    try:
        data = request.get_json()
        
        if not data:
            raise ValidationError("Request body is required")
        
        input_data = data.get('data')
        data_format = data.get('format', 'json')
        
        if not input_data:
            raise ValidationError("Input data is required")
        
        # Get model
        model = models_db.get(model_id)
        
        if not model:
            return jsonify({'error': 'Not Found', 'message': 'Model not found'}), 404
        
        # Check access permission
        if model['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        # Load model
        predictions = _make_predictions(model, input_data, data_format)
        
        # Store prediction
        prediction_id = str(uuid.uuid4())
        predictions_db[prediction_id] = {
            'id': prediction_id,
            'model_id': model_id,
            'user_id': g.user_id,
            'input_data': input_data,
            'predictions': predictions,
            'created_at': datetime.utcnow()
        }
        
        logger.info(f"Predictions made with model: {model['model_name']}")
        
        return jsonify({
            'predictions': predictions,
            'prediction_id': prediction_id,
            'model_name': model['model_name']
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Prediction failed'}), 500


def _make_predictions(model: Dict[str, Any], input_data: Any, data_format: str) -> List[Any]:
    """Make predictions using the model."""
    try:
        model_type = model['model_type']
        model_path = model['model_path']
        
        # Load model
        if model_type == 'random_forest':
            ml_model = RandomForestModel()
            ml_model.load_model(model_path)
        
        # Prepare input data
        if data_format == 'json':
            if isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = pd.DataFrame([input_data])
        elif data_format == 'csv':
            df = pd.read_csv(input_data)
        else:
            raise ValidationError("Unsupported data format")
        
        # Make predictions
        predictions = ml_model.predict(df)
        
        return predictions.tolist()
        
    except Exception as e:
        logger.error(f"Prediction execution error: {e}")
        raise MLModelError(f"Prediction failed: {str(e)}")


@ml_bp.route('/models/<model_id>/evaluate', methods=['POST'])
def evaluate_model(model_id: str):
    """
    Evaluate a trained model.
    
    Args:
        model_id: Model identifier
    
    Request body:
        - test_data: Test data for evaluation
        - metrics: List of metrics to calculate
    
    Returns:
        - 200: Evaluation results
        - 400: Validation error
        - 404: Model not found
        - 403: Access denied
    """
    try:
        data = request.get_json()
        
        if not data:
            raise ValidationError("Request body is required")
        
        test_data = data.get('test_data')
        metrics = data.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        
        if not test_data:
            raise ValidationError("Test data is required")
        
        # Get model
        model = models_db.get(model_id)
        
        if not model:
            return jsonify({'error': 'Not Found', 'message': 'Model not found'}), 404
        
        # Check access permission
        if model['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        # Evaluate model
        evaluation_results = _evaluate_model(model, test_data, metrics)
        
        logger.info(f"Model evaluated: {model['model_name']}")
        
        return jsonify({
            'evaluation': evaluation_results,
            'model_name': model['model_name']
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Model evaluation error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Model evaluation failed'}), 500


def _evaluate_model(model: Dict[str, Any], test_data: Any, metrics: List[str]) -> Dict[str, Any]:
    """Evaluate the model with test data."""
    try:
        model_type = model['model_type']
        model_path = model['model_path']
        
        # Load model
        if model_type == 'random_forest':
            ml_model = RandomForestModel()
            ml_model.load_model(model_path)
        
        # Prepare test data
        if isinstance(test_data, dict):
            X_test = pd.DataFrame([test_data['features']])
            y_test = test_data['target']
        else:
            X_test = pd.DataFrame(test_data['features'])
            y_test = test_data['target']
        
        # Evaluate model
        results = ml_model.evaluate(X_test, y_test, metrics=metrics)
        
        return results
        
    except Exception as e:
        logger.error(f"Model evaluation execution error: {e}")
        raise MLModelError(f"Model evaluation failed: {str(e)}")


@ml_bp.route('/models/<model_id>', methods=['DELETE'])
def delete_model(model_id: str):
    """
    Delete specific ML model.
    
    Args:
        model_id: Model identifier
    
    Returns:
        - 200: Model deleted successfully
        - 404: Model not found
        - 403: Access denied
    """
    try:
        model = models_db.get(model_id)
        
        if not model:
            return jsonify({'error': 'Not Found', 'message': 'Model not found'}), 404
        
        # Check access permission
        if model['user_id'] != g.user_id:
            return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
        
        # Delete model file
        import os
        if model.get('model_path') and os.path.exists(model['model_path']):
            os.remove(model['model_path'])
        
        # Delete model record
        del models_db[model_id]
        
        logger.info(f"Model deleted: {model['model_name']}")
        
        return jsonify({'message': 'Model deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Delete model error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to delete model'}), 500


@ml_bp.route('/jobs', methods=['GET'])
def list_training_jobs():
    """
    List all training jobs for the current user.
    
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
            job for job in training_jobs.values()
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
        logger.error(f"List training jobs error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to list jobs'}), 500 