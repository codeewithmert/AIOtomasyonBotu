"""
Error Handler Middleware

Provides centralized error handling for the AI Automation Bot API,
including error logging, user-friendly responses, and error tracking.
"""

import traceback
from typing import Dict, Any, Optional, Callable
from flask import request, g, jsonify
from werkzeug.exceptions import HTTPException

from ...core.exceptions import (
    BaseError, ValidationError, AuthenticationError, AuthorizationError,
    RateLimitExceededError, DataProcessingError, MLModelError,
    AutomationError, ConfigurationError, DatabaseError, NetworkError
)
from ...core.logger import get_logger

logger = get_logger(__name__)


class ErrorHandlerMiddleware:
    """Error handling middleware for API endpoints."""
    
    def __init__(self, log_errors: bool = True, include_traceback: bool = False,
                 error_handlers: Optional[Dict] = None):
        """
        Initialize error handling middleware.
        
        Args:
            log_errors: Whether to log errors
            include_traceback: Whether to include traceback in responses
            error_handlers: Custom error handlers
        """
        self.log_errors = log_errors
        self.include_traceback = include_traceback
        self.error_handlers = error_handlers or {}
        
        # Default error mappings
        self.error_mappings = {
            ValidationError: {'status_code': 400, 'error_type': 'VALIDATION_ERROR'},
            AuthenticationError: {'status_code': 401, 'error_type': 'AUTHENTICATION_ERROR'},
            AuthorizationError: {'status_code': 403, 'error_type': 'AUTHORIZATION_ERROR'},
            RateLimitExceededError: {'status_code': 429, 'error_type': 'RATE_LIMIT_ERROR'},
            DataProcessingError: {'status_code': 422, 'error_type': 'DATA_PROCESSING_ERROR'},
            MLModelError: {'status_code': 500, 'error_type': 'ML_MODEL_ERROR'},
            AutomationError: {'status_code': 500, 'error_type': 'AUTOMATION_ERROR'},
            ConfigurationError: {'status_code': 500, 'error_type': 'CONFIGURATION_ERROR'},
            DatabaseError: {'status_code': 500, 'error_type': 'DATABASE_ERROR'},
            NetworkError: {'status_code': 503, 'error_type': 'NETWORK_ERROR'},
        }
    
    def init_app(self, app):
        """
        Initialize error handling for Flask app.
        
        Args:
            app: Flask application instance
        """
        # Register error handlers for custom exceptions
        for exception_class, handler_info in self.error_mappings.items():
            @app.errorhandler(exception_class)
            def handle_custom_error(error, exception_class=exception_class, handler_info=handler_info):
                return self._handle_error(error, handler_info)
        
        # Register general exception handler
        @app.errorhandler(Exception)
        def handle_general_error(error):
            return self._handle_general_error(error)
        
        # Register HTTP exception handler
        @app.errorhandler(HTTPException)
        def handle_http_error(error):
            return self._handle_http_error(error)
        
        logger.info("Error handling middleware initialized")
    
    def _handle_error(self, error: BaseError, handler_info: Dict[str, Any]):
        """
        Handle custom application errors.
        
        Args:
            error: Custom error instance
            handler_info: Error handler information
            
        Returns:
            JSON error response
        """
        status_code = handler_info['status_code']
        error_type = handler_info['error_type']
        
        # Log error
        if self.log_errors:
            self._log_error(error, error_type, status_code)
        
        # Build response
        response_data = {
            'error': error_type,
            'message': str(error),
            'code': status_code,
            'request_id': getattr(g, 'request_id', None)
        }
        
        # Add additional error context
        if hasattr(error, 'context'):
            response_data['context'] = error.context
        
        # Add traceback if enabled
        if self.include_traceback:
            response_data['traceback'] = traceback.format_exc()
        
        return jsonify(response_data), status_code
    
    def _handle_general_error(self, error: Exception):
        """
        Handle general exceptions.
        
        Args:
            error: General exception
            
        Returns:
            JSON error response
        """
        # Log error
        if self.log_errors:
            self._log_error(error, 'INTERNAL_SERVER_ERROR', 500)
        
        # Build response
        response_data = {
            'error': 'INTERNAL_SERVER_ERROR',
            'message': 'An unexpected error occurred',
            'code': 500,
            'request_id': getattr(g, 'request_id', None)
        }
        
        # Add traceback if enabled
        if self.include_traceback:
            response_data['traceback'] = traceback.format_exc()
        
        return jsonify(response_data), 500
    
    def _handle_http_error(self, error: HTTPException):
        """
        Handle HTTP exceptions.
        
        Args:
            error: HTTP exception
            
        Returns:
            JSON error response
        """
        # Log error
        if self.log_errors:
            self._log_error(error, 'HTTP_ERROR', error.code)
        
        # Build response
        response_data = {
            'error': error.name,
            'message': error.description,
            'code': error.code,
            'request_id': getattr(g, 'request_id', None)
        }
        
        return jsonify(response_data), error.code
    
    def _log_error(self, error: Exception, error_type: str, status_code: int):
        """
        Log error details.
        
        Args:
            error: Error instance
            error_type: Type of error
            status_code: HTTP status code
        """
        try:
            # Get request information
            request_info = {
                'method': request.method,
                'url': request.url,
                'path': request.path,
                'remote_addr': request.remote_addr,
                'user_agent': request.user_agent.string if request.user_agent else None
            }
            
            # Get error information
            error_info = {
                'error_type': error_type,
                'error_message': str(error),
                'status_code': status_code,
                'exception_class': type(error).__name__,
                'traceback': traceback.format_exc()
            }
            
            # Log based on status code
            if status_code >= 500:
                logger.error("Server Error", extra={
                    'request_info': request_info,
                    'error_info': error_info,
                    'request_id': getattr(g, 'request_id', None)
                })
            elif status_code >= 400:
                logger.warning("Client Error", extra={
                    'request_info': request_info,
                    'error_info': error_info,
                    'request_id': getattr(g, 'request_id', None)
                })
            else:
                logger.info("Application Error", extra={
                    'request_info': request_info,
                    'error_info': error_info,
                    'request_id': getattr(g, 'request_id', None)
                })
                
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    def register_error_handler(self, exception_class: type, 
                             handler: Callable[[Exception], tuple]):
        """
        Register custom error handler.
        
        Args:
            exception_class: Exception class to handle
            handler: Error handler function
        """
        self.error_handlers[exception_class] = handler
        logger.info(f"Registered custom error handler for {exception_class.__name__}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        # This would typically integrate with a metrics system
        # For now, return basic structure
        return {
            'total_errors': 0,
            'error_types': {},
            'status_codes': {},
            'recent_errors': []
        }
    
    def format_error_response(self, error: Exception, include_details: bool = False) -> Dict[str, Any]:
        """
        Format error for response.
        
        Args:
            error: Error instance
            include_details: Whether to include detailed error information
            
        Returns:
            Formatted error dictionary
        """
        response = {
            'error': type(error).__name__,
            'message': str(error),
            'request_id': getattr(g, 'request_id', None)
        }
        
        if include_details:
            response['details'] = {
                'exception_class': type(error).__name__,
                'traceback': traceback.format_exc(),
                'context': getattr(error, 'context', None)
            }
        
        return response


def init_error_handler_middleware(app, config: dict):
    """
    Initialize error handling middleware from configuration.
    
    Args:
        app: Flask application instance
        config: Configuration dictionary
    """
    error_handler_config = config.get('error_handler', {})
    
    middleware = ErrorHandlerMiddleware(
        log_errors=error_handler_config.get('log_errors', True),
        include_traceback=error_handler_config.get('include_traceback', False),
        error_handlers=error_handler_config.get('error_handlers', {})
    )
    
    middleware.init_app(app)
    app.error_handler_middleware = middleware 