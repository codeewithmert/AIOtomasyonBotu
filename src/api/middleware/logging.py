"""
Logging Middleware

Provides comprehensive request/response logging for the AI Automation Bot API,
including performance metrics, structured logging, and error tracking.
"""

import time
import uuid
from typing import Dict, Any, Optional
from flask import request, g, jsonify
from werkzeug.exceptions import HTTPException

from ...core.logger import get_logger
from ...core.utils import sanitize_data

logger = get_logger(__name__)


class LoggingMiddleware:
    """Logging middleware for API requests and responses."""
    
    def __init__(self, log_requests: bool = True, log_responses: bool = True,
                 log_errors: bool = True, log_performance: bool = True,
                 sensitive_headers: Optional[list] = None,
                 sensitive_params: Optional[list] = None):
        """
        Initialize logging middleware.
        
        Args:
            log_requests: Whether to log incoming requests
            log_responses: Whether to log responses
            log_errors: Whether to log errors
            log_performance: Whether to log performance metrics
            sensitive_headers: Headers to sanitize in logs
            sensitive_params: Query parameters to sanitize in logs
        """
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_errors = log_errors
        self.log_performance = log_performance
        self.sensitive_headers = sensitive_headers or ['authorization', 'cookie', 'x-api-key']
        self.sensitive_params = sensitive_params or ['password', 'token', 'key']
    
    def init_app(self, app):
        """
        Initialize logging for Flask app.
        
        Args:
            app: Flask application instance
        """
        # Before request logging
        @app.before_request
        def before_request():
            # Generate request ID
            g.request_id = str(uuid.uuid4())
            g.start_time = time.time()
            
            if self.log_requests:
                self._log_request()
        
        # After request logging
        @app.after_request
        def after_request(response):
            if self.log_responses:
                self._log_response(response)
            
            if self.log_performance:
                self._log_performance(response)
            
            # Add request ID to response headers
            response.headers['X-Request-ID'] = g.request_id
            
            return response
        
        # Error logging
        @app.errorhandler(Exception)
        def handle_exception(error):
            if self.log_errors:
                self._log_error(error)
            
            # Return JSON error response
            if isinstance(error, HTTPException):
                response = jsonify({
                    'error': error.name,
                    'message': error.description,
                    'code': error.code,
                    'request_id': g.request_id
                })
                response.status_code = error.code
            else:
                response = jsonify({
                    'error': 'Internal Server Error',
                    'message': 'An unexpected error occurred',
                    'code': 500,
                    'request_id': g.request_id
                })
                response.status_code = 500
            
            return response
        
        logger.info("Logging middleware initialized")
    
    def _log_request(self):
        """Log incoming request details."""
        try:
            # Get request data
            method = request.method
            url = request.url
            path = request.path
            query_string = request.query_string.decode('utf-8')
            headers = dict(request.headers)
            remote_addr = request.remote_addr
            user_agent = request.user_agent.string
            
            # Sanitize sensitive data
            headers = self._sanitize_headers(headers)
            query_params = self._sanitize_query_params(query_string)
            
            # Log request
            log_data = {
                'request_id': g.request_id,
                'method': method,
                'url': url,
                'path': path,
                'query_params': query_params,
                'headers': headers,
                'remote_addr': remote_addr,
                'user_agent': user_agent,
                'timestamp': time.time()
            }
            
            logger.info("API Request", extra=log_data)
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    def _log_response(self, response):
        """Log response details."""
        try:
            # Get response data
            status_code = response.status_code
            content_length = len(response.get_data())
            content_type = response.content_type
            
            # Log response
            log_data = {
                'request_id': g.request_id,
                'status_code': status_code,
                'content_length': content_length,
                'content_type': content_type,
                'timestamp': time.time()
            }
            
            logger.info("API Response", extra=log_data)
            
        except Exception as e:
            logger.error(f"Error logging response: {e}")
    
    def _log_performance(self, response):
        """Log performance metrics."""
        try:
            # Calculate performance metrics
            duration = time.time() - g.start_time
            status_code = response.status_code
            
            # Log performance
            log_data = {
                'request_id': g.request_id,
                'duration_ms': round(duration * 1000, 2),
                'status_code': status_code,
                'timestamp': time.time()
            }
            
            # Use different log levels based on performance
            if duration > 1.0:  # Slow requests
                logger.warning("Slow API Request", extra=log_data)
            elif duration > 0.5:  # Medium requests
                logger.info("API Performance", extra=log_data)
            else:  # Fast requests
                logger.debug("API Performance", extra=log_data)
            
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    def _log_error(self, error):
        """Log error details."""
        try:
            # Get error data
            error_type = type(error).__name__
            error_message = str(error)
            status_code = getattr(error, 'code', 500)
            
            # Log error
            log_data = {
                'request_id': g.request_id,
                'error_type': error_type,
                'error_message': error_message,
                'status_code': status_code,
                'timestamp': time.time()
            }
            
            logger.error("API Error", extra=log_data, exc_info=True)
            
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize sensitive headers.
        
        Args:
            headers: Headers dictionary
            
        Returns:
            Sanitized headers dictionary
        """
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_query_params(self, query_string: str) -> Dict[str, str]:
        """
        Sanitize sensitive query parameters.
        
        Args:
            query_string: Query string
            
        Returns:
            Sanitized query parameters dictionary
        """
        if not query_string:
            return {}
        
        try:
            from urllib.parse import parse_qs, unquote
            
            params = parse_qs(query_string)
            sanitized = {}
            
            for key, values in params.items():
                if key.lower() in self.sensitive_params:
                    sanitized[key] = '[REDACTED]'
                else:
                    sanitized[key] = [unquote(v) for v in values]
            
            return sanitized
            
        except Exception:
            return {'raw': '[PARSE_ERROR]'}
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        Get request statistics.
        
        Returns:
            Dictionary with request statistics
        """
        # This would typically integrate with a metrics system
        # For now, return basic structure
        return {
            'total_requests': 0,
            'successful_requests': 0,
            'error_requests': 0,
            'average_response_time': 0.0,
            'slow_requests': 0
        }


def init_logging_middleware(app, config: dict):
    """
    Initialize logging middleware from configuration.
    
    Args:
        app: Flask application instance
        config: Configuration dictionary
    """
    logging_config = config.get('logging', {})
    
    middleware = LoggingMiddleware(
        log_requests=logging_config.get('log_requests', True),
        log_responses=logging_config.get('log_responses', True),
        log_errors=logging_config.get('log_errors', True),
        log_performance=logging_config.get('log_performance', True),
        sensitive_headers=logging_config.get('sensitive_headers'),
        sensitive_params=logging_config.get('sensitive_params')
    )
    
    middleware.init_app(app)
    app.logging_middleware = middleware 