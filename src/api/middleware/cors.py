"""
CORS Middleware

Handles Cross-Origin Resource Sharing (CORS) for the AI Automation Bot API,
allowing controlled access from different origins.
"""

from typing import List, Optional
from flask import request, make_response
from flask_cors import CORS

from ...core.logger import get_logger

logger = get_logger(__name__)


class CORSMiddleware:
    """CORS middleware for handling cross-origin requests."""
    
    def __init__(self, allowed_origins: Optional[List[str]] = None,
                 allowed_methods: Optional[List[str]] = None,
                 allowed_headers: Optional[List[str]] = None,
                 allow_credentials: bool = True,
                 max_age: int = 3600):
        """
        Initialize CORS middleware.
        
        Args:
            allowed_origins: List of allowed origins (None for all)
            allowed_methods: List of allowed HTTP methods
            allowed_headers: List of allowed headers
            allow_credentials: Whether to allow credentials
            max_age: Cache duration for preflight requests
        """
        self.allowed_origins = allowed_origins or ['*']
        self.allowed_methods = allowed_methods or ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        self.allowed_headers = allowed_headers or [
            'Content-Type', 'Authorization', 'X-Requested-With',
            'Accept', 'Origin', 'Access-Control-Request-Method',
            'Access-Control-Request-Headers'
        ]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
    
    def init_app(self, app):
        """
        Initialize CORS for Flask app.
        
        Args:
            app: Flask application instance
        """
        # Configure CORS
        CORS(app, 
             origins=self.allowed_origins,
             methods=self.allowed_methods,
             allow_headers=self.allowed_headers,
             supports_credentials=self.allow_credentials,
             max_age=self.max_age)
        
        # Add custom CORS headers
        @app.after_request
        def after_request(response):
            origin = request.headers.get('Origin')
            
            # Set CORS headers
            if origin and (origin in self.allowed_origins or '*' in self.allowed_origins):
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
                response.headers['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
                response.headers['Access-Control-Allow-Credentials'] = str(self.allow_credentials).lower()
                response.headers['Access-Control-Max-Age'] = str(self.max_age)
            
            return response
        
        # Handle preflight requests
        @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
        @app.route('/<path:path>', methods=['OPTIONS'])
        def handle_preflight(path):
            response = make_response()
            origin = request.headers.get('Origin')
            
            if origin and (origin in self.allowed_origins or '*' in self.allowed_origins):
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
                response.headers['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
                response.headers['Access-Control-Allow-Credentials'] = str(self.allow_credentials).lower()
                response.headers['Access-Control-Max-Age'] = str(self.max_age)
            
            return response
        
        logger.info("CORS middleware initialized")
    
    def is_origin_allowed(self, origin: str) -> bool:
        """
        Check if origin is allowed.
        
        Args:
            origin: Origin to check
            
        Returns:
            True if origin is allowed
        """
        return '*' in self.allowed_origins or origin in self.allowed_origins
    
    def get_allowed_origins(self) -> List[str]:
        """
        Get list of allowed origins.
        
        Returns:
            List of allowed origins
        """
        return self.allowed_origins.copy()
    
    def add_origin(self, origin: str) -> None:
        """
        Add origin to allowed list.
        
        Args:
            origin: Origin to add
        """
        if origin not in self.allowed_origins:
            self.allowed_origins.append(origin)
            logger.info(f"Added origin to CORS: {origin}")
    
    def remove_origin(self, origin: str) -> None:
        """
        Remove origin from allowed list.
        
        Args:
            origin: Origin to remove
        """
        if origin in self.allowed_origins:
            self.allowed_origins.remove(origin)
            logger.info(f"Removed origin from CORS: {origin}")


def init_cors_middleware(app, config: dict):
    """
    Initialize CORS middleware from configuration.
    
    Args:
        app: Flask application instance
        config: Configuration dictionary
    """
    cors_config = config.get('cors', {})
    
    middleware = CORSMiddleware(
        allowed_origins=cors_config.get('allowed_origins'),
        allowed_methods=cors_config.get('allowed_methods'),
        allowed_headers=cors_config.get('allowed_headers'),
        allow_credentials=cors_config.get('allow_credentials', True),
        max_age=cors_config.get('max_age', 3600)
    )
    
    middleware.init_app(app)
    app.cors_middleware = middleware 