"""
Authentication Middleware

Provides JWT token validation, role-based access control, and user session management
for the AI Automation Bot API.
"""

import time
import jwt
from typing import Optional, Dict, Any, List
from functools import wraps
from flask import request, g, jsonify, current_app
from werkzeug.exceptions import Unauthorized, Forbidden

from ...core.exceptions import AuthenticationError, AuthorizationError
from ...core.logger import get_logger

logger = get_logger(__name__)


class AuthMiddleware:
    """Authentication middleware for API endpoints."""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        """
        Initialize authentication middleware.
        
        Args:
            secret_key: Secret key for JWT token signing
            algorithm: JWT algorithm to use
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_blacklist = set()  # In production, use Redis
        
    def generate_token(self, user_id: str, roles: List[str], 
                      expires_in: int = 3600) -> str:
        """
        Generate JWT token for user.
        
        Args:
            user_id: User identifier
            roles: List of user roles
            expires_in: Token expiration time in seconds
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user_id,
            'roles': roles,
            'exp': time.time() + expires_in,
            'iat': time.time()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token and return payload.
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload dictionary
            
        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            # Check if token is blacklisted
            if token in self.token_blacklist:
                raise AuthenticationError("Token has been revoked")
            
            # Decode and validate token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            if payload.get('exp', 0) < time.time():
                raise AuthenticationError("Token has expired")
                
            return payload
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise AuthenticationError("Invalid token")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise AuthenticationError("Token validation failed")
    
    def revoke_token(self, token: str) -> None:
        """
        Revoke a JWT token by adding it to blacklist.
        
        Args:
            token: JWT token to revoke
        """
        self.token_blacklist.add(token)
        logger.info(f"Token revoked for user")
    
    def require_auth(self, roles: Optional[List[str]] = None):
        """
        Decorator to require authentication for endpoints.
        
        Args:
            roles: Required roles for access (None for any authenticated user)
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Get token from Authorization header
                auth_header = request.headers.get('Authorization')
                if not auth_header:
                    raise AuthenticationError("Authorization header required")
                
                # Extract token from "Bearer <token>" format
                try:
                    token = auth_header.split(' ')[1]
                except IndexError:
                    raise AuthenticationError("Invalid authorization header format")
                
                # Validate token
                payload = self.validate_token(token)
                
                # Store user info in Flask g object
                g.user_id = payload['user_id']
                g.user_roles = payload['roles']
                
                # Check role requirements
                if roles:
                    user_roles = set(payload['roles'])
                    required_roles = set(roles)
                    
                    if not user_roles.intersection(required_roles):
                        logger.warning(f"User {payload['user_id']} lacks required roles: {roles}")
                        raise AuthorizationError("Insufficient permissions")
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def optional_auth(self):
        """
        Decorator for optional authentication - sets user info if token provided.
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                
                if auth_header:
                    try:
                        token = auth_header.split(' ')[1]
                        payload = self.validate_token(token)
                        g.user_id = payload['user_id']
                        g.user_roles = payload['roles']
                    except Exception:
                        # Token invalid, but continue without authentication
                        g.user_id = None
                        g.user_roles = []
                else:
                    g.user_id = None
                    g.user_roles = []
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator


def init_auth_middleware(app, secret_key: str):
    """
    Initialize authentication middleware for Flask app.
    
    Args:
        app: Flask application instance
        secret_key: Secret key for JWT tokens
    """
    auth_middleware = AuthMiddleware(secret_key)
    app.auth_middleware = auth_middleware
    
    # Register error handlers
    @app.errorhandler(AuthenticationError)
    def handle_auth_error(error):
        return jsonify({
            'error': 'Authentication failed',
            'message': str(error),
            'code': 'AUTH_ERROR'
        }), 401
    
    @app.errorhandler(AuthorizationError)
    def handle_authz_error(error):
        return jsonify({
            'error': 'Authorization failed', 
            'message': str(error),
            'code': 'AUTHZ_ERROR'
        }), 403
    
    logger.info("Authentication middleware initialized") 