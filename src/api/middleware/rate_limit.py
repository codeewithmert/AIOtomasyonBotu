"""
Rate Limiting Middleware

Provides rate limiting functionality for the AI Automation Bot API,
preventing abuse and ensuring fair resource usage.
"""

import time
from typing import Dict, Optional, Tuple
from flask import request, g, jsonify
from functools import wraps

from ...core.exceptions import RateLimitExceededError
from ...core.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware:
    """Rate limiting middleware for API endpoints."""
    
    def __init__(self, default_limit: int = 100, default_window: int = 3600,
                 storage_backend: str = 'memory', redis_client=None):
        """
        Initialize rate limiting middleware.
        
        Args:
            default_limit: Default requests per window
            default_window: Default time window in seconds
            storage_backend: Storage backend ('memory' or 'redis')
            redis_client: Redis client for distributed rate limiting
        """
        self.default_limit = default_limit
        self.default_window = default_window
        self.storage_backend = storage_backend
        self.redis_client = redis_client
        
        # In-memory storage for rate limiting data
        self.rate_limit_data = {}
        
        # Rate limit configurations for different endpoints
        self.endpoint_limits = {
            '/api/auth/login': {'limit': 5, 'window': 300},  # 5 attempts per 5 minutes
            '/api/auth/register': {'limit': 3, 'window': 3600},  # 3 attempts per hour
            '/api/ml/train': {'limit': 10, 'window': 3600},  # 10 training jobs per hour
            '/api/data/collect': {'limit': 50, 'window': 300},  # 50 requests per 5 minutes
        }
    
    def init_app(self, app):
        """
        Initialize rate limiting for Flask app.
        
        Args:
            app: Flask application instance
        """
        # Register error handler
        @app.errorhandler(RateLimitExceededError)
        def handle_rate_limit_error(error):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': str(error),
                'retry_after': error.retry_after,
                'limit': error.limit,
                'window': error.window
            }), 429
        
        logger.info("Rate limiting middleware initialized")
    
    def rate_limit(self, limit: Optional[int] = None, window: Optional[int] = None,
                   key_func=None):
        """
        Decorator to apply rate limiting to endpoints.
        
        Args:
            limit: Requests per window (uses default if None)
            window: Time window in seconds (uses default if None)
            key_func: Function to generate rate limit key
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Get rate limit configuration
                endpoint_limit = limit or self.default_limit
                endpoint_window = window or self.default_window
                
                # Generate rate limit key
                if key_func:
                    key = key_func()
                else:
                    key = self._generate_key()
                
                # Check rate limit
                self._check_rate_limit(key, endpoint_limit, endpoint_window)
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def _generate_key(self) -> str:
        """
        Generate rate limit key based on request.
        
        Returns:
            Rate limit key string
        """
        # Use IP address as default key
        key = f"rate_limit:{request.remote_addr}"
        
        # Add user ID if authenticated
        if hasattr(g, 'user_id') and g.user_id:
            key += f":{g.user_id}"
        
        # Add endpoint path
        key += f":{request.path}"
        
        return key
    
    def _check_rate_limit(self, key: str, limit: int, window: int):
        """
        Check if request is within rate limit.
        
        Args:
            key: Rate limit key
            limit: Requests per window
            window: Time window in seconds
            
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        current_time = time.time()
        
        if self.storage_backend == 'redis' and self.redis_client:
            self._check_redis_rate_limit(key, limit, window, current_time)
        else:
            self._check_memory_rate_limit(key, limit, window, current_time)
    
    def _check_memory_rate_limit(self, key: str, limit: int, window: int, current_time: float):
        """
        Check rate limit using in-memory storage.
        
        Args:
            key: Rate limit key
            limit: Requests per window
            window: Time window in seconds
            current_time: Current timestamp
        """
        if key not in self.rate_limit_data:
            self.rate_limit_data[key] = []
        
        # Remove expired timestamps
        self.rate_limit_data[key] = [
            ts for ts in self.rate_limit_data[key]
            if current_time - ts < window
        ]
        
        # Check if limit exceeded
        if len(self.rate_limit_data[key]) >= limit:
            # Calculate retry after time
            oldest_request = min(self.rate_limit_data[key])
            retry_after = int(window - (current_time - oldest_request))
            
            raise RateLimitExceededError(
                f"Rate limit exceeded: {limit} requests per {window} seconds",
                retry_after=retry_after,
                limit=limit,
                window=window
            )
        
        # Add current request
        self.rate_limit_data[key].append(current_time)
    
    def _check_redis_rate_limit(self, key: str, limit: int, window: int, current_time: float):
        """
        Check rate limit using Redis storage.
        
        Args:
            key: Rate limit key
            limit: Requests per window
            window: Time window in seconds
            current_time: Current timestamp
        """
        try:
            # Use Redis sorted set for sliding window rate limiting
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, current_time - window)
            
            # Count current entries
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window)
            
            # Execute pipeline
            results = pipe.execute()
            current_count = results[1]
            
            # Check if limit exceeded
            if current_count >= limit:
                # Get oldest request time
                oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = oldest[0][1]
                    retry_after = int(window - (current_time - oldest_time))
                else:
                    retry_after = window
                
                raise RateLimitExceededError(
                    f"Rate limit exceeded: {limit} requests per {window} seconds",
                    retry_after=retry_after,
                    limit=limit,
                    window=window
                )
                
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fallback to memory rate limiting
            self._check_memory_rate_limit(key, limit, window, current_time)
    
    def get_rate_limit_info(self, key: str) -> Dict:
        """
        Get rate limit information for a key.
        
        Args:
            key: Rate limit key
            
        Returns:
            Dictionary with rate limit information
        """
        current_time = time.time()
        
        if self.storage_backend == 'redis' and self.redis_client:
            return self._get_redis_rate_limit_info(key, current_time)
        else:
            return self._get_memory_rate_limit_info(key, current_time)
    
    def _get_memory_rate_limit_info(self, key: str, current_time: float) -> Dict:
        """
        Get rate limit info from memory storage.
        
        Args:
            key: Rate limit key
            current_time: Current timestamp
            
        Returns:
            Rate limit information dictionary
        """
        if key not in self.rate_limit_data:
            return {
                'current': 0,
                'limit': self.default_limit,
                'window': self.default_window,
                'remaining': self.default_limit,
                'reset_time': current_time + self.default_window
            }
        
        # Get endpoint-specific limits
        endpoint = request.path
        limit = self.endpoint_limits.get(endpoint, {}).get('limit', self.default_limit)
        window = self.endpoint_limits.get(endpoint, {}).get('window', self.default_window)
        
        # Count current requests
        current_requests = [
            ts for ts in self.rate_limit_data[key]
            if current_time - ts < window
        ]
        
        current_count = len(current_requests)
        remaining = max(0, limit - current_count)
        
        # Calculate reset time
        if current_requests:
            reset_time = min(current_requests) + window
        else:
            reset_time = current_time + window
        
        return {
            'current': current_count,
            'limit': limit,
            'window': window,
            'remaining': remaining,
            'reset_time': reset_time
        }
    
    def _get_redis_rate_limit_info(self, key: str, current_time: float) -> Dict:
        """
        Get rate limit info from Redis storage.
        
        Args:
            key: Rate limit key
            current_time: Current timestamp
            
        Returns:
            Rate limit information dictionary
        """
        try:
            # Get endpoint-specific limits
            endpoint = request.path
            limit = self.endpoint_limits.get(endpoint, {}).get('limit', self.default_limit)
            window = self.endpoint_limits.get(endpoint, {}).get('window', self.default_window)
            
            # Remove expired entries and count current
            self.redis_client.zremrangebyscore(key, 0, current_time - window)
            current_count = self.redis_client.zcard(key)
            
            remaining = max(0, limit - current_count)
            
            # Calculate reset time
            oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
            if oldest:
                reset_time = oldest[0][1] + window
            else:
                reset_time = current_time + window
            
            return {
                'current': current_count,
                'limit': limit,
                'window': window,
                'remaining': remaining,
                'reset_time': reset_time
            }
            
        except Exception as e:
            logger.error(f"Redis rate limit info error: {e}")
            return self._get_memory_rate_limit_info(key, current_time)
    
    def reset_rate_limit(self, key: str) -> None:
        """
        Reset rate limit for a key.
        
        Args:
            key: Rate limit key to reset
        """
        if self.storage_backend == 'redis' and self.redis_client:
            self.redis_client.delete(key)
        else:
            self.rate_limit_data.pop(key, None)
        
        logger.info(f"Rate limit reset for key: {key}")


def init_rate_limit_middleware(app, config: dict, redis_client=None):
    """
    Initialize rate limiting middleware from configuration.
    
    Args:
        app: Flask application instance
        config: Configuration dictionary
        redis_client: Redis client for distributed rate limiting
    """
    rate_limit_config = config.get('rate_limit', {})
    
    middleware = RateLimitMiddleware(
        default_limit=rate_limit_config.get('default_limit', 100),
        default_window=rate_limit_config.get('default_window', 3600),
        storage_backend=rate_limit_config.get('storage_backend', 'memory'),
        redis_client=redis_client
    )
    
    middleware.init_app(app)
    app.rate_limit_middleware = middleware 