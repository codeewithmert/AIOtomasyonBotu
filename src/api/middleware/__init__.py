"""
API Middleware Package

This package contains middleware components for the AI Automation Bot API,
including authentication, rate limiting, logging, CORS, and error handling.
"""

from .auth import AuthMiddleware
from .cors import CORSMiddleware
from .logging import LoggingMiddleware
from .rate_limit import RateLimitMiddleware
from .error_handler import ErrorHandlerMiddleware

__all__ = [
    'AuthMiddleware',
    'CORSMiddleware', 
    'LoggingMiddleware',
    'RateLimitMiddleware',
    'ErrorHandlerMiddleware'
] 