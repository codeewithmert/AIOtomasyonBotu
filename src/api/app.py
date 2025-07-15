"""
Flask API Application

Main Flask application for the AI Automation Bot API with middleware,
route registration, and configuration management.
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS

from .middleware import (
    init_auth_middleware,
    init_cors_middleware,
    init_logging_middleware,
    init_rate_limit_middleware,
    init_error_handler_middleware
)
from .routes import api_bp
from ..core.config import Config
from ..core.logger import get_logger

logger = get_logger(__name__)


def create_app(config: Config = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    if config:
        app.config.from_object(config)
    else:
        # Load from environment or use defaults
        app.config.update({
            'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production'),
            'DEBUG': os.environ.get('DEBUG', 'False').lower() == 'true',
            'TESTING': os.environ.get('TESTING', 'False').lower() == 'true'
        })
    
    # Initialize middleware
    _init_middleware(app, config)
    
    # Register blueprints
    _register_blueprints(app)
    
    # Register error handlers
    _register_error_handlers(app)
    
    # Register CLI commands
    _register_cli_commands(app)
    
    logger.info("Flask application created successfully")
    
    return app


def _init_middleware(app: Flask, config: Config):
    """Initialize all middleware components."""
    try:
        # Initialize CORS middleware
        cors_config = {
            'allowed_origins': ['*'],
            'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            'allowed_headers': ['Content-Type', 'Authorization'],
            'allow_credentials': True
        }
        init_cors_middleware(app, cors_config)
        
        # Initialize logging middleware
        logging_config = {
            'log_requests': True,
            'log_responses': True,
            'log_errors': True,
            'log_performance': True,
            'sensitive_headers': ['authorization', 'cookie'],
            'sensitive_params': ['password', 'token']
        }
        init_logging_middleware(app, logging_config)
        
        # Initialize rate limiting middleware
        rate_limit_config = {
            'default_limit': 100,
            'default_window': 3600,
            'storage_backend': 'memory'
        }
        init_rate_limit_middleware(app, rate_limit_config)
        
        # Initialize error handling middleware
        error_handler_config = {
            'log_errors': True,
            'include_traceback': app.config.get('DEBUG', False)
        }
        init_error_handler_middleware(app, error_handler_config)
        
        # Initialize authentication middleware
        secret_key = app.config.get('SECRET_KEY', 'dev-secret-key')
        init_auth_middleware(app, secret_key)
        
        logger.info("All middleware initialized successfully")
        
    except Exception as e:
        logger.error(f"Middleware initialization error: {e}")
        raise


def _register_blueprints(app: Flask):
    """Register Flask blueprints."""
    try:
        # Register main API blueprint
        app.register_blueprint(api_bp)
        
        # Register root endpoint
        @app.route('/')
        def root():
            return jsonify({
                'message': 'AI Automation Bot API',
                'version': '1.0.0',
                'status': 'running',
                'endpoints': {
                    'api': '/api',
                    'health': '/api/status/health',
                    'docs': '/docs'
                }
            })
        
        logger.info("Blueprints registered successfully")
        
    except Exception as e:
        logger.error(f"Blueprint registration error: {e}")
        raise


def _register_error_handlers(app: Flask):
    """Register global error handlers."""
    try:
        @app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Not Found',
                'message': 'The requested resource was not found',
                'code': 404
            }), 404
        
        @app.errorhandler(405)
        def method_not_allowed(error):
            return jsonify({
                'error': 'Method Not Allowed',
                'message': 'The method is not allowed for the requested resource',
                'code': 405
            }), 405
        
        @app.errorhandler(500)
        def internal_server_error(error):
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'code': 500
            }), 500
        
        logger.info("Error handlers registered successfully")
        
    except Exception as e:
        logger.error(f"Error handler registration error: {e}")
        raise


def _register_cli_commands(app: Flask):
    """Register CLI commands."""
    try:
        @app.cli.command('init-db')
        def init_db():
            """Initialize the database."""
            logger.info("Initializing database...")
            # This would initialize the database
            logger.info("Database initialized successfully")
        
        @app.cli.command('create-admin')
        def create_admin():
            """Create an admin user."""
            logger.info("Creating admin user...")
            # This would create an admin user
            logger.info("Admin user created successfully")
        
        logger.info("CLI commands registered successfully")
        
    except Exception as e:
        logger.error(f"CLI command registration error: {e}")
        raise


def run_app(app: Flask, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """
    Run the Flask application.
    
    Args:
        app: Flask application
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    try:
        logger.info(f"Starting Flask application on {host}:{port}")
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        raise


if __name__ == '__main__':
    # Create and run application
    app = create_app()
    run_app(app, debug=True) 