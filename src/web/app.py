"""
Web Application

Main web application for the AI Automation Bot with Flask and Bootstrap styling.
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash

from .routes import web_bp
from ..core.config import Config
from ..core.logger import get_logger

logger = get_logger(__name__)


def create_web_app(config: Config = None) -> Flask:
    """
    Create and configure Flask web application.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured Flask web application
    """
    # Create Flask app
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
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
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'web.login'
    login_manager.login_message = 'Please log in to access this page.'
    
    @login_manager.user_loader
    def load_user(user_id):
        # This would load user from database
        # For now, return None
        return None
    
    # Register blueprints
    app.register_blueprint(web_bp)
    
    # Register error handlers
    _register_error_handlers(app)
    
    # Register context processors
    _register_context_processors(app)
    
    logger.info("Web application created successfully")
    
    return app


def _register_error_handlers(app: Flask):
    """Register error handlers for web application."""
    
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error):
        return render_template('errors/403.html'), 403


def _register_context_processors(app: Flask):
    """Register context processors for web application."""
    
    @app.context_processor
    def inject_config():
        """Inject configuration into templates."""
        return {
            'app_name': 'AI Automation Bot',
            'app_version': '1.0.0',
            'current_year': 2024
        }
    
    @app.context_processor
    def inject_user():
        """Inject user information into templates."""
        return {
            'current_user': current_user
        }


def run_web_app(app: Flask, host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
    """
    Run the Flask web application.
    
    Args:
        app: Flask web application
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    try:
        logger.info(f"Starting web application on {host}:{port}")
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Web application startup error: {e}")
        raise


if __name__ == '__main__':
    # Create and run web application
    app = create_web_app()
    run_web_app(app, debug=True) 