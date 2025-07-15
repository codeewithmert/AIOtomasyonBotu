"""
API Routes Package

This package contains all API route definitions for the AI Automation Bot,
including authentication, data management, ML operations, automation, and reporting.
"""

from flask import Blueprint
from .auth import auth_bp
from .data import data_bp
from .ml import ml_bp
from .automation import automation_bp
from .reporting import reporting_bp
from .status import status_bp

# Create main API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Register route blueprints
api_bp.register_blueprint(auth_bp, url_prefix='/auth')
api_bp.register_blueprint(data_bp, url_prefix='/data')
api_bp.register_blueprint(ml_bp, url_prefix='/ml')
api_bp.register_blueprint(automation_bp, url_prefix='/automation')
api_bp.register_blueprint(reporting_bp, url_prefix='/reporting')
api_bp.register_blueprint(status_bp, url_prefix='/status')

__all__ = [
    'api_bp',
    'auth_bp',
    'data_bp', 
    'ml_bp',
    'automation_bp',
    'reporting_bp',
    'status_bp'
] 