"""
Web Interface Package

This package contains web interface components for the AI Automation Bot,
including Flask routes, templates, and static assets with Bootstrap styling.
"""

from .app import create_web_app
from .routes import web_bp
from .templates import render_template

__all__ = [
    'create_web_app',
    'web_bp',
    'render_template'
] 