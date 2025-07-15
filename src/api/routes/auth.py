"""
Authentication Routes

Provides authentication endpoints for user login, logout, registration,
token management, and user profile operations.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any
from flask import Blueprint, request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash

from ...core.exceptions import AuthenticationError, ValidationError
from ...core.logger import get_logger
from ...core.utils import validate_email, validate_password_strength

logger = get_logger(__name__)

# Create authentication blueprint
auth_bp = Blueprint('auth', __name__)

# In-memory user storage (replace with database in production)
users_db = {}
tokens_db = {}


@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user.
    
    Request body:
        - username: User's username
        - email: User's email address
        - password: User's password
        - full_name: User's full name (optional)
    
    Returns:
        - 201: User created successfully
        - 400: Validation error
        - 409: User already exists
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data:
            raise ValidationError("Request body is required")
        
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name', '')
        
        # Validate input
        if not username or not email or not password:
            raise ValidationError("Username, email, and password are required")
        
        if not validate_email(email):
            raise ValidationError("Invalid email format")
        
        if not validate_password_strength(password):
            raise ValidationError("Password must be at least 8 characters with uppercase, lowercase, number, and special character")
        
        # Check if user already exists
        if username in users_db or any(u['email'] == email for u in users_db.values()):
            raise ValidationError("Username or email already exists")
        
        # Create user
        user_id = secrets.token_urlsafe(16)
        user = {
            'id': user_id,
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(password),
            'full_name': full_name,
            'roles': ['user'],
            'created_at': datetime.utcnow(),
            'last_login': None,
            'is_active': True
        }
        
        users_db[user_id] = user
        
        logger.info(f"User registered: {username}")
        
        return jsonify({
            'message': 'User registered successfully',
            'user_id': user_id,
            'username': username
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Registration failed'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Authenticate user and return access token.
    
    Request body:
        - username: User's username or email
        - password: User's password
    
    Returns:
        - 200: Login successful
        - 400: Validation error
        - 401: Authentication failed
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data:
            raise ValidationError("Request body is required")
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            raise ValidationError("Username and password are required")
        
        # Find user by username or email
        user = None
        for u in users_db.values():
            if u['username'] == username or u['email'] == username:
                user = u
                break
        
        if not user:
            raise AuthenticationError("Invalid username or password")
        
        if not user['is_active']:
            raise AuthenticationError("Account is deactivated")
        
        # Verify password
        if not check_password_hash(user['password_hash'], password):
            raise AuthenticationError("Invalid username or password")
        
        # Update last login
        user['last_login'] = datetime.utcnow()
        
        # Generate access token
        token = request.app.auth_middleware.generate_token(
            user_id=user['id'],
            roles=user['roles'],
            expires_in=3600  # 1 hour
        )
        
        # Store token
        tokens_db[token] = {
            'user_id': user['id'],
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(hours=1)
        }
        
        logger.info(f"User logged in: {user['username']}")
        
        return jsonify({
            'message': 'Login successful',
            'access_token': token,
            'token_type': 'Bearer',
            'expires_in': 3600,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'full_name': user['full_name'],
                'roles': user['roles']
            }
        }), 200
        
    except (ValidationError, AuthenticationError) as e:
        return jsonify({'error': type(e).__name__, 'message': str(e)}), 400 if isinstance(e, ValidationError) else 401
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Login failed'}), 500


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """
    Logout user and invalidate token.
    
    Headers:
        - Authorization: Bearer <token>
    
    Returns:
        - 200: Logout successful
        - 401: Authentication required
    """
    try:
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise AuthenticationError("Authorization header required")
        
        token = auth_header.split(' ')[1]
        
        # Revoke token
        request.app.auth_middleware.revoke_token(token)
        
        # Remove from tokens database
        tokens_db.pop(token, None)
        
        logger.info(f"User logged out: {g.user_id}")
        
        return jsonify({'message': 'Logout successful'}), 200
        
    except AuthenticationError as e:
        return jsonify({'error': 'Authentication Error', 'message': str(e)}), 401
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Logout failed'}), 500


@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """
    Refresh access token.
    
    Headers:
        - Authorization: Bearer <token>
    
    Returns:
        - 200: Token refreshed successfully
        - 401: Authentication required
    """
    try:
        # Get current token
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise AuthenticationError("Authorization header required")
        
        current_token = auth_header.split(' ')[1]
        
        # Validate current token
        payload = request.app.auth_middleware.validate_token(current_token)
        
        # Get user
        user = users_db.get(payload['user_id'])
        if not user:
            raise AuthenticationError("User not found")
        
        # Generate new token
        new_token = request.app.auth_middleware.generate_token(
            user_id=user['id'],
            roles=user['roles'],
            expires_in=3600
        )
        
        # Store new token and revoke old one
        tokens_db[new_token] = {
            'user_id': user['id'],
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(hours=1)
        }
        
        request.app.auth_middleware.revoke_token(current_token)
        tokens_db.pop(current_token, None)
        
        logger.info(f"Token refreshed for user: {user['username']}")
        
        return jsonify({
            'message': 'Token refreshed successfully',
            'access_token': new_token,
            'token_type': 'Bearer',
            'expires_in': 3600
        }), 200
        
    except AuthenticationError as e:
        return jsonify({'error': 'Authentication Error', 'message': str(e)}), 401
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Token refresh failed'}), 500


@auth_bp.route('/profile', methods=['GET'])
def get_profile():
    """
    Get current user profile.
    
    Headers:
        - Authorization: Bearer <token>
    
    Returns:
        - 200: User profile
        - 401: Authentication required
    """
    try:
        user = users_db.get(g.user_id)
        if not user:
            raise AuthenticationError("User not found")
        
        return jsonify({
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'full_name': user['full_name'],
                'roles': user['roles'],
                'created_at': user['created_at'].isoformat(),
                'last_login': user['last_login'].isoformat() if user['last_login'] else None,
                'is_active': user['is_active']
            }
        }), 200
        
    except AuthenticationError as e:
        return jsonify({'error': 'Authentication Error', 'message': str(e)}), 401
    except Exception as e:
        logger.error(f"Get profile error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to get profile'}), 500


@auth_bp.route('/profile', methods=['PUT'])
def update_profile():
    """
    Update current user profile.
    
    Headers:
        - Authorization: Bearer <token>
    
    Request body:
        - full_name: New full name (optional)
        - email: New email (optional)
    
    Returns:
        - 200: Profile updated successfully
        - 400: Validation error
        - 401: Authentication required
    """
    try:
        data = request.get_json()
        
        user = users_db.get(g.user_id)
        if not user:
            raise AuthenticationError("User not found")
        
        # Update fields
        if 'full_name' in data:
            user['full_name'] = data['full_name']
        
        if 'email' in data:
            email = data['email']
            if not validate_email(email):
                raise ValidationError("Invalid email format")
            
            # Check if email is already taken
            for u in users_db.values():
                if u['id'] != g.user_id and u['email'] == email:
                    raise ValidationError("Email already taken")
            
            user['email'] = email
        
        logger.info(f"Profile updated for user: {user['username']}")
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'full_name': user['full_name'],
                'roles': user['roles']
            }
        }), 200
        
    except (ValidationError, AuthenticationError) as e:
        return jsonify({'error': type(e).__name__, 'message': str(e)}), 400 if isinstance(e, ValidationError) else 401
    except Exception as e:
        logger.error(f"Update profile error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to update profile'}), 500


@auth_bp.route('/change-password', methods=['POST'])
def change_password():
    """
    Change user password.
    
    Headers:
        - Authorization: Bearer <token>
    
    Request body:
        - current_password: Current password
        - new_password: New password
    
    Returns:
        - 200: Password changed successfully
        - 400: Validation error
        - 401: Authentication required
    """
    try:
        data = request.get_json()
        
        if not data:
            raise ValidationError("Request body is required")
        
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            raise ValidationError("Current password and new password are required")
        
        user = users_db.get(g.user_id)
        if not user:
            raise AuthenticationError("User not found")
        
        # Verify current password
        if not check_password_hash(user['password_hash'], current_password):
            raise AuthenticationError("Current password is incorrect")
        
        # Validate new password
        if not validate_password_strength(new_password):
            raise ValidationError("New password must be at least 8 characters with uppercase, lowercase, number, and special character")
        
        # Update password
        user['password_hash'] = generate_password_hash(new_password)
        
        logger.info(f"Password changed for user: {user['username']}")
        
        return jsonify({'message': 'Password changed successfully'}), 200
        
    except (ValidationError, AuthenticationError) as e:
        return jsonify({'error': type(e).__name__, 'message': str(e)}), 400 if isinstance(e, ValidationError) else 401
    except Exception as e:
        logger.error(f"Change password error: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': 'Failed to change password'}), 500 