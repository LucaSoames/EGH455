from flask import request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import check_password_hash, generate_password_hash
from app.api import api_bp
from app.models import User, SystemLog, db
from app.schemas import UserSchema
from datetime import datetime, timedelta

user_schema = UserSchema()
users_schema = UserSchema(many=True)

@api_bp.route('/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        json_data = request.get_json()
        
        if not json_data or not json_data.get('username') or not json_data.get('password'):
            return jsonify({
                'success': False,
                'error': 'Username and password are required'
            }), 400
        
        username = json_data['username']
        password = json_data['password']
        
        # Find user
        user = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        if not user.is_active:
            return jsonify({
                'success': False,
                'error': 'Account is deactivated'
            }), 401
        
        # Create access token
        access_token = create_access_token(
            identity=user.id,
            expires_delta=timedelta(hours=8)
        )
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Log the login
        log_entry = SystemLog(
            user_id=user.id,
            action='login',
            resource_type='auth',
            ip_address=request.remote_addr
        )
        db.session.add(log_entry)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'access_token': access_token,
            'user': user_schema.dump(user)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/auth/register', methods=['POST'])
@jwt_required()
def register():
    """Register a new user (admin only)"""
    try:
        # Check if current user is admin
        current_user_id = int(get_jwt_identity())
        current_user = User.query.get(current_user_id)
        
        if not current_user or current_user.role != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin privileges required'
            }), 403
        
        json_data = request.get_json()
        
        # Validate input data
        errors = user_schema.validate(json_data)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Check if username or email already exists
        existing_user = User.query.filter(
            (User.username == json_data['username']) | 
            (User.email == json_data['email'])
        ).first()
        
        if existing_user:
            return jsonify({
                'success': False,
                'error': 'Username or email already exists'
            }), 400
        
        # Create new user
        user_data = json_data.copy()
        password = user_data.pop('password')
        user_data['password_hash'] = generate_password_hash(password)
        
        user = User(**user_data)
        db.session.add(user)
        db.session.commit()
        
        # Log the registration
        log_entry = SystemLog(
            user_id=current_user_id,
            action='create_user',
            resource_type='user',
            resource_id=user.id,
            details=f'Created user: {user.username}',
            ip_address=request.remote_addr
        )
        db.session.add(log_entry)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': user_schema.dump(user),
            'message': 'User created successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/auth/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user profile"""
    try:
        current_user_id = int(get_jwt_identity())
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': user_schema.dump(user)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/auth/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update current user profile"""
    try:
        current_user_id = int(get_jwt_identity())
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
        
        json_data = request.get_json()
        
        # Remove fields that shouldn't be updated via profile
        restricted_fields = ['password_hash', 'role', 'is_active', 'id']
        for field in restricted_fields:
            json_data.pop(field, None)
        
        # Validate input data
        errors = user_schema.validate(json_data, partial=True)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Update user fields
        for field, value in json_data.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': user_schema.dump(user),
            'message': 'Profile updated successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/auth/change-password', methods=['PUT'])
@jwt_required()
def change_password():
    """Change user password"""
    try:
        current_user_id = int(get_jwt_identity())
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
        
        json_data = request.get_json()
        
        if not json_data.get('current_password') or not json_data.get('new_password'):
            return jsonify({
                'success': False,
                'error': 'Current password and new password are required'
            }), 400
        
        # Verify current password
        if not check_password_hash(user.password_hash, json_data['current_password']):
            return jsonify({
                'success': False,
                'error': 'Current password is incorrect'
            }), 400
        
        # Validate new password
        if len(json_data['new_password']) < 6:
            return jsonify({
                'success': False,
                'error': 'New password must be at least 6 characters long'
            }), 400
        
        # Update password
        user.password_hash = generate_password_hash(json_data['new_password'])
        db.session.commit()
        
        # Log the password change
        log_entry = SystemLog(
            user_id=user.id,
            action='change_password',
            resource_type='auth',
            ip_address=request.remote_addr
        )
        db.session.add(log_entry)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    """Get all users (admin only)"""
    try:
        current_user_id = int(get_jwt_identity())
        current_user = User.query.get(current_user_id)
        
        if not current_user or current_user.role != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin privileges required'
            }), 403
        
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        
        users = User.query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'data': users_schema.dump(users.items),
            'pagination': {
                'page': users.page,
                'pages': users.pages,
                'per_page': users.per_page,
                'total': users.total
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/users/<int:user_id>/status', methods=['PUT'])
@jwt_required()
def update_user_status(user_id):
    """Update user status (admin only)"""
    try:
        current_user_id = int(get_jwt_identity())
        current_user = User.query.get(current_user_id)
        
        if not current_user or current_user.role != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin privileges required'
            }), 403
        
        user = User.query.get_or_404(user_id)
        json_data = request.get_json()
        
        if 'is_active' not in json_data:
            return jsonify({
                'success': False,
                'error': 'is_active field is required'
            }), 400
        
        user.is_active = json_data['is_active']
        db.session.commit()
        
        # Log the status change
        log_entry = SystemLog(
            user_id=current_user_id,
            action='update_user_status',
            resource_type='user',
            resource_id=user_id,
            details=f'Set user {user.username} active status to {user.is_active}',
            ip_address=request.remote_addr
        )
        db.session.add(log_entry)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': user_schema.dump(user),
            'message': 'User status updated successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
