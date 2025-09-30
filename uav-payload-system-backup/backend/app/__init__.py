from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_jwt_extended import JWTManager
from flask_marshmallow import Marshmallow
import os
from config import config

# Initialize extensions
db = SQLAlchemy()
socketio = SocketIO()
jwt = JWTManager()
ma = Marshmallow()

def create_app(config_name=None):
    """Application factory pattern"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions with app
    db.init_app(app)
    ma.init_app(app)
    jwt.init_app(app)
    
    # Configure JWT identity loaders to handle string/int conversion
    @jwt.user_identity_loader
    def user_identity_lookup(user):
        return str(user)  # Always store as string in JWT
    
    @jwt.user_lookup_loader
    def user_lookup_callback(_jwt_header, jwt_data):
        identity = jwt_data["sub"]
        from app.models import User
        return User.query.get(int(identity))  # Convert back to int for database lookup
    
    # Configure CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Configure SocketIO
    socketio.init_app(app, cors_allowed_origins=app.config['CORS_ORIGINS'])
    
    # Register blueprints
    from app.api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    from app.websocket import websocket_bp
    app.register_blueprint(websocket_bp)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app
