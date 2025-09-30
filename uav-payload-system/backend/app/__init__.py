from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_socketio import SocketIO
# JWT removed for simplicity
import os
from config import config

# Initialize extensions
db = SQLAlchemy()
socketio = SocketIO()

def create_app(config_name=None):
    """Application factory pattern - minimal setup"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions with app
    db.init_app(app)
    
    # Configure CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Configure SocketIO
    socketio.init_app(app, cors_allowed_origins=app.config['CORS_ORIGINS'])
    
    # Register single API blueprint (contains all endpoints + socket events)
    from app.api import api
    app.register_blueprint(api)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app