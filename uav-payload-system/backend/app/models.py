from datetime import datetime
from app import db
from werkzeug.security import generate_password_hash, check_password_hash

# ESSENTIAL MODELS - Core UAV system with Auth

class User(db.Model):
    """User authentication model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='operator')  # admin, operator, viewer
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class TelemetryData(db.Model):
    """Real-time telemetry data from THE UAV (single UAV system)"""
    __tablename__ = 'telemetry_data'
    
    id = db.Column(db.Integer, primary_key=True)
    # No uav_id needed - only one UAV in the system
    
    # Core telemetry data that hardware will send
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    altitude = db.Column(db.Float, nullable=True)
    battery_level = db.Column(db.Float, nullable=True)  # Percentage 0-100
    temperature = db.Column(db.Float, nullable=True)    # Celsius
    humidity = db.Column(db.Float, nullable=True)       # Percentage 0-100
    status = db.Column(db.String(50), nullable=False, default='normal')  # normal, warning, critical
    
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'battery_level': self.battery_level,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'status': self.status,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class UAV(db.Model):
    """Simplified UAV model - just basics"""
    __tablename__ = 'uavs'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    serial_number = db.Column(db.String(50), unique=True, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='inactive')  # active, inactive, offline
    video_stream_url = db.Column(db.String(255), nullable=True)  # For video streaming
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'serial_number': self.serial_number,
            'status': self.status,
            'video_stream_url': self.video_stream_url,
            'created_at': self.created_at.isoformat()
        }

class SystemLog(db.Model):
    """Audit logging for system events"""
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)  # Can be null for system events
    action = db.Column(db.String(100), nullable=False)  # login, logout, telemetry_received, etc.
    resource = db.Column(db.String(100), nullable=True)  # uav, telemetry, user, etc.
    resource_id = db.Column(db.Integer, nullable=True)  # ID of the affected resource
    details = db.Column(db.Text, nullable=True)  # Additional details in JSON format
    ip_address = db.Column(db.String(45), nullable=True)  # IPv4 or IPv6
    user_agent = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'resource': self.resource,
            'resource_id': self.resource_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat()
        }

# Removed complex models:
# - Mission (complex mission planning)
# - Payload (if just doing sensor data)
# - Waypoint (flight planning)