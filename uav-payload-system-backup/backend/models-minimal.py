from datetime import datetime
from app import db

# ESSENTIAL MODELS ONLY - Core UAV system functionality

class TelemetryData(db.Model):
    """Real-time telemetry data from UAV hardware"""
    __tablename__ = 'telemetry_data'
    
    id = db.Column(db.Integer, primary_key=True)
    uav_id = db.Column(db.Integer, nullable=False)  # Simple reference, no FK constraints for speed
    
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
            'uav_id': self.uav_id,
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
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'serial_number': self.serial_number,
            'status': self.status,
            'created_at': self.created_at.isoformat()
        }

# Remove these complex models if not needed:
# - Mission (complex mission planning)
# - Payload (if just doing sensor data)
# - Waypoint (if no flight planning)  
# - User/Auth (if no authentication needed)
# - SystemLog (if not doing audit logging)