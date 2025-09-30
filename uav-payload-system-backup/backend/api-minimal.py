"""
Minimal API Routes - Only essential endpoints for UAV telemetry system
"""
from flask import Blueprint, request, jsonify
from app import db, socketio
from models_minimal import TelemetryData, UAV
from datetime import datetime

# Create blueprint
api = Blueprint('api', __name__, url_prefix='/api')

# ============================================================================
# TELEMETRY ENDPOINTS - Core functionality
# ============================================================================

@api.route('/telemetry', methods=['POST'])
def receive_telemetry():
    """Receive telemetry data from hardware"""
    try:
        data = request.get_json()
        
        # Create telemetry record
        telemetry = TelemetryData(
            uav_id=data.get('uav_id', 1),  # Default UAV if not specified
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            altitude=data.get('altitude'),
            battery_level=data.get('battery_level'),
            temperature=data.get('temperature'),
            humidity=data.get('humidity'),
            status=data.get('status', 'normal')
        )
        
        db.session.add(telemetry)
        db.session.commit()
        
        # Broadcast to all connected clients via Socket.IO
        socketio.emit('telemetry_update', telemetry.to_dict())
        
        return jsonify({'status': 'success', 'id': telemetry.id}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api.route('/telemetry/latest', methods=['GET'])
def get_latest_telemetry():
    """Get latest telemetry data"""
    try:
        uav_id = request.args.get('uav_id', 1)
        telemetry = TelemetryData.query.filter_by(uav_id=uav_id).order_by(TelemetryData.timestamp.desc()).first()
        
        if telemetry:
            return jsonify(telemetry.to_dict()), 200
        else:
            return jsonify({'message': 'No telemetry data found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# UAV ENDPOINTS - Basic UAV management
# ============================================================================

@api.route('/uavs', methods=['GET'])
def get_uavs():
    """Get all UAVs"""
    try:
        uavs = UAV.query.all()
        return jsonify([uav.to_dict() for uav in uavs]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/uavs', methods=['POST'])
def create_uav():
    """Create new UAV"""
    try:
        data = request.get_json()
        uav = UAV(
            name=data['name'],
            serial_number=data['serial_number'],
            status=data.get('status', 'inactive')
        )
        db.session.add(uav)
        db.session.commit()
        return jsonify(uav.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# SOCKET.IO EVENTS - Real-time communication
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_telemetry')
def handle_telemetry_request(data):
    """Send latest telemetry to requesting client"""
    try:
        uav_id = data.get('uav_id', 1)
        telemetry = TelemetryData.query.filter_by(uav_id=uav_id).order_by(TelemetryData.timestamp.desc()).first()
        
        if telemetry:
            socketio.emit('telemetry_update', telemetry.to_dict())
    except Exception as e:
        socketio.emit('error', {'message': str(e)})

# ============================================================================
# REMOVED ENDPOINTS (add back if needed):
# ============================================================================
# - Authentication (/auth/login, /auth/logout)
# - Mission management (/missions)
# - Payload management (/payloads)
# - Dashboard stats (/dashboard/stats)
# - Video streaming (/video)
# - File uploads
# - Complex reporting
# - User management