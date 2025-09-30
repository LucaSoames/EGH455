from flask import Blueprint, request
from flask_socketio import emit, join_room, leave_room
from app import socketio
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity

websocket_bp = Blueprint('websocket', __name__)

@socketio.on('connect')
def handle_connect(auth):
    """Handle client connection"""
    try:
        # Verify JWT token for WebSocket connections
        if auth and 'token' in auth:
            # You would verify the token here in a production app
            print(f"Client connected: {request.sid}")
            emit('connected', {'status': 'Connected to UAV Payload System'})
        else:
            print("Client connection rejected: No valid token")
            return False
    except Exception as e:
        print(f"Connection error: {str(e)}")
        return False

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('join_telemetry')
def handle_join_telemetry():
    """Join telemetry updates room"""
    try:
        join_room('telemetry')
        emit('status', {'message': 'Joined telemetry updates'})
        print(f"Client {request.sid} joined telemetry room")
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('leave_telemetry')
def handle_leave_telemetry():
    """Leave telemetry updates room"""
    try:
        leave_room('telemetry')
        emit('status', {'message': 'Left telemetry updates'})
        print(f"Client {request.sid} left telemetry room")
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('join_mission')
def handle_join_mission(data):
    """Join mission-specific updates room"""
    try:
        mission_id = data.get('mission_id')
        if mission_id:
            room = f'mission_{mission_id}'
            join_room(room)
            emit('status', {'message': f'Joined mission {mission_id} updates'})
            print(f"Client {request.sid} joined mission {mission_id} room")
        else:
            emit('error', {'message': 'Mission ID required'})
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('leave_mission')
def handle_leave_mission(data):
    """Leave mission-specific updates room"""
    try:
        mission_id = data.get('mission_id')
        if mission_id:
            room = f'mission_{mission_id}'
            leave_room(room)
            emit('status', {'message': f'Left mission {mission_id} updates'})
            print(f"Client {request.sid} left mission {mission_id} room")
        else:
            emit('error', {'message': 'Mission ID required'})
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('join_uav')
def handle_join_uav(data):
    """Join UAV-specific updates room"""
    try:
        uav_id = data.get('uav_id')
        if uav_id:
            room = f'uav_{uav_id}'
            join_room(room)
            emit('status', {'message': f'Joined UAV {uav_id} updates'})
            print(f"Client {request.sid} joined UAV {uav_id} room")
        else:
            emit('error', {'message': 'UAV ID required'})
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('leave_uav')
def handle_leave_uav(data):
    """Leave UAV-specific updates room"""
    try:
        uav_id = data.get('uav_id')
        if uav_id:
            room = f'uav_{uav_id}'
            leave_room(room)
            emit('status', {'message': f'Left UAV {uav_id} updates'})
            print(f"Client {request.sid} left UAV {uav_id} room")
        else:
            emit('error', {'message': 'UAV ID required'})
    except Exception as e:
        emit('error', {'message': str(e)})

# Utility functions for emitting updates
def emit_telemetry_update(telemetry_data):
    """Emit telemetry update to subscribed clients"""
    socketio.emit('telemetry_update', telemetry_data, room='telemetry')
    
    # Also emit to UAV-specific room
    if 'uav_id' in telemetry_data:
        uav_room = f'uav_{telemetry_data["uav_id"]}'
        socketio.emit('uav_telemetry', telemetry_data, room=uav_room)

def emit_mission_update(mission_data):
    """Emit mission update to subscribed clients"""
    mission_id = mission_data.get('id')
    if mission_id:
        room = f'mission_{mission_id}'
        socketio.emit('mission_update', mission_data, room=room)
    
    # Also emit to general mission updates
    socketio.emit('mission_status_change', mission_data, broadcast=True)

def emit_uav_status_update(uav_data):
    """Emit UAV status update to subscribed clients"""
    uav_id = uav_data.get('id')
    if uav_id:
        room = f'uav_{uav_id}'
        socketio.emit('uav_status_update', uav_data, room=room)
    
    # Also emit to general UAV updates
    socketio.emit('uav_status_change', uav_data, broadcast=True)

def emit_system_alert(alert_data):
    """Emit system alert to all connected clients"""
    socketio.emit('system_alert', alert_data, broadcast=True)

# Hardware-specific WebSocket handlers and emitters
@socketio.on('join_hardware')
def handle_join_hardware(data):
    """Join hardware-specific updates room"""
    try:
        uav_id = data.get('uav_id')
        if uav_id:
            room = f'hardware_{uav_id}'
            join_room(room)
            emit('status', {'message': f'Joined hardware updates for UAV {uav_id}'})
            print(f"Client {request.sid} joined hardware room for UAV {uav_id}")
        else:
            emit('error', {'message': 'UAV ID required for hardware updates'})
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('leave_hardware')  
def handle_leave_hardware(data):
    """Leave hardware-specific updates room"""
    try:
        uav_id = data.get('uav_id')
        if uav_id:
            room = f'hardware_{uav_id}'
            leave_room(room)
            emit('status', {'message': f'Left hardware updates for UAV {uav_id}'})
            print(f"Client {request.sid} left hardware room for UAV {uav_id}")
        else:
            emit('error', {'message': 'UAV ID required'})
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('join_environmental')
def handle_join_environmental():
    """Join environmental sensor updates room"""
    try:
        join_room('environmental')
        emit('status', {'message': 'Joined environmental sensor updates'})
        print(f"Client {request.sid} joined environmental room")
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('leave_environmental')
def handle_leave_environmental():
    """Leave environmental sensor updates room"""
    try:
        leave_room('environmental')
        emit('status', {'message': 'Left environmental sensor updates'})
        print(f"Client {request.sid} left environmental room")
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('join_drilling')
def handle_join_drilling():
    """Join drilling system updates room"""
    try:
        join_room('drilling')
        emit('status', {'message': 'Joined drilling system updates'})
        print(f"Client {request.sid} joined drilling room")
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('leave_drilling')
def handle_leave_drilling():
    """Leave drilling system updates room"""
    try:
        leave_room('drilling')
        emit('status', {'message': 'Left drilling system updates'})
        print(f"Client {request.sid} left drilling room")
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('join_video_stream')
def handle_join_video_stream(data):
    """Join video stream updates room"""
    try:
        uav_id = data.get('uav_id')
        if uav_id:
            room = f'video_{uav_id}'
            join_room(room)
            emit('status', {'message': f'Joined video stream updates for UAV {uav_id}'})
            print(f"Client {request.sid} joined video room for UAV {uav_id}")
        else:
            emit('error', {'message': 'UAV ID required for video stream updates'})
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('leave_video_stream')
def handle_leave_video_stream(data):
    """Leave video stream updates room"""
    try:
        uav_id = data.get('uav_id')
        if uav_id:
            room = f'video_{uav_id}'
            leave_room(room)
            emit('status', {'message': f'Left video stream updates for UAV {uav_id}'})
            print(f"Client {request.sid} left video room for UAV {uav_id}")
        else:
            emit('error', {'message': 'UAV ID required'})
    except Exception as e:
        emit('error', {'message': str(e)})

# Hardware-specific emitter functions
def emit_hardware_status_update(uav_id, status_data):
    """Emit hardware status update to subscribed clients"""
    room = f'hardware_{uav_id}'
    socketio.emit('hardware_status_update', {
        'uav_id': uav_id,
        'status': status_data,
        'timestamp': status_data.get('timestamp')
    }, room=room)
    
    # Also emit to general hardware room if exists
    socketio.emit('hardware_status_change', {
        'uav_id': uav_id,
        'status': status_data
    }, broadcast=True)

def emit_environmental_update(sensor_data):
    """Emit environmental sensor update to subscribed clients"""
    socketio.emit('environmental_update', sensor_data, room='environmental')
    
    # Also emit to UAV-specific room if UAV ID present
    if 'uav_id' in sensor_data:
        uav_room = f'uav_{sensor_data["uav_id"]}'
        socketio.emit('environmental_data', sensor_data, room=uav_room)

def emit_air_quality_update(air_quality_data):
    """Emit air quality update to subscribed clients"""
    socketio.emit('air_quality_update', air_quality_data, room='environmental')
    
    # Emit alerts for dangerous air quality levels
    if air_quality_data.get('air_quality', {}).get('aqi', 0) > 150:
        emit_system_alert({
            'type': 'air_quality_warning',
            'severity': 'high',
            'message': f'High AQI detected: {air_quality_data.get("air_quality", {}).get("aqi", 0)}',
            'uav_id': air_quality_data.get('uav_id'),
            'timestamp': air_quality_data.get('timestamp')
        })

def emit_drilling_update(drilling_data):
    """Emit drilling system update to subscribed clients"""
    socketio.emit('drilling_update', drilling_data, room='drilling')
    
    # Also emit to UAV-specific room
    if 'uav_id' in drilling_data:
        uav_room = f'uav_{drilling_data["uav_id"]}'
        socketio.emit('drilling_status', drilling_data, room=uav_room)

def emit_target_detection_update(detection_data):
    """Emit target detection update to subscribed clients"""
    # Emit to general telemetry room
    socketio.emit('target_detection_update', detection_data, room='telemetry')
    
    # Also emit to UAV-specific room
    if 'uav_id' in detection_data:
        uav_room = f'uav_{detection_data["uav_id"]}'
        socketio.emit('target_detected', detection_data, room=uav_room)

def emit_video_stream_update(uav_id, stream_data):
    """Emit video stream status update to subscribed clients"""
    room = f'video_{uav_id}'
    socketio.emit('video_stream_update', {
        'uav_id': uav_id,
        'stream_status': stream_data,
        'timestamp': stream_data.get('timestamp')
    }, room=room)
