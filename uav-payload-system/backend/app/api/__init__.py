"""
Minimal API - Essential endpoints only
Auth + Telemetry + Video Streaming + Socket.IO
"""
from flask import Blueprint, request, jsonify, Response
# JWT removed for simplicity
from app import db, socketio
from app.models import TelemetryData, UAV, User, SystemLog
from datetime import datetime, timedelta
import time
import threading

# Create blueprint
api = Blueprint('api', __name__, url_prefix='/api')

# ============================================================================
# AUDIT LOGGING HELPER FUNCTIONS
# ============================================================================

def log_event(action, resource=None, resource_id=None, details=None, user_id=None):
    """Helper function to log system events"""
    try:
        # Get request info
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
        user_agent = request.headers.get('User-Agent')
        
        # Create log entry
        log_entry = SystemLog(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        db.session.add(log_entry)
        db.session.commit()
        return log_entry
        
    except Exception as e:
        print(f"Audit logging failed: {e}")
        return None

# Authentication removed for simplicity

# ============================================================================
# TELEMETRY ENDPOINTS
# ============================================================================

@api.route('/telemetry', methods=['POST'])
def receive_telemetry():
    """Receive telemetry data from hardware"""
    try:
        data = request.get_json(force=True, silent=True) or {}

    # Support both flat schema and TAIP nested schema under 'environmental_data'
        env = data.get('environmental_data') or {}

        # Extract environmental readings
        temperature = data.get('temperature')
        humidity = data.get('humidity')

        if temperature is None:
            temperature = env.get('temperature_c')
        if humidity is None:
            humidity = env.get('humidity_rh')

        # Optional fields not present in TAIP packet may be None
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        altitude = data.get('altitude')
        battery_level = data.get('battery_level')
        status = data.get('status', 'normal')

        # Create telemetry record (no UAV ID needed - single UAV system)
        telemetry = TelemetryData(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            battery_level=battery_level,
            temperature=temperature,
            humidity=humidity,
            status=status
        )
        
        db.session.add(telemetry)
        db.session.commit()
        
        # Log telemetry received
        log_event('telemetry_received', 'telemetry', telemetry.id, 
                 f"Telemetry data received from UAV")
        
        # Build broadcast payload: include DB fields plus TAIP extras (no DB migration needed)
        event_payload = telemetry.to_dict()
        # Pass through TAIP gauge pressure if provided
        if 'gauge_pressure_bar' in data:
            event_payload['gauge_pressure_bar'] = data.get('gauge_pressure_bar')
        # Optionally include raw environmental data if provided by TAIP
        if env:
            event_payload['environmental_data'] = {
                'temperature_c': env.get('temperature_c'),
                'pressure_hpa': env.get('pressure_hpa'),
                'humidity_rh': env.get('humidity_rh'),
                'light_lux': env.get('light_lux'),
            }
        
        # Broadcast to all connected clients
        socketio.emit('telemetry_update', event_payload)
        
        return jsonify({'status': 'success', 'id': telemetry.id}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api.route('/telemetry/latest', methods=['GET'])
def get_latest_telemetry():
    """Get latest telemetry data from THE UAV"""
    try:
        # No UAV ID needed - just get the latest record
        telemetry = TelemetryData.query.order_by(TelemetryData.timestamp.desc()).first()
        
        if telemetry:
            return jsonify(telemetry.to_dict()), 200
        else:
            return jsonify({'message': 'No telemetry data found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# UAV management endpoints removed - single UAV system

# ============================================================================
# AUDIT LOG ENDPOINTS
# ============================================================================

@api.route('/audit/events', methods=['GET'])
def get_audit_events():
    """Get audit log events with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        action_filter = request.args.get('action')
        resource_filter = request.args.get('resource')
        user_filter = request.args.get('user_id', type=int)
        
        # Build query
        query = SystemLog.query
        
        if action_filter:
            query = query.filter(SystemLog.action == action_filter)
        if resource_filter:
            query = query.filter(SystemLog.resource == resource_filter)
        if user_filter:
            query = query.filter(SystemLog.user_id == user_filter)
        
        # Order by timestamp descending (most recent first)
        query = query.order_by(SystemLog.timestamp.desc())
        
        # Paginate
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        events = pagination.items
        
        return jsonify({
            'events': [event.to_dict() for event in events],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_prev': pagination.has_prev,
                'has_next': pagination.has_next
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/audit/stats', methods=['GET'])
def get_audit_stats():
    """Get audit log statistics"""
    try:
        # Get counts by action type
        from sqlalchemy import func
        
        action_stats = db.session.query(
            SystemLog.action,
            func.count(SystemLog.id).label('count')
        ).group_by(SystemLog.action).all()
        
        # Get recent activity (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_count = SystemLog.query.filter(SystemLog.timestamp >= recent_cutoff).count()
        
        # Get total events
        total_events = SystemLog.query.count()
        
        return jsonify({
            'total_events': total_events,
            'recent_events_24h': recent_count,
            'action_breakdown': [{'action': stat.action, 'count': stat.count} for stat in action_stats]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# VIDEO STREAMING ENDPOINTS
# ============================================================================

# Global latest-frame buffer pushed by TAIP client
_latest_frame_bytes = None
_frame_lock = threading.Lock()

@api.route('/frame', methods=['POST'])
def receive_frame():
    """Receive a JPEG frame from TAIP and store as latest.

    Expected: Content-Type: image/jpeg and raw JPEG bytes in body.
    """
    global _latest_frame_bytes
    try:
        # Basic content-type check (non-fatal)
        # Some clients may omit; we still accept raw bytes
        # ct = request.headers.get('Content-Type', '')
        frame_bytes = request.get_data()
        if not frame_bytes:
            return jsonify({
                'status': 'error',
                'message': 'empty frame payload'
            }), 400

        # Optionally validate JPEG header (starts with 0xFFD8)
        if not (len(frame_bytes) >= 2 and frame_bytes[0] == 0xFF and frame_bytes[1] == 0xD8):
            # Still accept but log a warning
            print('Warning: received frame does not appear to be JPEG start (FFD8)')

        with _frame_lock:
            _latest_frame_bytes = frame_bytes

        # Optionally broadcast an event to notify clients a frame arrived (not needed for MJPEG)
        # socketio.emit('video_frame_received', {'ts': time.time()})

        return jsonify({'status': 'ok'}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def generate_frames_from_buffer():
    """Generate MJPEG bytes from the latest-frame buffer updated by TAIP."""
    global _latest_frame_bytes
    try:
        last_sent = None
        while True:
            with _frame_lock:
                current = _latest_frame_bytes
            if current is not None and current is not last_sent:
                # Send only when a new frame is available (reduces bandwidth)
                last_sent = current
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current + b'\r\n')
            else:
                # No new frame yet; sleep briefly to avoid busy loop
                time.sleep(0.03)
    except GeneratorExit:
        # Client disconnected
        pass
    except Exception as e:
        print(f"Video streaming error: {e}")

@api.route('/video/stream')
def video_stream():
    """Live video stream from THE UAV (single UAV system)"""
    # Stream frames pushed from TAIP via /api/frame
    return Response(generate_frames_from_buffer(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================================
# SOCKET.IO EVENTS
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
    """Send latest telemetry to requesting client (single UAV system)"""
    try:
        # No UAV ID needed - just get latest telemetry
        telemetry = TelemetryData.query.order_by(TelemetryData.timestamp.desc()).first()
        
        if telemetry:
            socketio.emit('telemetry_update', telemetry.to_dict())
    except Exception as e:
        socketio.emit('error', {'message': str(e)})

# ============================================================================
# REMOVED ENDPOINTS:
# - Mission management (/missions)
# - Payload management (/payloads)
# - Dashboard stats (/dashboard/stats) 
# - Complex reporting
# - User management (admin only)
# - Hardware configuration
# ============================================================================