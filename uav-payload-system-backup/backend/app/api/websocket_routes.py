from flask import request, jsonify
from app.api import api_bp
from app.websocket import (
    emit_environmental_update,
    emit_air_quality_update, 
    emit_hardware_status_update,
    emit_drilling_update,
    emit_target_detection_update,
    emit_video_stream_update
)
import logging

logger = logging.getLogger(__name__)

@api_bp.route('/websocket/emit/environmental', methods=['POST'])
def emit_environmental_websocket():
    """Emit environmental sensor data via WebSocket"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['uav_id', 'timestamp', 'environmental']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Emit the environmental update
        emit_environmental_update(data)
        
        return jsonify({
            'success': True,
            'message': 'Environmental update emitted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to emit environmental update: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/websocket/emit/air-quality', methods=['POST'])
def emit_air_quality_websocket():
    """Emit air quality data via WebSocket"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required_fields = ['uav_id', 'timestamp', 'air_quality']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Emit the air quality update
        emit_air_quality_update(data)
        
        return jsonify({
            'success': True,
            'message': 'Air quality update emitted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to emit air quality update: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/websocket/emit/hardware-status', methods=['POST'])
def emit_hardware_status_websocket():
    """Emit hardware status update via WebSocket"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required_fields = ['uav_id', 'timestamp', 'status']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Emit the hardware status update
        emit_hardware_status_update(data['uav_id'], data['status'])
        
        return jsonify({
            'success': True,
            'message': 'Hardware status update emitted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to emit hardware status update: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/websocket/emit/drilling', methods=['POST'])
def emit_drilling_websocket():
    """Emit drilling system update via WebSocket"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required_fields = ['uav_id', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Emit the drilling update
        emit_drilling_update(data)
        
        return jsonify({
            'success': True,
            'message': 'Drilling update emitted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to emit drilling update: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/websocket/emit/target-detection', methods=['POST'])
def emit_target_detection_websocket():
    """Emit target detection update via WebSocket"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required_fields = ['uav_id', 'timestamp', 'detections']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Emit the target detection update
        emit_target_detection_update(data)
        
        return jsonify({
            'success': True,
            'message': 'Target detection update emitted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to emit target detection update: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/websocket/emit/video-stream', methods=['POST'])
def emit_video_stream_websocket():
    """Emit video stream status update via WebSocket"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required_fields = ['uav_id', 'stream_status']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Emit the video stream update
        emit_video_stream_update(data['uav_id'], data['stream_status'])
        
        return jsonify({
            'success': True,
            'message': 'Video stream update emitted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to emit video stream update: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/websocket/rooms', methods=['GET'])
def get_websocket_rooms():
    """Get information about active WebSocket rooms"""
    try:
        # In a production app, you'd track active rooms and connections
        # For now, return the available room types
        room_types = {
            'telemetry': 'General telemetry updates',
            'environmental': 'Environmental sensor updates',
            'drilling': 'Drilling system updates',
            'uav_{id}': 'UAV-specific updates',
            'mission_{id}': 'Mission-specific updates',
            'hardware_{id}': 'Hardware-specific updates for UAV',
            'video_{id}': 'Video stream updates for UAV'
        }
        
        return jsonify({
            'success': True,
            'data': {
                'available_rooms': room_types,
                'description': 'Available WebSocket room types for real-time updates'
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get WebSocket rooms: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500