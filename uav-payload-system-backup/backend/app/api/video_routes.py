from flask import request, Response, jsonify
from flask_jwt_extended import jwt_required
from app.api import api_bp
import requests
import os
import logging

logger = logging.getLogger(__name__)

@api_bp.route('/video/stream/<int:uav_id>', methods=['GET'])
@jwt_required()
def proxy_video_stream(uav_id):
    """Proxy video stream from hardware camera"""
    try:
        # Get hardware IP for the specific UAV
        hardware_ip = os.environ.get(f"UAV_{uav_id}_IP", "192.168.1.100")
        hardware_port = os.environ.get(f"UAV_{uav_id}_PORT", "5000")
        
        # Construct the hardware video feed URL
        hardware_url = f"http://{hardware_ip}:{hardware_port}/video_feed"
        
        def generate():
            """Generator function to stream video frames"""
            try:
                response = requests.get(hardware_url, stream=True, timeout=30)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
                        
            except requests.exceptions.RequestException as e:
                logger.error(f"Video stream error for UAV {uav_id}: {e}")
                # Return a simple error frame
                yield b'--frame\r\nContent-Type: text/plain\r\n\r\nVideo stream unavailable\r\n'
        
        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to proxy video stream for UAV {uav_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'Video stream unavailable: {str(e)}'
        }), 500

@api_bp.route('/video/snapshot/<int:uav_id>', methods=['POST'])
@jwt_required()
def capture_snapshot(uav_id):
    """Capture a snapshot from the video stream"""
    try:
        hardware_ip = os.environ.get(f"UAV_{uav_id}_IP", "192.168.1.100")
        hardware_port = os.environ.get(f"UAV_{uav_id}_PORT", "5000")
        
        # Request snapshot from hardware
        hardware_url = f"http://{hardware_ip}:{hardware_port}/snapshot"
        
        response = requests.post(hardware_url, timeout=10)
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'message': 'Snapshot captured successfully',
                'data': {
                    'uav_id': uav_id,
                    'timestamp': response.json().get('timestamp'),
                    'filename': response.json().get('filename')
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Hardware snapshot failed: {response.status_code}'
            }), 500
            
    except Exception as e:
        logger.error(f"Snapshot capture failed for UAV {uav_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'Snapshot capture failed: {str(e)}'
        }), 500

@api_bp.route('/video/recording/<int:uav_id>', methods=['POST'])
@jwt_required()
def control_recording(uav_id):
    """Start/stop video recording"""
    try:
        data = request.get_json()
        action = data.get('action', 'start')  # 'start' or 'stop'
        
        if action not in ['start', 'stop']:
            return jsonify({
                'success': False,
                'error': 'Invalid action. Must be "start" or "stop"'
            }), 400
        
        hardware_ip = os.environ.get(f"UAV_{uav_id}_IP", "192.168.1.100")
        hardware_port = os.environ.get(f"UAV_{uav_id}_PORT", "5000")
        
        # Send recording command to hardware
        hardware_url = f"http://{hardware_ip}:{hardware_port}/recording"
        
        response = requests.post(hardware_url, json={'action': action}, timeout=10)
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'message': f'Recording {action} command sent successfully',
                'data': {
                    'uav_id': uav_id,
                    'action': action,
                    'status': response.json().get('status', 'unknown')
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Recording control failed: {response.status_code}'
            }), 500
            
    except Exception as e:
        logger.error(f"Recording control failed for UAV {uav_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'Recording control failed: {str(e)}'
        }), 500

@api_bp.route('/video/settings/<int:uav_id>', methods=['GET', 'POST'])
@jwt_required()
def video_settings(uav_id):
    """Get or update video stream settings"""
    try:
        hardware_ip = os.environ.get(f"UAV_{uav_id}_IP", "192.168.1.100")
        hardware_port = os.environ.get(f"UAV_{uav_id}_PORT", "5000")
        hardware_url = f"http://{hardware_ip}:{hardware_port}/video_settings"
        
        if request.method == 'GET':
            # Get current video settings
            response = requests.get(hardware_url, timeout=10)
            
            if response.status_code == 200:
                return jsonify({
                    'success': True,
                    'data': {
                        'uav_id': uav_id,
                        'settings': response.json()
                    }
                }), 200
            else:
                # Return default settings if hardware doesn't respond
                return jsonify({
                    'success': True,
                    'data': {
                        'uav_id': uav_id,
                        'settings': {
                            'resolution': '640x480',
                            'fps': 30,
                            'quality': 70,
                            'format': 'mjpeg'
                        }
                    }
                }), 200
                
        elif request.method == 'POST':
            # Update video settings
            settings = request.get_json()
            
            response = requests.post(hardware_url, json=settings, timeout=10)
            
            if response.status_code == 200:
                return jsonify({
                    'success': True,
                    'message': 'Video settings updated successfully',
                    'data': {
                        'uav_id': uav_id,
                        'settings': response.json()
                    }
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': f'Settings update failed: {response.status_code}'
                }), 500
                
    except Exception as e:
        logger.error(f"Video settings operation failed for UAV {uav_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'Video settings operation failed: {str(e)}'
        }), 500

@api_bp.route('/video/streams', methods=['GET'])
@jwt_required()
def list_video_streams():
    """List all available video streams"""
    try:
        # Get all UAVs and their video stream status
        from app.models import UAV
        
        uavs = UAV.query.filter_by(status='active').all()
        streams = []
        
        for uav in uavs:
            hardware_ip = os.environ.get(f"UAV_{uav.id}_IP")
            
            if hardware_ip:
                stream_data = {
                    'uav_id': uav.id,
                    'serial_number': uav.serial_number,
                    'model': uav.model,
                    'stream_url': f'/api/video/stream/{uav.id}',
                    'hardware_ip': hardware_ip,
                    'status': 'available'
                }
                
                # Test if stream is actually available
                try:
                    hardware_port = os.environ.get(f"UAV_{uav.id}_PORT", "5000")
                    test_url = f"http://{hardware_ip}:{hardware_port}/video_feed"
                    test_response = requests.head(test_url, timeout=2)
                    stream_data['status'] = 'online' if test_response.status_code == 200 else 'offline'
                except:
                    stream_data['status'] = 'offline'
                    
                streams.append(stream_data)
        
        return jsonify({
            'success': True,
            'data': streams,
            'count': len(streams)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list video streams: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to list video streams: {str(e)}'
        }), 500