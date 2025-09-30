from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.api import api_bp
from app.models import TelemetryData, UAV, db
from app.schemas import TelemetryDataSchema
from datetime import datetime, timedelta
from sqlalchemy import and_, desc
import requests
import os

telemetry_schema = TelemetryDataSchema()

@api_bp.route('/hardware/air-quality', methods=['GET'])
@jwt_required()
def get_air_quality_data():
    """Get latest air quality data from hardware sensors"""
    try:
        uav_id = request.args.get('uav_id', type=int)
        hours = request.args.get('hours', 1, type=int)
        
        query = TelemetryData.query.filter(
            TelemetryData.air_quality_data.isnot(None)
        )
        
        if uav_id:
            query = query.filter(TelemetryData.uav_id == uav_id)
            
        # Filter by time range
        since_time = datetime.utcnow() - timedelta(hours=hours)
        query = query.filter(TelemetryData.timestamp >= since_time)
        
        telemetry_data = query.order_by(desc(TelemetryData.timestamp)).limit(100).all()
        
        air_quality_readings = []
        for data in telemetry_data:
            if data.air_quality_data:
                reading = {
                    'timestamp': data.timestamp.isoformat(),
                    'uav_id': data.uav_id,
                    'air_quality': data.air_quality_data,
                    'environmental': data.environmental_data or {},
                    'location': {
                        'latitude': data.latitude,
                        'longitude': data.longitude,
                        'altitude': data.altitude
                    }
                }
                air_quality_readings.append(reading)
        
        return jsonify({
            'success': True,
            'data': air_quality_readings,
            'count': len(air_quality_readings)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/hardware/environmental', methods=['GET'])
@jwt_required()
def get_environmental_data():
    """Get environmental sensor data (temperature, humidity, pressure, light)"""
    try:
        uav_id = request.args.get('uav_id', type=int)
        hours = request.args.get('hours', 24, type=int)
        
        query = TelemetryData.query.filter(
            TelemetryData.environmental_data.isnot(None)
        )
        
        if uav_id:
            query = query.filter(TelemetryData.uav_id == uav_id)
            
        since_time = datetime.utcnow() - timedelta(hours=hours)
        query = query.filter(TelemetryData.timestamp >= since_time)
        
        telemetry_data = query.order_by(desc(TelemetryData.timestamp)).limit(200).all()
        
        environmental_readings = []
        for data in telemetry_data:
            if data.environmental_data:
                reading = {
                    'timestamp': data.timestamp.isoformat(),
                    'uav_id': data.uav_id,
                    'environmental': data.environmental_data,
                    'location': {
                        'latitude': data.latitude,
                        'longitude': data.longitude,
                        'altitude': data.altitude
                    }
                }
                environmental_readings.append(reading)
        
        return jsonify({
            'success': True,
            'data': environmental_readings,
            'count': len(environmental_readings)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/hardware/drilling', methods=['GET'])
@jwt_required()
def get_drilling_data():
    """Get drilling mechanism status and data"""
    try:
        uav_id = request.args.get('uav_id', type=int)
        
        query = TelemetryData.query.filter(
            TelemetryData.drilling_data.isnot(None)
        )
        
        if uav_id:
            query = query.filter(TelemetryData.uav_id == uav_id)
        
        # Get latest drilling data
        latest_data = query.order_by(desc(TelemetryData.timestamp)).first()
        
        if latest_data and latest_data.drilling_data:
            return jsonify({
                'success': True,
                'data': {
                    'timestamp': latest_data.timestamp.isoformat(),
                    'uav_id': latest_data.uav_id,
                    'drilling': latest_data.drilling_data,
                    'location': {
                        'latitude': latest_data.latitude,
                        'longitude': latest_data.longitude,
                        'altitude': latest_data.altitude
                    }
                }
            }), 200
        else:
            return jsonify({
                'success': True,
                'data': None,
                'message': 'No drilling data available'
            }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/hardware/drilling/control', methods=['POST'])
@jwt_required()
def control_drilling():
    """Send drilling commands to hardware"""
    try:
        data = request.get_json()
        
        required_fields = ['uav_id', 'action']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False, 
                    'error': f'Missing required field: {field}'
                }), 400
        
        uav_id = data['uav_id']
        action = data['action']  # 'start', 'stop', 'reverse'
        duration = data.get('duration', 10)  # seconds
        
        # Validate action
        if action not in ['start', 'stop', 'reverse']:
            return jsonify({
                'success': False,
                'error': 'Invalid action. Must be start, stop, or reverse'
            }), 400
        
        # In a real implementation, this would send commands to the hardware
        # For now, we'll simulate by storing the command in telemetry data
        command_data = {
            'command': action,
            'duration': duration,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'pending'
        }
        
        # TODO: Send actual command to hardware via HTTP request or direct GPIO control
        # hardware_url = os.environ.get("HARDWARE_URL", "http://raspberry-pi:5000")
        # response = requests.post(f"{hardware_url}/drilling/control", json=command_data)
        
        return jsonify({
            'success': True,
            'message': f'Drilling command {action} sent successfully',
            'data': command_data
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/hardware/target-detection', methods=['GET'])
@jwt_required()
def get_target_detection_data():
    """Get target detection results from computer vision"""
    try:
        uav_id = request.args.get('uav_id', type=int)
        hours = request.args.get('hours', 1, type=int)
        
        query = TelemetryData.query.filter(
            TelemetryData.target_detection_data.isnot(None)
        )
        
        if uav_id:
            query = query.filter(TelemetryData.uav_id == uav_id)
            
        since_time = datetime.utcnow() - timedelta(hours=hours)
        query = query.filter(TelemetryData.timestamp >= since_time)
        
        telemetry_data = query.order_by(desc(TelemetryData.timestamp)).limit(50).all()
        
        detection_data = []
        for data in telemetry_data:
            if data.target_detection_data:
                reading = {
                    'timestamp': data.timestamp.isoformat(),
                    'uav_id': data.uav_id,
                    'detections': data.target_detection_data,
                    'location': {
                        'latitude': data.latitude,
                        'longitude': data.longitude,
                        'altitude': data.altitude
                    }
                }
                detection_data.append(reading)
        
        return jsonify({
            'success': True,
            'data': detection_data,
            'count': len(detection_data)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/hardware/video/stream', methods=['GET'])
@jwt_required()
def get_video_stream_url():
    """Get video stream URL for hardware camera"""
    try:
        uav_id = request.args.get('uav_id', type=int)
        
        if not uav_id:
            return jsonify({
                'success': False,
                'error': 'uav_id parameter is required'
            }), 400
        
        # In a real implementation, this would return the actual stream URL
        # based on the UAV's IP address and configuration
        hardware_ip = os.environ.get(f"UAV_{uav_id}_IP", "192.168.1.100")
        stream_url = f"http://{hardware_ip}:5000/video_feed"
        
        return jsonify({
            'success': True,
            'data': {
                'stream_url': stream_url,
                'uav_id': uav_id,
                'status': 'active'
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/hardware/status', methods=['GET'])
@jwt_required()
def get_hardware_status():
    """Get overall hardware system status"""
    try:
        uav_id = request.args.get('uav_id', type=int)
        
        query = TelemetryData.query.filter(
            TelemetryData.hardware_status.isnot(None)
        )
        
        if uav_id:
            query = query.filter(TelemetryData.uav_id == uav_id)
        
        # Get latest hardware status
        latest_status = query.order_by(desc(TelemetryData.timestamp)).first()
        
        if latest_status and latest_status.hardware_status:
            status_data = {
                'timestamp': latest_status.timestamp.isoformat(),
                'uav_id': latest_status.uav_id,
                'hardware_status': latest_status.hardware_status,
                'system_health': 'healthy' if all(
                    latest_status.hardware_status.get(key, False) 
                    for key in ['sensors_online', 'camera_online']
                ) else 'degraded'
            }
            
            return jsonify({
                'success': True,
                'data': status_data
            }), 200
        else:
            return jsonify({
                'success': True,
                'data': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'hardware_status': {
                        'sensors_online': False,
                        'camera_online': False,
                        'servo_online': False
                    },
                    'system_health': 'unknown'
                }
            }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/hardware/calibrate', methods=['POST'])
@jwt_required()
def calibrate_sensors():
    """Calibrate hardware sensors"""
    try:
        data = request.get_json()
        
        sensor_type = data.get('sensor_type', 'all')  # 'gas', 'environmental', 'camera', 'all'
        uav_id = data.get('uav_id')
        
        if not uav_id:
            return jsonify({
                'success': False,
                'error': 'uav_id is required'
            }), 400
        
        # In a real implementation, this would send calibration commands to hardware
        calibration_result = {
            'sensor_type': sensor_type,
            'uav_id': uav_id,
            'status': 'completed',
            'timestamp': datetime.utcnow().isoformat(),
            'message': f'{sensor_type} sensors calibrated successfully'
        }
        
        # TODO: Send actual calibration command to hardware
        # hardware_url = os.environ.get(f"UAV_{uav_id}_URL", "http://raspberry-pi:5000")
        # response = requests.post(f"{hardware_url}/calibrate", json={'sensor_type': sensor_type})
        
        return jsonify({
            'success': True,
            'data': calibration_result
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500