from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.api import api_bp
from app.models import TelemetryData, UAV, Mission, db
from app.schemas import TelemetryDataSchema
from datetime import datetime, timedelta
from sqlalchemy import and_

telemetry_schema = TelemetryDataSchema()
telemetry_list_schema = TelemetryDataSchema(many=True)

@api_bp.route('/telemetry', methods=['POST'])
def receive_telemetry():
    """Receive telemetry data from UAV (external endpoint)"""
    try:
        json_data = request.get_json()
        
        # Validate input data
        errors = telemetry_schema.validate(json_data)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Check if UAV exists
        uav = UAV.query.get(json_data['uav_id'])
        if not uav:
            return jsonify({'success': False, 'error': 'UAV not found'}), 400
        
        # Create telemetry data
        telemetry = TelemetryData(**json_data)
        db.session.add(telemetry)
        db.session.commit()
        
        # Emit real-time update via WebSocket
        from app import socketio
        socketio.emit('telemetry_update', telemetry_schema.dump(telemetry), room='telemetry')
        
        return jsonify({
            'success': True,
            'message': 'Telemetry data received'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/telemetry/uav/<int:uav_id>', methods=['GET'])
@jwt_required()
def get_uav_telemetry(uav_id):
    """Get telemetry data for a specific UAV"""
    try:
        # Validate UAV exists
        uav = UAV.query.get_or_404(uav_id)
        
        # Query parameters
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        limit = request.args.get('limit', 100, type=int)
        latest = request.args.get('latest', type=bool, default=False)
        
        query = TelemetryData.query.filter_by(uav_id=uav_id)
        
        # Apply time filters
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            query = query.filter(TelemetryData.timestamp >= start_dt)
        if end_time:
            end_dt = datetime.fromisoformat(end_time)
            query = query.filter(TelemetryData.timestamp <= end_dt)
        
        # Order by timestamp (latest first)
        query = query.order_by(TelemetryData.timestamp.desc())
        
        if latest:
            # Get only the latest telemetry
            telemetry = query.first()
            if telemetry:
                return jsonify({
                    'success': True,
                    'data': telemetry_schema.dump(telemetry)
                }), 200
            else:
                return jsonify({
                    'success': True,
                    'data': None,
                    'message': 'No telemetry data found'
                }), 200
        else:
            # Get paginated results
            telemetry_data = query.limit(limit).all()
            return jsonify({
                'success': True,
                'data': telemetry_list_schema.dump(telemetry_data),
                'count': len(telemetry_data)
            }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/telemetry/mission/<int:mission_id>', methods=['GET'])
@jwt_required()
def get_mission_telemetry(mission_id):
    """Get telemetry data for a specific mission"""
    try:
        # Validate mission exists
        mission = Mission.query.get_or_404(mission_id)
        
        # Query parameters
        limit = request.args.get('limit', 100, type=int)
        
        query = TelemetryData.query.filter_by(mission_id=mission_id)\
            .order_by(TelemetryData.timestamp.desc())
        
        telemetry_data = query.limit(limit).all()
        
        return jsonify({
            'success': True,
            'data': telemetry_list_schema.dump(telemetry_data),
            'count': len(telemetry_data)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/telemetry/live', methods=['GET'])
@jwt_required()
def get_live_telemetry():
    """Get latest telemetry data for all active UAVs"""
    try:
        # Get all active UAVs
        active_uavs = UAV.query.filter_by(status='active').all()
        
        live_data = []
        for uav in active_uavs:
            # Get latest telemetry for each UAV
            latest_telemetry = TelemetryData.query.filter_by(uav_id=uav.id)\
                .order_by(TelemetryData.timestamp.desc()).first()
            
            if latest_telemetry:
                telemetry_data = telemetry_schema.dump(latest_telemetry)
                telemetry_data['uav_info'] = {
                    'serial_number': uav.serial_number,
                    'model': uav.model
                }
                live_data.append(telemetry_data)
        
        return jsonify({
            'success': True,
            'data': live_data
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/telemetry/stats/<int:uav_id>', methods=['GET'])
@jwt_required()
def get_telemetry_stats(uav_id):
    """Get telemetry statistics for a UAV"""
    try:
        # Validate UAV exists
        uav = UAV.query.get_or_404(uav_id)
        
        # Time range for stats (last 24 hours by default)
        hours = request.args.get('hours', 24, type=int)
        since_time = datetime.utcnow() - timedelta(hours=hours)
        
        telemetry_data = TelemetryData.query.filter(
            and_(
                TelemetryData.uav_id == uav_id,
                TelemetryData.timestamp >= since_time
            )
        ).all()
        
        if not telemetry_data:
            return jsonify({
                'success': True,
                'data': None,
                'message': 'No telemetry data in specified time range'
            }), 200
        
        # Calculate statistics
        speeds = [t.speed for t in telemetry_data if t.speed is not None]
        altitudes = [t.altitude for t in telemetry_data if t.altitude is not None]
        battery_levels = [t.battery_level for t in telemetry_data if t.battery_level is not None]
        
        stats = {
            'time_range_hours': hours,
            'total_data_points': len(telemetry_data),
            'first_timestamp': min(t.timestamp for t in telemetry_data).isoformat() if telemetry_data else None,
            'last_timestamp': max(t.timestamp for t in telemetry_data).isoformat() if telemetry_data else None,
            'speed': {
                'min': min(speeds) if speeds else None,
                'max': max(speeds) if speeds else None,
                'avg': sum(speeds) / len(speeds) if speeds else None
            },
            'altitude': {
                'min': min(altitudes) if altitudes else None,
                'max': max(altitudes) if altitudes else None,
                'avg': sum(altitudes) / len(altitudes) if altitudes else None
            },
            'battery': {
                'min': min(battery_levels) if battery_levels else None,
                'max': max(battery_levels) if battery_levels else None,
                'avg': sum(battery_levels) / len(battery_levels) if battery_levels else None,
                'current': battery_levels[-1] if battery_levels else None
            },
            'flight_path': [
                {
                    'latitude': t.latitude,
                    'longitude': t.longitude,
                    'altitude': t.altitude,
                    'timestamp': t.timestamp.isoformat()
                } for t in telemetry_data[-50:]  # Last 50 points for flight path
            ]
        }
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/telemetry/alerts', methods=['GET'])
@jwt_required()
def get_telemetry_alerts():
    """Get telemetry-based alerts and warnings"""
    try:
        # Get recent telemetry data with warnings or errors
        hours = request.args.get('hours', 24, type=int)
        since_time = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = TelemetryData.query.filter(
            and_(
                TelemetryData.timestamp >= since_time,
                TelemetryData.system_status.in_(['warning', 'error'])
            )
        ).order_by(TelemetryData.timestamp.desc()).limit(50).all()
        
        alert_data = []
        for alert in alerts:
            uav = UAV.query.get(alert.uav_id)
            alert_info = telemetry_schema.dump(alert)
            alert_info['uav_info'] = {
                'serial_number': uav.serial_number,
                'model': uav.model
            }
            alert_data.append(alert_info)
        
        return jsonify({
            'success': True,
            'data': alert_data
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
