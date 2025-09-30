from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.api import api_bp
from app.models import UAV, Mission, Payload, TelemetryData, SystemLog, User, db
from app.schemas import DashboardStatsSchema, UAVStatusSummarySchema, MissionSummarySchema
from datetime import datetime, timedelta
from sqlalchemy import func, and_

dashboard_stats_schema = DashboardStatsSchema()
uav_status_schema = UAVStatusSummarySchema(many=True)
mission_summary_schema = MissionSummarySchema(many=True)

@api_bp.route('/dashboard/stats', methods=['GET'])
@jwt_required()
def get_dashboard_stats():
    """Get overall system statistics for dashboard"""
    try:
        # Get today's date for filtering
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        
        # Calculate statistics
        stats = {
            'total_uavs': UAV.query.count(),
            'active_uavs': UAV.query.filter_by(status='active').count(),
            'total_missions': Mission.query.count(),
            'active_missions': Mission.query.filter_by(status='active').count(),
            'completed_missions_today': Mission.query.filter(
                and_(
                    Mission.status == 'completed',
                    Mission.actual_end_time >= today_start
                )
            ).count(),
            'total_payloads': Payload.query.count(),
            'available_payloads': Payload.query.filter_by(status='available').count(),
            'system_alerts': TelemetryData.query.filter(
                and_(
                    TelemetryData.system_status.in_(['warning', 'error']),
                    TelemetryData.timestamp >= today_start
                )
            ).count()
        }
        
        return jsonify({
            'success': True,
            'data': dashboard_stats_schema.dump(stats)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/dashboard/uav-status', methods=['GET'])
@jwt_required()
def get_uav_status_summary():
    """Get UAV status summary for dashboard"""
    try:
        uavs = UAV.query.all()
        uav_summaries = []
        
        for uav in uavs:
            # Get current mission
            current_mission = Mission.query.filter_by(
                uav_id=uav.id, 
                status='active'
            ).first()
            
            # Get latest telemetry
            latest_telemetry = TelemetryData.query.filter_by(uav_id=uav.id)\
                .order_by(TelemetryData.timestamp.desc()).first()
            
            summary = {
                'uav_id': uav.id,
                'serial_number': uav.serial_number,
                'model': uav.model,
                'status': uav.status,
                'current_mission_id': current_mission.id if current_mission else None,
                'battery_level': latest_telemetry.battery_level if latest_telemetry else None,
                'last_telemetry': latest_telemetry.timestamp if latest_telemetry else None,
                'location': {
                    'latitude': latest_telemetry.latitude,
                    'longitude': latest_telemetry.longitude,
                    'altitude': latest_telemetry.altitude
                } if latest_telemetry else None
            }
            
            uav_summaries.append(summary)
        
        return jsonify({
            'success': True,
            'data': uav_status_schema.dump(uav_summaries)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/dashboard/mission-summary', methods=['GET'])
@jwt_required()
def get_mission_summary():
    """Get mission summary for dashboard"""
    try:
        # Get active and planned missions
        missions = Mission.query.filter(
            Mission.status.in_(['active', 'planned'])
        ).join(UAV).all()
        
        mission_summaries = []
        
        for mission in missions:
            # Calculate progress for active missions
            progress_percentage = 0
            estimated_completion = None
            
            if mission.status == 'active' and mission.actual_start_time:
                elapsed_time = datetime.utcnow() - mission.actual_start_time
                total_time = timedelta(minutes=mission.estimated_duration)
                progress_percentage = min((elapsed_time.total_seconds() / total_time.total_seconds()) * 100, 100)
                
                if progress_percentage < 100:
                    remaining_time = total_time - elapsed_time
                    estimated_completion = datetime.utcnow() + remaining_time
            
            summary = {
                'mission_id': mission.id,
                'name': mission.name,
                'status': mission.status,
                'priority': mission.priority,
                'uav_serial': mission.uav.serial_number,
                'progress_percentage': progress_percentage,
                'estimated_completion': estimated_completion
            }
            
            mission_summaries.append(summary)
        
        return jsonify({
            'success': True,
            'data': mission_summary_schema.dump(mission_summaries)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/dashboard/recent-activity', methods=['GET'])
@jwt_required()
def get_recent_activity():
    """Get recent system activity"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        recent_logs = SystemLog.query.join(User, SystemLog.user_id == User.id, isouter=True)\
            .order_by(SystemLog.timestamp.desc())\
            .limit(limit).all()
        
        activity_data = []
        for log in recent_logs:
            activity = {
                'id': log.id,
                'action': log.action,
                'resource_type': log.resource_type,
                'resource_id': log.resource_id,
                'details': log.details,
                'username': log.user.username if log.user_id else 'System',
                'timestamp': log.timestamp.isoformat()
            }
            activity_data.append(activity)
        
        return jsonify({
            'success': True,
            'data': activity_data
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/dashboard/fleet-health', methods=['GET'])
@jwt_required()
def get_fleet_health():
    """Get fleet health metrics"""
    try:
        # Get UAVs with their latest telemetry
        uavs = UAV.query.all()
        fleet_health = []
        
        for uav in uavs:
            latest_telemetry = TelemetryData.query.filter_by(uav_id=uav.id)\
                .order_by(TelemetryData.timestamp.desc()).first()
            
            health_status = 'unknown'
            issues = []
            
            if latest_telemetry:
                # Determine health based on telemetry
                if latest_telemetry.system_status == 'error':
                    health_status = 'critical'
                    if latest_telemetry.error_messages:
                        issues.append(latest_telemetry.error_messages)
                elif latest_telemetry.system_status == 'warning':
                    health_status = 'warning'
                else:
                    health_status = 'good'
                
                # Check battery level
                if latest_telemetry.battery_level < 20:
                    health_status = 'warning' if health_status == 'good' else health_status
                    issues.append('Low battery')
                
                # Check signal strength
                if latest_telemetry.signal_strength < -80:
                    health_status = 'warning' if health_status == 'good' else health_status
                    issues.append('Weak signal')
                
                # Check if telemetry is stale (older than 5 minutes)
                if (datetime.utcnow() - latest_telemetry.timestamp).seconds > 300:
                    health_status = 'warning' if health_status == 'good' else health_status
                    issues.append('Stale telemetry data')
            
            health_info = {
                'uav_id': uav.id,
                'serial_number': uav.serial_number,
                'model': uav.model,
                'status': uav.status,
                'health_status': health_status,
                'issues': issues,
                'last_contact': latest_telemetry.timestamp.isoformat() if latest_telemetry else None,
                'battery_level': latest_telemetry.battery_level if latest_telemetry else None,
                'signal_strength': latest_telemetry.signal_strength if latest_telemetry else None
            }
            
            fleet_health.append(health_info)
        
        return jsonify({
            'success': True,
            'data': fleet_health
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/dashboard/mission-metrics', methods=['GET'])
@jwt_required()
def get_mission_metrics():
    """Get mission performance metrics"""
    try:
        # Time range for metrics (last 30 days by default)
        days = request.args.get('days', 30, type=int)
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Mission completion statistics
        total_missions = Mission.query.filter(Mission.created_at >= since_date).count()
        completed_missions = Mission.query.filter(
            and_(
                Mission.status == 'completed',
                Mission.created_at >= since_date
            )
        ).count()
        aborted_missions = Mission.query.filter(
            and_(
                Mission.status == 'aborted',
                Mission.created_at >= since_date
            )
        ).count()
        
        # Average mission duration for completed missions
        completed_with_times = Mission.query.filter(
            and_(
                Mission.status == 'completed',
                Mission.actual_start_time.isnot(None),
                Mission.actual_end_time.isnot(None),
                Mission.created_at >= since_date
            )
        ).all()
        
        if completed_with_times:
            durations = [(m.actual_end_time - m.actual_start_time).total_seconds() / 60 
                        for m in completed_with_times]
            avg_duration = sum(durations) / len(durations)
        else:
            avg_duration = 0
        
        # Mission types breakdown
        mission_types = db.session.query(
            Mission.mission_type,
            func.count(Mission.id)
        ).filter(Mission.created_at >= since_date)\
         .group_by(Mission.mission_type).all()
        
        metrics = {
            'time_period_days': days,
            'total_missions': total_missions,
            'completed_missions': completed_missions,
            'aborted_missions': aborted_missions,
            'success_rate': (completed_missions / total_missions * 100) if total_missions > 0 else 0,
            'average_duration_minutes': round(avg_duration, 2),
            'mission_types': [{'type': mt[0], 'count': mt[1]} for mt in mission_types]
        }
        
        return jsonify({
            'success': True,
            'data': metrics
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
