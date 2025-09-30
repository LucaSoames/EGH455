from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.api import api_bp
from app.models import Mission, UAV, Payload, Waypoint, db
from app.schemas import MissionSchema, WaypointSchema
from datetime import datetime, timedelta
from sqlalchemy import and_, or_

mission_schema = MissionSchema()
missions_schema = MissionSchema(many=True)
waypoint_schema = WaypointSchema()
waypoints_schema = WaypointSchema(many=True)

@api_bp.route('/missions', methods=['GET'])
@jwt_required()
def get_missions():
    """Get all missions with optional filtering"""
    try:
        # Query parameters
        status = request.args.get('status')
        uav_id = request.args.get('uav_id', type=int)
        mission_type = request.args.get('mission_type')
        priority = request.args.get('priority')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        
        query = Mission.query
        
        # Apply filters
        if status:
            query = query.filter(Mission.status == status)
        if uav_id:
            query = query.filter(Mission.uav_id == uav_id)
        if mission_type:
            query = query.filter(Mission.mission_type == mission_type)
        if priority:
            query = query.filter(Mission.priority == priority)
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
            query = query.filter(Mission.planned_start_time >= start_dt)
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
            query = query.filter(Mission.planned_start_time <= end_dt)
        
        # Order by priority and planned start time
        query = query.order_by(
            Mission.priority.desc(),
            Mission.planned_start_time.asc()
        )
        
        missions = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'data': missions_schema.dump(missions.items),
            'pagination': {
                'page': missions.page,
                'pages': missions.pages,
                'per_page': missions.per_page,
                'total': missions.total
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/missions/<int:mission_id>', methods=['GET'])
@jwt_required()
def get_mission(mission_id):
    """Get a specific mission by ID"""
    try:
        mission = Mission.query.get_or_404(mission_id)
        return jsonify({
            'success': True,
            'data': mission_schema.dump(mission)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404

@api_bp.route('/missions', methods=['POST'])
@jwt_required()
def create_mission():
    """Create a new mission"""
    try:
        json_data = request.get_json()
        
        # Validate input data
        errors = mission_schema.validate(json_data)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Check if UAV exists and is available
        uav = UAV.query.get(json_data['uav_id'])
        if not uav:
            return jsonify({'success': False, 'error': 'UAV not found'}), 400
        
        if uav.status != 'active':
            return jsonify({
                'success': False, 
                'error': 'UAV is not active and cannot be assigned to missions'
            }), 400
        
        # Check if UAV has conflicting missions
        planned_start = datetime.fromisoformat(json_data['planned_start_time'])
        estimated_end = planned_start + timedelta(minutes=json_data['estimated_duration'])
        
        conflicting_missions = Mission.query.filter(
            and_(
                Mission.uav_id == json_data['uav_id'],
                Mission.status.in_(['planned', 'active']),
                or_(
                    and_(
                        Mission.planned_start_time <= planned_start,
                        Mission.planned_start_time + timedelta(minutes=Mission.estimated_duration) > planned_start
                    ),
                    and_(
                        Mission.planned_start_time < estimated_end,
                        Mission.planned_start_time >= planned_start
                    )
                )
            )
        ).first()
        
        if conflicting_missions:
            return jsonify({
                'success': False,
                'error': 'UAV has conflicting mission during this time period'
            }), 400
        
        # Check payload if specified
        if 'payload_id' in json_data and json_data['payload_id']:
            payload = Payload.query.get(json_data['payload_id'])
            if not payload:
                return jsonify({'success': False, 'error': 'Payload not found'}), 400
            
            if payload.status != 'available':
                return jsonify({
                    'success': False,
                    'error': 'Payload is not available'
                }), 400
            
            # Check payload weight against UAV capacity
            if payload.weight > uav.max_payload_weight:
                return jsonify({
                    'success': False,
                    'error': 'Payload weight exceeds UAV capacity'
                }), 400
        
        # Create mission
        mission = Mission(**json_data)
        db.session.add(mission)
        
        # Mark payload as deployed if assigned
        if 'payload_id' in json_data and json_data['payload_id']:
            payload.status = 'deployed'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': mission_schema.dump(mission),
            'message': 'Mission created successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/missions/<int:mission_id>', methods=['PUT'])
@jwt_required()
def update_mission(mission_id):
    """Update an existing mission"""
    try:
        mission = Mission.query.get_or_404(mission_id)
        json_data = request.get_json()
        
        # Prevent updating active missions
        if mission.status == 'active':
            return jsonify({
                'success': False,
                'error': 'Cannot update active mission'
            }), 400
        
        # Validate input data
        errors = mission_schema.validate(json_data, partial=True)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Update mission fields
        for field, value in json_data.items():
            if hasattr(mission, field) and field != 'id':
                setattr(mission, field, value)
        
        mission.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': mission_schema.dump(mission),
            'message': 'Mission updated successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/missions/<int:mission_id>/status', methods=['PUT'])
@jwt_required()
def update_mission_status(mission_id):
    """Update mission status"""
    try:
        mission = Mission.query.get_or_404(mission_id)
        json_data = request.get_json()
        
        if 'status' not in json_data:
            return jsonify({'success': False, 'error': 'Status is required'}), 400
        
        new_status = json_data['status']
        valid_statuses = ['planned', 'active', 'completed', 'aborted']
        
        if new_status not in valid_statuses:
            return jsonify({
                'success': False,
                'error': f'Status must be one of: {valid_statuses}'
            }), 400
        
        old_status = mission.status
        mission.status = new_status
        
        # Update timestamps based on status
        if new_status == 'active' and old_status == 'planned':
            mission.actual_start_time = datetime.utcnow()
        elif new_status in ['completed', 'aborted'] and old_status == 'active':
            mission.actual_end_time = datetime.utcnow()
            # Release payload if mission is completed/aborted
            if mission.payload:
                mission.payload.status = 'available'
        
        mission.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': mission_schema.dump(mission),
            'message': 'Mission status updated successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/missions/<int:mission_id>', methods=['DELETE'])
@jwt_required()
def delete_mission(mission_id):
    """Delete a mission"""
    try:
        mission = Mission.query.get_or_404(mission_id)
        
        # Prevent deleting active missions
        if mission.status == 'active':
            return jsonify({
                'success': False,
                'error': 'Cannot delete active mission'
            }), 400
        
        # Release payload if assigned
        if mission.payload:
            mission.payload.status = 'available'
        
        db.session.delete(mission)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Mission deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/missions/<int:mission_id>/waypoints', methods=['GET'])
@jwt_required()
def get_mission_waypoints(mission_id):
    """Get waypoints for a mission"""
    try:
        mission = Mission.query.get_or_404(mission_id)
        waypoints = Waypoint.query.filter_by(mission_id=mission_id)\
            .order_by(Waypoint.sequence_number).all()
        
        return jsonify({
            'success': True,
            'data': waypoints_schema.dump(waypoints)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/missions/<int:mission_id>/waypoints', methods=['POST'])
@jwt_required()
def add_mission_waypoint(mission_id):
    """Add a waypoint to a mission"""
    try:
        mission = Mission.query.get_or_404(mission_id)
        
        if mission.status == 'active':
            return jsonify({
                'success': False,
                'error': 'Cannot modify waypoints of active mission'
            }), 400
        
        json_data = request.get_json()
        json_data['mission_id'] = mission_id
        
        # Validate input data
        errors = waypoint_schema.validate(json_data)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        waypoint = Waypoint(**json_data)
        db.session.add(waypoint)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': waypoint_schema.dump(waypoint),
            'message': 'Waypoint added successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/missions/types', methods=['GET'])
@jwt_required()
def get_mission_types():
    """Get available mission types"""
    mission_types = [
        'surveillance',
        'delivery',
        'mapping',
        'rescue',
        'inspection',
        'monitoring',
        'reconnaissance',
        'search'
    ]
    
    return jsonify({
        'success': True,
        'data': mission_types
    }), 200
