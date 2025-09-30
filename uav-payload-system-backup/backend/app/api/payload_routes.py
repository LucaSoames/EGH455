from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.api import api_bp
from app.models import Payload, Mission, db
from app.schemas import PayloadSchema
from datetime import datetime

payload_schema = PayloadSchema()
payloads_schema = PayloadSchema(many=True)

@api_bp.route('/payloads', methods=['GET'])
@jwt_required()
def get_payloads():
    """Get all payloads with optional filtering"""
    try:
        status = request.args.get('status')
        payload_type = request.args.get('type')
        max_weight = request.args.get('max_weight', type=float)
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        
        query = Payload.query
        
        if status:
            query = query.filter(Payload.status == status)
        if payload_type:
            query = query.filter(Payload.payload_type == payload_type)
        if max_weight:
            query = query.filter(Payload.weight <= max_weight)
        
        payloads = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'data': payloads_schema.dump(payloads.items),
            'pagination': {
                'page': payloads.page,
                'pages': payloads.pages,
                'per_page': payloads.per_page,
                'total': payloads.total
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/payloads/<int:payload_id>', methods=['GET'])
@jwt_required()
def get_payload(payload_id):
    """Get a specific payload by ID"""
    try:
        payload = Payload.query.get_or_404(payload_id)
        return jsonify({
            'success': True,
            'data': payload_schema.dump(payload)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404

@api_bp.route('/payloads', methods=['POST'])
@jwt_required()
def create_payload():
    """Create a new payload"""
    try:
        json_data = request.get_json()
        
        # Validate input data
        errors = payload_schema.validate(json_data)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        payload = Payload(**json_data)
        db.session.add(payload)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': payload_schema.dump(payload),
            'message': 'Payload created successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/payloads/<int:payload_id>', methods=['PUT'])
@jwt_required()
def update_payload(payload_id):
    """Update an existing payload"""
    try:
        payload = Payload.query.get_or_404(payload_id)
        json_data = request.get_json()
        
        # Prevent updating deployed payloads
        if payload.status == 'deployed':
            return jsonify({
                'success': False,
                'error': 'Cannot update deployed payload'
            }), 400
        
        # Validate input data
        errors = payload_schema.validate(json_data, partial=True)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Update payload fields
        for field, value in json_data.items():
            if hasattr(payload, field):
                setattr(payload, field, value)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': payload_schema.dump(payload),
            'message': 'Payload updated successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/payloads/<int:payload_id>', methods=['DELETE'])
@jwt_required()
def delete_payload(payload_id):
    """Delete a payload"""
    try:
        payload = Payload.query.get_or_404(payload_id)
        
        # Check if payload is deployed
        if payload.status == 'deployed':
            return jsonify({
                'success': False,
                'error': 'Cannot delete deployed payload'
            }), 400
        
        # Check if payload has active missions
        active_missions = Mission.query.filter_by(payload_id=payload_id, status='active').count()
        if active_missions > 0:
            return jsonify({
                'success': False,
                'error': 'Cannot delete payload with active missions'
            }), 400
        
        db.session.delete(payload)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Payload deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/payloads/<int:payload_id>/status', methods=['PUT'])
@jwt_required()
def update_payload_status(payload_id):
    """Update payload status"""
    try:
        payload = Payload.query.get_or_404(payload_id)
        json_data = request.get_json()
        
        if 'status' not in json_data:
            return jsonify({'success': False, 'error': 'Status is required'}), 400
        
        valid_statuses = ['available', 'deployed', 'maintenance']
        if json_data['status'] not in valid_statuses:
            return jsonify({
                'success': False,
                'error': f'Status must be one of: {valid_statuses}'
            }), 400
        
        payload.status = json_data['status']
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': payload_schema.dump(payload),
            'message': 'Payload status updated successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/payloads/types', methods=['GET'])
@jwt_required()
def get_payload_types():
    """Get available payload types"""
    payload_types = [
        'camera',
        'sensor',
        'delivery_package',
        'medical_supplies',
        'surveillance_equipment',
        'communication_relay',
        'environmental_sensor',
        'thermal_camera',
        'gps_tracker',
        'emergency_beacon'
    ]
    
    return jsonify({
        'success': True,
        'data': payload_types
    }), 200

@api_bp.route('/payloads/<int:payload_id>/missions', methods=['GET'])
@jwt_required()
def get_payload_missions(payload_id):
    """Get missions that use this payload"""
    try:
        payload = Payload.query.get_or_404(payload_id)
        
        missions = Mission.query.filter_by(payload_id=payload_id)\
            .order_by(Mission.planned_start_time.desc()).all()
        
        from app.schemas import MissionSchema
        mission_schema = MissionSchema(many=True, only=['id', 'name', 'status', 'mission_type', 'planned_start_time', 'actual_start_time', 'actual_end_time'])
        
        return jsonify({
            'success': True,
            'data': mission_schema.dump(missions)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
