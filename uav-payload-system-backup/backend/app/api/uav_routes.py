from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.api import api_bp
from app.models import UAV, db
from app.schemas import UAVSchema
from datetime import datetime

uav_schema = UAVSchema()
uavs_schema = UAVSchema(many=True)

@api_bp.route('/uavs', methods=['GET'])
@jwt_required()
def get_uavs():
    """Get all UAVs with optional filtering"""
    try:
        status = request.args.get('status')
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        
        query = UAV.query
        
        if status:
            query = query.filter(UAV.status == status)
        
        uavs = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'data': uavs_schema.dump(uavs.items),
            'pagination': {
                'page': uavs.page,
                'pages': uavs.pages,
                'per_page': uavs.per_page,
                'total': uavs.total
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/uavs/<int:uav_id>', methods=['GET'])
@jwt_required()
def get_uav(uav_id):
    """Get a specific UAV by ID"""
    try:
        uav = UAV.query.get_or_404(uav_id)
        return jsonify({
            'success': True,
            'data': uav_schema.dump(uav)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404

@api_bp.route('/uavs', methods=['POST'])
@jwt_required()
def create_uav():
    """Create a new UAV"""
    try:
        json_data = request.get_json()
        
        # Validate input data
        errors = uav_schema.validate(json_data)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Check if serial number already exists
        existing_uav = UAV.query.filter_by(serial_number=json_data['serial_number']).first()
        if existing_uav:
            return jsonify({
                'success': False, 
                'error': 'UAV with this serial number already exists'
            }), 400
        
        uav = UAV(**json_data)
        db.session.add(uav)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': uav_schema.dump(uav),
            'message': 'UAV created successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/uavs/<int:uav_id>', methods=['PUT'])
@jwt_required()
def update_uav(uav_id):
    """Update an existing UAV"""
    try:
        uav = UAV.query.get_or_404(uav_id)
        json_data = request.get_json()
        
        # Validate input data
        errors = uav_schema.validate(json_data, partial=True)
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Update UAV fields
        for field, value in json_data.items():
            if hasattr(uav, field):
                setattr(uav, field, value)
        
        uav.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': uav_schema.dump(uav),
            'message': 'UAV updated successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/uavs/<int:uav_id>', methods=['DELETE'])
@jwt_required()
def delete_uav(uav_id):
    """Delete a UAV"""
    try:
        uav = UAV.query.get_or_404(uav_id)
        
        # Check if UAV has active missions
        from app.models import Mission
        active_missions = Mission.query.filter_by(uav_id=uav_id, status='active').count()
        if active_missions > 0:
            return jsonify({
                'success': False,
                'error': 'Cannot delete UAV with active missions'
            }), 400
        
        db.session.delete(uav)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'UAV deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/uavs/<int:uav_id>/status', methods=['PUT'])
@jwt_required()
def update_uav_status(uav_id):
    """Update UAV status"""
    try:
        uav = UAV.query.get_or_404(uav_id)
        json_data = request.get_json()
        
        if 'status' not in json_data:
            return jsonify({'success': False, 'error': 'Status is required'}), 400
        
        valid_statuses = ['active', 'inactive', 'maintenance']
        if json_data['status'] not in valid_statuses:
            return jsonify({
                'success': False, 
                'error': f'Status must be one of: {valid_statuses}'
            }), 400
        
        uav.status = json_data['status']
        uav.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': uav_schema.dump(uav),
            'message': 'UAV status updated successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
