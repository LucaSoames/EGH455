from marshmallow import Schema, fields, validate, post_load
from app import ma
from app.models import UAV, Mission, Payload, TelemetryData, Waypoint, User, SystemLog

class UAVSchema(ma.SQLAlchemyAutoSchema):
    """Schema for UAV serialization"""
    class Meta:
        model = UAV
        load_instance = True
        include_relationships = True
        
    serial_number = fields.Str(required=True, validate=validate.Length(min=1, max=50))
    model = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    max_payload_weight = fields.Float(required=True, validate=validate.Range(min=0))
    max_altitude = fields.Float(required=True, validate=validate.Range(min=0))
    max_speed = fields.Float(required=True, validate=validate.Range(min=0))
    battery_capacity = fields.Float(required=True, validate=validate.Range(min=0))
    communication_range = fields.Float(required=True, validate=validate.Range(min=0))
    status = fields.Str(validate=validate.OneOf(['active', 'inactive', 'maintenance']))

class PayloadSchema(ma.SQLAlchemyAutoSchema):
    """Schema for Payload serialization"""
    class Meta:
        model = Payload
        load_instance = True
        
    name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    payload_type = fields.Str(required=True, validate=validate.Length(min=1, max=50))
    weight = fields.Float(required=True, validate=validate.Range(min=0))
    status = fields.Str(validate=validate.OneOf(['available', 'deployed', 'maintenance']))

class WaypointSchema(ma.SQLAlchemyAutoSchema):
    """Schema for Waypoint serialization"""
    class Meta:
        model = Waypoint
        load_instance = True
        
    sequence_number = fields.Int(required=True, validate=validate.Range(min=1))
    latitude = fields.Float(required=True, validate=validate.Range(min=-90, max=90))
    longitude = fields.Float(required=True, validate=validate.Range(min=-180, max=180))
    altitude = fields.Float(required=True, validate=validate.Range(min=0))

class MissionSchema(ma.SQLAlchemyAutoSchema):
    """Schema for Mission serialization"""
    class Meta:
        model = Mission
        load_instance = True
        include_relationships = True
        
    # Nested schemas
    uav = fields.Nested(UAVSchema, only=['id', 'serial_number', 'model', 'status'])
    payload = fields.Nested(PayloadSchema, only=['id', 'name', 'payload_type', 'weight'])
    waypoints = fields.Nested(WaypointSchema, many=True)
    
    name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    mission_type = fields.Str(required=True, validate=validate.Length(min=1, max=50))
    uav_id = fields.Int(required=True)
    start_latitude = fields.Float(required=True, validate=validate.Range(min=-90, max=90))
    start_longitude = fields.Float(required=True, validate=validate.Range(min=-180, max=180))
    end_latitude = fields.Float(required=True, validate=validate.Range(min=-90, max=90))
    end_longitude = fields.Float(required=True, validate=validate.Range(min=-180, max=180))
    planned_altitude = fields.Float(required=True, validate=validate.Range(min=0))
    status = fields.Str(validate=validate.OneOf(['planned', 'active', 'completed', 'aborted']))
    priority = fields.Str(validate=validate.OneOf(['low', 'medium', 'high', 'critical']))

class TelemetryDataSchema(ma.SQLAlchemyAutoSchema):
    """Schema for TelemetryData serialization"""
    class Meta:
        model = TelemetryData
        load_instance = True
        
    latitude = fields.Float(required=True, validate=validate.Range(min=-90, max=90))
    longitude = fields.Float(required=True, validate=validate.Range(min=-180, max=180))
    altitude = fields.Float(required=True, validate=validate.Range(min=0))
    heading = fields.Float(required=True, validate=validate.Range(min=0, max=360))
    speed = fields.Float(required=True, validate=validate.Range(min=0))
    battery_level = fields.Float(required=True, validate=validate.Range(min=0, max=100))
    system_status = fields.Str(validate=validate.OneOf(['normal', 'warning', 'error']))
    
    # Hardware integration JSON fields
    air_quality_data = fields.Dict(missing=None)
    environmental_data = fields.Dict(missing=None)
    drilling_data = fields.Dict(missing=None)
    target_detection_data = fields.Dict(missing=None)
    hardware_status = fields.Dict(missing=None)

class UserSchema(ma.SQLAlchemyAutoSchema):
    """Schema for User serialization"""
    class Meta:
        model = User
        load_instance = True
        exclude = ['password_hash']
        
    username = fields.Str(required=True, validate=validate.Length(min=3, max=80))
    email = fields.Email(required=True)
    role = fields.Str(validate=validate.OneOf(['admin', 'operator', 'viewer']))
    password = fields.Str(load_only=True, required=True, validate=validate.Length(min=6))

class SystemLogSchema(ma.SQLAlchemyAutoSchema):
    """Schema for SystemLog serialization"""
    class Meta:
        model = SystemLog
        load_instance = True
        
    action = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    resource_type = fields.Str(required=True, validate=validate.Length(min=1, max=50))

# Dashboard specific schemas
class DashboardStatsSchema(Schema):
    """Schema for dashboard statistics"""
    total_uavs = fields.Int()
    active_uavs = fields.Int()
    total_missions = fields.Int()
    active_missions = fields.Int()
    completed_missions_today = fields.Int()
    total_payloads = fields.Int()
    available_payloads = fields.Int()
    system_alerts = fields.Int()

class UAVStatusSummarySchema(Schema):
    """Schema for UAV status summary"""
    uav_id = fields.Int()
    serial_number = fields.Str()
    model = fields.Str()
    status = fields.Str()
    current_mission_id = fields.Int(allow_none=True)
    battery_level = fields.Float(allow_none=True)
    last_telemetry = fields.DateTime(allow_none=True)
    location = fields.Dict(allow_none=True)

class MissionSummarySchema(Schema):
    """Schema for mission summary"""
    mission_id = fields.Int()
    name = fields.Str()
    status = fields.Str()
    priority = fields.Str()
    uav_serial = fields.Str()
    progress_percentage = fields.Float()
    estimated_completion = fields.DateTime(allow_none=True)
