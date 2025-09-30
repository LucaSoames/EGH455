from flask import Blueprint

# Create the API blueprint
api_bp = Blueprint('api', __name__)

# Import all route modules
from app.api import uav_routes, mission_routes, payload_routes, telemetry_routes, auth_routes, dashboard_routes, hardware_routes, video_routes, websocket_routes
