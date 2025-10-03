#!/usr/bin/env python3
"""
Integrated Web Server for EGH455 TAIP System
Serves the React frontend and provides API endpoints for telemetry and video streaming.
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, Response, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import cv2
import numpy as np
from pathlib import Path
import base64

# Import database functionality
import db
db.init_db()

app = Flask(__name__)
CORS(app)

# Configuration
FRONTEND_BUILD_PATH = Path("/home/pi/EGH455/frontend/frontend/build")
STATIC_PATH = FRONTEND_BUILD_PATH / "static"

class IntegratedWebServer:
    """Main web server class that handles both frontend serving and API endpoints."""
    
    def __init__(self):
        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.telemetry_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.audit_logs = []
        self.system_status = {
            "camera_connected": False,
            "taip_running": False,
            "last_update": None
        }
        
        print("Integrated Web Server initialized")
    
    def update_telemetry(self, telemetry_data: Dict[str, Any]):
        """Update the latest telemetry data from TAIP system."""
        with self.telemetry_lock:
            self.latest_telemetry = telemetry_data
            self.system_status["last_update"] = datetime.now().isoformat()
            self.system_status["taip_running"] = True
            
            # Add to audit logs
            self.audit_logs.append({
                "timestamp": datetime.now().isoformat(),
                "type": "telemetry_update",
                "data": f"Pressure: {telemetry_data.get('gauge_pressure_bar', 'N/A')} bar, "
                       f"Detections: {len(telemetry_data.get('detections', []))}"
            })
            
            # Keep only last 100 audit entries
            if len(self.audit_logs) > 100:
                self.audit_logs = self.audit_logs[-100:]
    
    def update_frame(self, frame: np.ndarray):
        """Update the latest video frame from TAIP system."""
        with self.frame_lock:
            self.latest_frame = frame.copy() if frame is not None else None
            self.system_status["camera_connected"] = frame is not None
    
    def get_latest_telemetry(self) -> Optional[Dict[str, Any]]:
        """Get the latest telemetry data."""
        with self.telemetry_lock:
            return self.latest_telemetry.copy() if self.latest_telemetry else None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest video frame."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

# Global server instance
web_server = IntegratedWebServer()

# ============================================================================
# FRONTEND SERVING ROUTES
# ============================================================================

@app.route('/')
def serve_frontend():
    """Serve the React app's index.html"""
    return send_file(FRONTEND_BUILD_PATH / 'index.html')

@app.route('/static/<path:path>')
def serve_static_files(path):
    """Serve static files (JS, CSS, etc.) from React build"""
    return send_from_directory(STATIC_PATH, path)

@app.route('/manifest.json')
def serve_manifest():
    """Serve the React app's manifest.json"""
    return send_file(FRONTEND_BUILD_PATH / 'manifest.json')

# Catch-all route for React Router (SPA routing)
@app.route('/<path:path>')
def serve_spa_routes(path):
    """Handle React Router routes by serving index.html"""
    # Check if it's a static file request
    if '.' in path and path.split('.')[-1] in ['js', 'css', 'png', 'jpg', 'ico', 'svg']:
        try:
            return send_from_directory(FRONTEND_BUILD_PATH, path)
        except:
            pass
    
    # For all other routes, serve the React app
    return send_file(FRONTEND_BUILD_PATH / 'index.html')

# ============================================================================
# API ROUTES FOR TAIP INTEGRATION
# ============================================================================

@app.route('/api/telemetry', methods=['GET'])
def get_telemetry():
    """Get the latest telemetry data for the frontend."""
    telemetry = web_server.get_latest_telemetry()
    
    if not telemetry:
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "status": "no_data",
            "message": "No telemetry data available"
        })
    
    # Format the data for the React frontend
    formatted_data = {
        "timestamp": telemetry.get("timestamp", datetime.now().isoformat()),
        "gauge_pressure_bar": telemetry.get("gauge_pressure_bar"),
        "environmental_data": telemetry.get("environmental_data", {}),
        "detections": telemetry.get("detections", []),
        "aruco_markers": telemetry.get("aruco_markers", []),
        "system_status": telemetry.get("system_status", web_server.system_status)
    }
    
    return jsonify(formatted_data)

@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    """Get system status information."""
    return jsonify(web_server.system_status)

@app.route('/api/audit-logs', methods=['GET'])
def get_audit_logs():
    """Get audit logs for the frontend."""
    return jsonify({
        "logs": web_server.audit_logs[-50:],  # Return last 50 logs
        "total_count": len(web_server.audit_logs)
    })

@app.route('/api/video-stream')
def video_stream():
    """Serve video stream with YOLO detections overlaid."""
    def generate_frames():
        while True:
            frame = web_server.get_latest_frame()
            
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                # Send a placeholder frame if no data available
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera data...", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.1)  # 10 FPS
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================================
# ROUTES FOR TAIP SYSTEM TO POST DATA
# ============================================================================

@app.route('/telemetry', methods=['POST'])
def receive_telemetry():
    """Receive telemetry data from TAIP system."""
    try:
        data = request.get_json()
        if data:
            web_server.update_telemetry(data)
            return jsonify({"status": "success", "message": "Telemetry received"})
        else:
            return jsonify({"status": "error", "message": "No data received"}), 400
    except Exception as e:
        print(f"Error receiving telemetry: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/frame', methods=['POST'])
def receive_frame():
    """Receive video frame from TAIP system."""
    try:
        # Check if frame is sent as binary data
        if request.content_type and 'image' in request.content_type:
            # Decode image from binary data
            nparr = np.frombuffer(request.data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Try to get frame from JSON (base64 encoded)
            data = request.get_json()
            if data and 'frame' in data:
                # Decode base64 frame
                frame_data = base64.b64decode(data['frame'])
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                return jsonify({"status": "error", "message": "No frame data"}), 400
        
        if frame is not None:
            web_server.update_frame(frame)
            return jsonify({"status": "success", "message": "Frame received"})
        else:
            return jsonify({"status": "error", "message": "Failed to decode frame"}), 400
            
    except Exception as e:
        print(f"Error receiving frame: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================================================
# LEGACY COMPATIBILITY ROUTES
# ============================================================================

@app.route('/sensor_data')
def sensor_data():
    """Legacy route for sensor data (compatibility with existing code)."""
    latest = db.get_latest_reading() or {}
    return jsonify(latest)

# ============================================================================
# STARTUP AND MAIN
# ============================================================================

def start_server():
    """Start the integrated web server."""
    print("=" * 60)
    print("EGH455 Integrated Web Server Starting...")
    print("=" * 60)
    print(f"Frontend build path: {FRONTEND_BUILD_PATH}")
    print(f"Static files path: {STATIC_PATH}")
    print("")
    print("Available endpoints:")
    print("  Frontend:        http://<pi-ip>:5000/")
    print("  Telemetry API:   http://<pi-ip>:5000/api/telemetry")
    print("  Video Stream:    http://<pi-ip>:5000/api/video-stream")
    print("  System Status:   http://<pi-ip>:5000/api/system-status")
    print("  Audit Logs:      http://<pi-ip>:5000/api/audit-logs")
    print("")
    print("TAIP Integration endpoints:")
    print("  POST /telemetry  - Receives TAIP telemetry data")
    print("  POST /frame      - Receives TAIP video frames")
    print("=" * 60)
    
    # Check if React build exists
    if not FRONTEND_BUILD_PATH.exists():
        print(f"WARNING: React build not found at {FRONTEND_BUILD_PATH}")
        print("Please run 'npm run build' in the frontend directory")
    else:
        print(f"✓ React build found at {FRONTEND_BUILD_PATH}")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    start_server()