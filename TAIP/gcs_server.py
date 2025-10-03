#!/usr/bin/env python3
"""
Ground Control Station (GCS) Server for EGH455 TAIP Subsystem

This server runs on the GCS laptop (or Pi if needed) and receives:
- Telemetry data from the Pi via POST /telemetry
- Video frames from the Pi via POST /frame

It serves the React frontend and provides WebSocket support for real-time updates.

Usage:
    # On GCS laptop:
    python3 gcs_server.py
    
    # Or specify custom host/port:
    python3 gcs_server.py --host 0.0.0.0 --port 5000
"""

import os
import sys
import argparse
import cv2
import json
import base64
import threading
import time
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Determine if we're running on Pi or laptop
try:
    import config
    PROJECT_ROOT = config.PROJECT_ROOT
except ImportError:
    # Running on laptop without config
    PROJECT_ROOT = Path(__file__).parent.parent

class GCSServer:
    """Ground Control Station server that receives data from the Pi."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'taip_gcs_secret_2024'
        
        # Enable CORS for all domains
        CORS(self.app)
        
        # Initialize SocketIO with CORS support
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            logger=True,
            engineio_logger=True
        )
        
        # Thread-safe data storage
        self._data_lock = threading.Lock()
        self._latest_telemetry: Optional[Dict[str, Any]] = None
        self._latest_frame: Optional[bytes] = None
        self._frame_count = 0
        self._telemetry_count = 0
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
        print(f"=" * 60)
        print(f"GCS Server initialized on {host}:{port}")
        print(f"=" * 60)
    
    def _setup_routes(self):
        """Setup Flask HTTP routes."""
        
        @self.app.route('/')
        def serve_frontend():
            """Serve the React frontend index.html."""
            frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
            if frontend_path.exists():
                return send_from_directory(str(frontend_path), 'index.html')
            else:
                return {
                    "error": "Frontend build not found",
                    "message": "Please run 'npm run build' in the frontend directory",
                    "path": str(frontend_path)
                }, 404
        
        @self.app.route('/<path:path>')
        def serve_static(path):
            """Serve static files from the React build directory."""
            frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
            if frontend_path.exists():
                return send_from_directory(str(frontend_path), path)
            else:
                return {"error": "Frontend build not found"}, 404
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "telemetry_received": self._telemetry_count,
                "frames_received": self._frame_count
            }
        
        @self.app.route('/telemetry', methods=['POST'])
        def receive_telemetry():
            """Receive telemetry data from the Pi."""
            try:
                data = request.get_json()
                if data:
                    with self._data_lock:
                        self._latest_telemetry = data
                        self._telemetry_count += 1
                    
                    # Broadcast to all WebSocket clients
                    self.socketio.emit('telemetry_update', data)
                    return {"status": "ok"}, 200
                else:
                    return {"error": "No data received"}, 400
            except Exception as e:
                print(f"Error receiving telemetry: {e}")
                return {"error": str(e)}, 500
        
        @self.app.route('/frame', methods=['POST'])
        def receive_frame():
            """Receive video frame from the Pi."""
            try:
                frame_data = request.data
                if frame_data:
                    with self._data_lock:
                        self._latest_frame = frame_data
                        self._frame_count += 1
                    
                    # Convert to base64 and broadcast to WebSocket clients
                    frame_b64 = base64.b64encode(frame_data).decode('utf-8')
                    self.socketio.emit('video_frame', {'frame': frame_b64})
                    return {"status": "ok"}, 200
                else:
                    return {"error": "No frame data received"}, 400
            except Exception as e:
                print(f"Error receiving frame: {e}")
                return {"error": str(e)}, 500
        
        @self.app.route('/api/telemetry')
        def get_telemetry():
            """HTTP endpoint to get latest telemetry data."""
            with self._data_lock:
                if self._latest_telemetry:
                    return self._latest_telemetry
                else:
                    return {"error": "No telemetry data available"}, 404
    
    def _setup_socket_handlers(self):
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"Client connected: {request.sid}")
            emit('connected', {'status': 'Connected to GCS server'})
            
            # Send latest data if available
            with self._data_lock:
                if self._latest_telemetry:
                    emit('telemetry_update', self._latest_telemetry)
                if self._latest_frame:
                    frame_b64 = base64.b64encode(self._latest_frame).decode('utf-8')
                    emit('video_frame', {'frame': frame_b64})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_telemetry')
        def handle_request_telemetry():
            """Handle client requests for telemetry data."""
            with self._data_lock:
                if self._latest_telemetry:
                    emit('telemetry_update', self._latest_telemetry)
                else:
                    emit('error', {'message': 'No telemetry data available'})
        
        @self.socketio.on('request_video_frame')
        def handle_request_video_frame():
            """Handle client requests for video frames."""
            with self._data_lock:
                if self._latest_frame:
                    frame_b64 = base64.b64encode(self._latest_frame).decode('utf-8')
                    emit('video_frame', {'frame': frame_b64})
                else:
                    emit('error', {'message': 'No video frame available'})
    
    def run(self, debug: bool = False):
        """Run the GCS server."""
        print(f"Starting GCS server...")
        print(f"Frontend: http://{self.host}:{self.port}/")
        print(f"API endpoints:")
        print(f"  POST /telemetry  - Receive telemetry from Pi")
        print(f"  POST /frame      - Receive video frames from Pi")
        print(f"  GET  /api/health - Health check")
        print(f"=" * 60)
        
        self.socketio.run(
            self.app, 
            host=self.host, 
            port=self.port, 
            debug=debug,
            use_reloader=False
        )

def main():
    parser = argparse.ArgumentParser(description='GCS Server for TAIP System')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    server = GCSServer(host=args.host, port=args.port)
    
    try:
        server.run(debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down GCS server...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()