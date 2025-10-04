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
    python3 gcs_server.py --host 0.0.0.0 --port 3000
"""

import os
import sys
import argparse
import cv2
import json
import base64
import threading
import time
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Determine if we're running on Pi or laptop
try:
    # import config
    # PROJECT_ROOT = config.PROJECT_ROOT
    PROJECT_ROOT = Path(__file__).parent.parent
except ImportError:
    # Running on laptop without config
    PROJECT_ROOT = Path(__file__).parent.parent

class GCSServer:
    """Ground Control Station server that receives data from the Pi."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        
        frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
        static_path = frontend_path / "static"

        self.app = Flask(__name__, static_folder=str(static_path), static_url_path='/static')
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
        
        # Debug: Print frontend path on startup
        frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
        print(f"Frontend build path: {frontend_path}")
        print(f"Frontend path exists: {frontend_path.exists()}")
        if frontend_path.exists():
            print(f"Frontend contents: {list(frontend_path.iterdir())[:5]}")
        
        # API routes first (before catch-all)
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
        
        @self.app.route('/')
        def serve_frontend():
            """Serve the React frontend index.html."""
            frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
            index_path = frontend_path / "index.html"
            print(f"[ROOT] Serving index.html")
            print(f"[ROOT] Frontend path: {frontend_path}")
            print(f"[ROOT] Frontend exists: {frontend_path.exists()}")
            print(f"[ROOT] Index path: {index_path}")
            print(f"[ROOT] Index exists: {index_path.exists()}")
            
            if not frontend_path.exists():
                error_msg = {
                    "error": "Frontend build not found",
                    "message": "Please run 'npm run build' in the frontend directory",
                    "path": str(frontend_path)
                }
                print(f"[ROOT] ERROR: {error_msg}")
                return error_msg, 404
            
            if not index_path.exists():
                error_msg = {
                    "error": "index.html not found",
                    "path": str(index_path)
                }
                print(f"[ROOT] ERROR: {error_msg}")
                return error_msg, 404
            
            try:
                return send_from_directory(str(frontend_path), 'index.html')
            except Exception as e:
                print(f"[ROOT] ERROR: {e}")
                return {"error": str(e)}, 500
        
        # Catch-all route for client-side routing (must be last)
        @self.app.route('/<path:path>')
        def serve_other_files(path):
            """Serve other files (manifest, favicon, etc) from the React build directory."""
            frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
            requested_file = frontend_path / path
            print(f"[CATCH-ALL] Requested: /{path}")
            print(f"[CATCH-ALL] Full path: {requested_file}")
            print(f"[CATCH-ALL] Exists: {requested_file.exists()}")
            if frontend_path.exists():
                try:
                    result = send_from_directory(str(frontend_path), path)
                    print(f"[CATCH-ALL] Successfully served: {path}")
                    return result
                except Exception as e:
                    # If file not found, return index.html for client-side routing
                    print(f"[CATCH-ALL] File not found, serving index.html. Error: {e}")
                    return send_from_directory(str(frontend_path), 'index.html')
            else:
                print(f"[CATCH-ALL] ERROR: Frontend build not found at {frontend_path}")
                return {"error": "Frontend build not found"}, 404
        
        @self.app.route('/debug/paths')
        def debug_paths():
            """Debug endpoint to check file paths."""
            try:
                frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
                static_dir = frontend_path / "static"
                
                result = {
                    "project_root": str(PROJECT_ROOT),
                    "project_root_exists": PROJECT_ROOT.exists(),
                    "frontend_path": str(frontend_path),
                    "frontend_exists": frontend_path.exists(),
                    "static_dir": str(static_dir),
                    "static_exists": static_dir.exists(),
                }
                
                # List frontend contents
                if frontend_path.exists():
                    try:
                        result["frontend_contents"] = [f.name for f in frontend_path.iterdir()]
                    except Exception as e:
                        result["frontend_contents_error"] = str(e)
                
                # List static contents
                if static_dir.exists():
                    try:
                        static_files = []
                        for f in static_dir.rglob("*"):
                            if f.is_file():
                                try:
                                    rel_path = f.relative_to(static_dir)
                                    static_files.append(str(rel_path))
                                except:
                                    static_files.append(f.name)
                        result["static_contents"] = static_files[:30]  # First 30 files
                    except Exception as e:
                        result["static_contents_error"] = str(e)
                
                return jsonify(result)
                
            except Exception as e:
                import traceback
                return jsonify({
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }), 500

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
    parser.add_argument('--port', type=int, default=3000, help='Port to bind to (default: 3000)')
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