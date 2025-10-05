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
        
        # Disable Flask request logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        # Enable CORS for all domains
        CORS(self.app)
        
        # Initialize SocketIO with CORS support (disable verbose logging)
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            logger=False,  # Disable SocketIO logging
            engineio_logger=False  # Disable Engine.IO logging
        )
        
        # Thread-safe data storage
        self._data_lock = threading.Lock()
        self._latest_telemetry: Optional[Dict[str, Any]] = None
        self._latest_frame: Optional[bytes] = None
        self._frame_count = 0
        self._telemetry_count = 0
        
        # LCD control state
        self._lcd_tab_index = 0
        
        # Add counters for periodic status updates
        self._last_status_time = time.time()
        self._status_interval = 5.0  # Print status every 5 seconds
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
        print(f"=" * 60)
        print(f"GCS Server initialized on {host}:{port}")
        print(f"=" * 60)

    def _print_periodic_status(self):
        """Print status update periodically instead of every frame/telemetry."""
        now = time.time()
        if (now - self._last_status_time) >= self._status_interval:
            print(f"[STATUS] Telemetry: {self._telemetry_count} | Frames: {self._frame_count}")
            self._last_status_time = now

    def _setup_routes(self):
        """Setup Flask HTTP routes."""
        
        # Debug: Print frontend path on startup (only once)
        frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
        print(f"Frontend build path: {frontend_path}")
        print(f"Frontend path exists: {frontend_path.exists()}")
        
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
                    
                    # Broadcast to WebSocket clients
                    self.socketio.emit('telemetry_update', data)
                    
                    # Print periodic status instead of every telemetry
                    self._print_periodic_status()
                    
                    return {"status": "ok"}, 200
                else:
                    return {"error": "No data received"}, 400
            except Exception as e:
                print(f"[ERROR] Telemetry error: {e}")
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
                    
                    # Print periodic status instead of every frame
                    self._print_periodic_status()
                    
                    return {"status": "ok"}, 200
                else:
                    return {"error": "No frame data received"}, 400
            except Exception as e:
                print(f"[ERROR] Frame error: {e}")
                return {"error": str(e)}, 500

        @self.app.route('/api/telemetry')
        def get_telemetry():
            """HTTP endpoint to get latest telemetry data."""
            with self._data_lock:
                if self._latest_telemetry:
                    return self._latest_telemetry, 200
                else:
                    return {"error": "No telemetry available"}, 404

        @self.app.route('/api/lcd/tab', methods=['POST'])
        def set_lcd_tab():
            """Set LCD tab on the Pi."""
            try:
                data = request.get_json()
                tab_index = data.get('tab_index')
                
                if tab_index is not None and 0 <= tab_index <= 2:
                    with self._data_lock:
                        self._lcd_tab_index = tab_index
                    
                    # Broadcast to Pi via SocketIO
                    self.socketio.emit('lcd_tab_command', {'tab_index': tab_index})
                    
                    # Also broadcast to all web clients for state sync
                    self.socketio.emit('lcd_tab_update', {'tab_index': tab_index})
                    
                    print(f"[LCD] Tab changed to: {tab_index}")
                    
                    return {"status": "ok", "tab_index": tab_index}, 200
                else:
                    return {"error": "Invalid tab_index (must be 0-2)"}, 400
            except Exception as e:
                print(f"[ERROR] LCD tab error: {e}")
                return {"error": str(e)}, 500

        # Add GET endpoint to retrieve current LCD tab state
        @self.app.route('/api/lcd/tab', methods=['GET'])
        def get_lcd_tab():
            """Get current LCD tab index."""
            with self._data_lock:
                return {"tab_index": self._lcd_tab_index}, 200

        @self.app.route('/')
        def serve_frontend():
            """Serve the React frontend index.html."""
            frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
            index_path = frontend_path / "index.html"
            if index_path.exists():
                return send_from_directory(str(frontend_path), 'index.html')
            else:
                print(f"[ERROR] Frontend build not found at {frontend_path}")
                return {"error": "Frontend build not found"}, 404

        # Catch-all route for client-side routing (must be last)
        @self.app.route('/<path:path>')
        def serve_other_files(path):
            """Serve other files (manifest, favicon, etc) from the React build directory."""
            frontend_path = PROJECT_ROOT / "frontend" / "frontend" / "build"
            requested_file = frontend_path / path
            
            if frontend_path.exists():
                try:
                    result = send_from_directory(str(frontend_path), path)
                    return result
                except Exception:
                    # If file not found, return index.html for client-side routing
                    return send_from_directory(str(frontend_path), 'index.html')
            else:
                print(f"[ERROR] Frontend build not found at {frontend_path}")
                return {"error": "Frontend build not found"}, 404

    def _setup_socket_handlers(self):
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"[SOCKET] Client connected: {request.sid}")
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
            print(f"[SOCKET] Client disconnected: {request.sid}")
        
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
            use_reloader=False,
            log_output=False  # Suppress socketio logs
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