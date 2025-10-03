# /home/pi/EGH455/TAIP/web_server.py

"""
Web Server for the EGH455 TAIP Subsystem

This module provides a Flask-SocketIO web server that serves the React frontend
and provides real-time data APIs via WebSocket connections. It integrates with
the main TAIP application to provide live telemetry data and video streaming.
"""

import os
import cv2
import json
import base64
import threading
import time
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import config
from data_models import PayloadData, EnvironmentalData


class WebServer:
    """Flask-SocketIO web server for the TAIP frontend integration."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'taip_secret_key_2024'
        
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
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
        print(f"Web server initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask HTTP routes."""
        
        @self.app.route('/')
        def serve_frontend():
            """Serve the React frontend index.html."""
            frontend_path = Path(config.PROJECT_ROOT) / "frontend" / "frontend" / "build"
            if frontend_path.exists():
                return send_from_directory(str(frontend_path), 'index.html')
            else:
                return {"error": "Frontend build not found. Please run 'npm run build' in the frontend directory."}, 404
        
        @self.app.route('/<path:path>')
        def serve_static(path):
            """Serve static files from the React build directory."""
            frontend_path = Path(config.PROJECT_ROOT) / "frontend" / "frontend" / "build"
            if frontend_path.exists():
                return send_from_directory(str(frontend_path), path)
            else:
                return {"error": "Frontend build not found"}, 404
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
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
            emit('connected', {'status': 'Connected to TAIP system'})
            
            # Send latest telemetry data if available
            with self._data_lock:
                if self._latest_telemetry:
                    emit('telemetry_update', self._latest_telemetry)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_telemetry')
        def handle_request_telemetry(data):
            """Handle client requests for telemetry data."""
            print(f"Telemetry requested by {request.sid}")
            with self._data_lock:
                if self._latest_telemetry:
                    emit('telemetry_update', self._latest_telemetry)
                else:
                    emit('error', {'message': 'No telemetry data available'})
        
        @self.socketio.on('request_video_frame')
        def handle_request_video_frame(data):
            """Handle client requests for video frames."""
            with self._data_lock:
                if self._latest_frame:
                    # Convert bytes to base64 for transmission
                    frame_b64 = base64.b64encode(self._latest_frame).decode('utf-8')
                    emit('video_frame', {'frame': frame_b64})
                else:
                    emit('error', {'message': 'No video frame available'})
    
    def update_telemetry(self, payload_data: PayloadData):
        """Update telemetry data from the main TAIP application."""
        try:
            # Convert TAIP data format to frontend-expected format
            telemetry_data = {
                'timestamp': payload_data.timestamp,
                'status': 'normal',  # Default status, can be enhanced based on data analysis
                'gauge_pressure_bar': payload_data.gauge_pressure_bar,
            }
            
            # Add environmental data if available
            if payload_data.environmental_data:
                env_data = payload_data.environmental_data
                telemetry_data.update({
                    'temperature': env_data.temperature_c,
                    'humidity': env_data.humidity_rh,
                    'pressure_hpa': env_data.pressure_hpa,
                    'light_lux': env_data.light_lux,
                })
                
                # Add mock battery level (you may want to get this from actual hardware)
                telemetry_data['battery_level'] = 85.0  # Mock value
                
                # Add mock GPS coordinates (you may want to get these from actual GPS)
                telemetry_data['latitude'] = -27.4975  # Mock Brisbane coordinates
                telemetry_data['longitude'] = 153.0137
                telemetry_data['altitude'] = 50.0  # Mock altitude
            
            # Determine status based on gauge pressure
            if payload_data.gauge_pressure_bar is not None:
                if payload_data.gauge_pressure_bar < 1.0:
                    telemetry_data['status'] = 'critical'
                elif payload_data.gauge_pressure_bar < 3.0:
                    telemetry_data['status'] = 'warning'
                else:
                    telemetry_data['status'] = 'normal'
            
            # Thread-safe update
            with self._data_lock:
                self._latest_telemetry = telemetry_data
            
            # Broadcast to all connected clients
            self.socketio.emit('telemetry_update', telemetry_data)
            
        except Exception as e:
            print(f"Error updating telemetry: {e}")
    
    def update_video_frame(self, frame):
        """Update video frame from the main TAIP application."""
        try:
            if frame is not None:
                # Encode frame as JPEG
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if success:
                    frame_bytes = buffer.tobytes()
                    
                    # Thread-safe update
                    with self._data_lock:
                        self._latest_frame = frame_bytes
                    
                    # Convert to base64 and broadcast to all connected clients
                    frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                    self.socketio.emit('video_frame', {'frame': frame_b64})
                    
        except Exception as e:
            print(f"Error updating video frame: {e}")
    
    def run(self, debug: bool = False):
        """Run the web server."""
        print(f"Starting web server on {self.host}:{self.port}")
        self.socketio.run(
            self.app, 
            host=self.host, 
            port=self.port, 
            debug=debug,
            use_reloader=False  # Disable reloader to avoid issues with threading
        )
    
    def run_in_thread(self, debug: bool = False):
        """Run the web server in a background thread."""
        def _run():
            self.run(debug=debug)
        
        server_thread = threading.Thread(target=_run, daemon=True)
        server_thread.start()
        return server_thread


# Standalone test
if __name__ == '__main__':
    import numpy as np
    from data_models import EnvironmentalData, PayloadData
    
    # Create web server
    web_server = WebServer()
    
    # Start server in background thread
    server_thread = web_server.run_in_thread(debug=True)
    
    try:
        # Simulate data updates
        for i in range(100):
            # Create mock telemetry data
            mock_env_data = EnvironmentalData(
                temperature_c=20.0 + i * 0.1,
                pressure_hpa=1013.25 + i * 0.01,
                humidity_rh=45.0 + i * 0.2,
                light_lux=300.0 + i * 2
            )
            
            mock_payload = PayloadData(
                timestamp=datetime.now().isoformat(),
                gauge_pressure_bar=5.0 + i * 0.05,
                environmental_data=mock_env_data
            )
            
            # Create mock video frame
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(mock_frame, f"Frame {i}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            # Update server
            web_server.update_telemetry(mock_payload)
            web_server.update_video_frame(mock_frame)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down test...")
    
    print("Test completed.")