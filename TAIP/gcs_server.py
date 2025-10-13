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
import signal
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
    # Uncomment above if config is available
    PROJECT_ROOT = Path(__file__).parent.parent
except ImportError:
    # Running on laptop without config
    PROJECT_ROOT = Path(__file__).parent.parent

# Import audit logger
try:
    from audit_logger import (get_audit_logger, log_system, log_telemetry, 
                              log_network, log_sensor, log_vision)
    AUDIT_LOGGING_AVAILABLE = True
except ImportError:
    AUDIT_LOGGING_AVAILABLE = False
    print("[WARNING] Audit logging not available")

class GCSServer:
    """Ground Control Station server that receives data from the Pi."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        self.running = True
        
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
        
        # Initialize audit logger
        if AUDIT_LOGGING_AVAILABLE:
            self.audit_logger = get_audit_logger()
            log_system("GCS Server Started", f"Initialized on {host}:{port}", "success")
        else:
            self.audit_logger = None
        
        print(f"=" * 60)
        print(f"GCS Server initialized on {host}:{port}")
        print(f"Audit Logging: {'Enabled' if AUDIT_LOGGING_AVAILABLE else 'Disabled'}")
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
                    
                    # Log telemetry events
                    if AUDIT_LOGGING_AVAILABLE:
                        # Log pressure reading
                        pressure = data.get('gauge_pressure_bar')
                        if pressure is not None:
                            status = "success" if pressure > 3.0 else ("warning" if pressure > 1.0 else "error")
                            log_telemetry(
                                "Pressure Reading",
                                f"Gauge pressure: {pressure:.2f} bar",
                                status=status,
                                pressure=pressure
                            )
                        
                        # Log environmental data
                        env_data = data.get('environmental_data')
                        if env_data:
                            temp = env_data.get('temperature_c')
                            humidity = env_data.get('humidity_rh')
                            press = env_data.get('pressure_hpa')
                            light = env_data.get('light_lux')
                            
                            # Determine status based on temperature
                            env_status = "info"
                            if temp is not None:
                                if temp > 45:
                                    env_status = "error"
                                elif temp > 35 or temp < 5:
                                    env_status = "warning"
                                else:
                                    env_status = "success"
                            
                            details_parts = []
                            if temp is not None:
                                details_parts.append(f"Temp: {temp:.1f}°C")
                            if humidity is not None:
                                details_parts.append(f"Humidity: {humidity:.1f}%")
                            if press is not None:
                                details_parts.append(f"Pressure: {press:.1f} hPa")
                            if light is not None:
                                details_parts.append(f"Light: {light:.1f} lux")
                            
                            if details_parts:
                                log_sensor(
                                    "Environmental Reading",
                                    ", ".join(details_parts),
                                    status=env_status,
                                    temperature=temp,
                                    humidity=humidity,
                                    pressure_hpa=press,
                                    light=light
                                )
                        
                        # Log YOLO detections
                        detections = data.get('yolo_detections', [])
                        if detections:
                            detection_summary = f"{len(detections)} objects detected"
                            if detections:
                                classes = [d.get('class_name', 'unknown') for d in detections]
                                detection_summary += f": {', '.join(classes)}"
                            log_vision(
                                "Object Detection",
                                detection_summary,
                                status="info",
                                count=len(detections),
                                classes=classes if detections else []
                            )
                        
                        # Log ArUco markers
                        aruco_markers = data.get('aruco_markers', [])
                        if aruco_markers:
                            marker_ids = [m.get('marker_id') for m in aruco_markers]
                            distances = [m.get('distance_m') for m in aruco_markers]
                            details = f"{len(aruco_markers)} marker(s): "
                            details += ", ".join([f"ID {mid} at {dist:.2f}m" 
                                                for mid, dist in zip(marker_ids, distances)])
                            log_vision(
                                "ArUco Detection",
                                details,
                                status="success",
                                marker_count=len(aruco_markers),
                                marker_ids=marker_ids
                            )
                        
                        # Log drill events
                        drill_events = data.get('drill_events', [])
                        for event in drill_events:
                            action = event.get('action', 'Unknown')
                            details = event.get('details', '')
                            status = event.get('status', 'info')
                            metadata = event.get('metadata', {})
                            
                            audit_logger = get_audit_logger()
                            audit_logger.log(
                                event_type='drill',
                                action=action,
                                details=details,
                                status=status,
                                metadata=metadata
                            )
                    
                    # Broadcast the FULL telemetry data to WebSocket clients
                    self.socketio.emit('telemetry_update', data)
                    
                    # Print periodic status instead of every telemetry
                    self._print_periodic_status()
                    
                    return {"status": "ok"}, 200
                else:
                    return {"error": "No data received"}, 400
            except Exception as e:
                print(f"[ERROR] Telemetry error: {e}")
                import traceback
                traceback.print_exc()
                if AUDIT_LOGGING_AVAILABLE:
                    log_network("Telemetry Error", str(e), "error")
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
                    
                    # Log frame reception periodically (every 100 frames to avoid spam)
                    if AUDIT_LOGGING_AVAILABLE and self._frame_count % 100 == 0:
                        log_network(
                            "Video Frames Received",
                            f"Total frames: {self._frame_count}",
                            status="info",
                            frame_count=self._frame_count
                        )
                    
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
                if AUDIT_LOGGING_AVAILABLE:
                    log_network("Frame Error", str(e), "error")
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
                    
                    # Log LCD change
                    if AUDIT_LOGGING_AVAILABLE:
                        tab_names = ['IP', 'Camera', 'Temperature']
                        log_system(
                            "LCD Tab Changed",
                            f"Display switched to: {tab_names[tab_index]} (Tab {tab_index})",
                            "info",
                            tab_index=tab_index,
                            tab_name=tab_names[tab_index]
                        )
                    
                    return {"status": "ok", "tab_index": tab_index}, 200
                else:
                    return {"error": "Invalid tab_index (must be 0-2)"}, 400
            except Exception as e:
                print(f"[ERROR] LCD tab error: {e}")
                if AUDIT_LOGGING_AVAILABLE:
                    log_network("LCD Tab Error", str(e), "error")
                return {"error": str(e)}, 500

        # Add GET endpoint to retrieve current LCD tab state
        @self.app.route('/api/lcd/tab', methods=['GET'])
        def get_lcd_tab():
            """Get current LCD tab index."""
            with self._data_lock:
                return {"tab_index": self._lcd_tab_index}, 200

        # Audit log endpoints
        @self.app.route('/api/audit/logs', methods=['GET'])
        def get_audit_logs():
            """Get audit logs with optional filtering and pagination."""
            if not AUDIT_LOGGING_AVAILABLE:
                return {"error": "Audit logging not available"}, 503
            
            try:
                # Get query parameters
                limit = int(request.args.get('limit', 100))
                offset = int(request.args.get('offset', 0))
                event_type = request.args.get('event_type')
                status = request.args.get('status')
                search = request.args.get('search')
                start_date = request.args.get('start_date')
                end_date = request.args.get('end_date')
                
                # Get logs and total count
                logs = self.audit_logger.get_logs(
                    limit=limit,
                    offset=offset,
                    event_type=event_type,
                    status=status,
                    search=search,
                    start_date=start_date,
                    end_date=end_date
                )
                
                total_count = self.audit_logger.get_log_count(
                    event_type=event_type,
                    status=status,
                    search=search,
                    start_date=start_date,
                    end_date=end_date
                )
                
                return {
                    "logs": logs,
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset
                }, 200
                
            except Exception as e:
                print(f"[ERROR] Audit logs error: {e}")
                return {"error": str(e)}, 500

        @self.app.route('/api/audit/stats', methods=['GET'])
        def get_audit_stats():
            """Get audit log statistics."""
            if not AUDIT_LOGGING_AVAILABLE:
                return {"error": "Audit logging not available"}, 503
            
            try:
                stats = self.audit_logger.get_stats()
                return stats, 200
            except Exception as e:
                print(f"[ERROR] Audit stats error: {e}")
                return {"error": str(e)}, 500

        @self.app.route('/api/audit/clear', methods=['POST'])
        def clear_old_audit_logs():
            """Clear old audit logs (admin function)."""
            if not AUDIT_LOGGING_AVAILABLE:
                return {"error": "Audit logging not available"}, 503
            
            try:
                data = request.get_json() or {}
                days = int(data.get('days', 30))
                deleted_count = self.audit_logger.clear_old_logs(days)
                
                log_system("Audit Logs Cleared", 
                          f"Deleted {deleted_count} logs older than {days} days",
                          "warning")
                
                return {
                    "status": "ok",
                    "deleted_count": deleted_count,
                    "days": days
                }, 200
            except Exception as e:
                print(f"[ERROR] Clear audit logs error: {e}")
                return {"error": str(e)}, 500

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
            
            if AUDIT_LOGGING_AVAILABLE:
                log_network("Client Connected", f"WebSocket client: {request.sid}", "info")
            
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
            
            if AUDIT_LOGGING_AVAILABLE:
                log_network("Client Disconnected", f"WebSocket client: {request.sid}", "info")
        
        @self.socketio.on('request_telemetry')
        def handle_request_telemetry(data=None):
            """Handle client requests for telemetry data."""
            with self._data_lock:
                if self._latest_telemetry:
                    emit('telemetry_update', self._latest_telemetry)
                else:
                    emit('error', {'message': 'No telemetry data available'})
        
        @self.socketio.on('request_video_frame')
        def handle_request_video_frame(data=None):
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
        print(f"Press CTRL+C to stop the server gracefully")
        print(f"=" * 60)
        
        try:
            self.socketio.run(
                self.app, 
                host=self.host, 
                port=self.port, 
                debug=debug,
                use_reloader=False,
                log_output=False  # Suppress socketio logs
            )
        except KeyboardInterrupt:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown the GCS server."""
        if not self.running:
            return
            
        print("\n" + "=" * 60)
        print("Shutting down GCS server...")
        print("=" * 60)
        
        self.running = False
        
        # Print final statistics
        with self._data_lock:
            print(f"Final Statistics:")
            print(f"  Total Telemetry Received: {self._telemetry_count}")
            print(f"  Total Frames Received: {self._frame_count}")
        
        # Notify all connected clients
        try:
            self.socketio.emit('server_shutdown', {'message': 'Server is shutting down'})
            time.sleep(0.5)  # Give clients time to receive the message
        except Exception as e:
            print(f"Error notifying clients: {e}")
        
        print("✓ GCS server shutdown complete")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='GCS Server for TAIP System')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=3000, help='Port to bind to (default: 3000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    server = GCSServer(host=args.host, port=args.port)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n[SIGNAL] Received signal {signum}")
        server.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)   # Handle CTRL+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle kill command
    
    try:
        server.run(debug=args.debug)
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        traceback.print_exc()
        server.shutdown()
        sys.exit(1)

if __name__ == '__main__':
    main()