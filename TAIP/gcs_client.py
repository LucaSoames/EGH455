# /home/pi/EGH455/TAIP/gcs_client.py

"""
GCS Client for the EGH455 TAIP Subsystem

This module handles all network communication with the Ground Control Station (GCS).
It uses a non-blocking approach with a thread pool to send telemetry data and
video frames without stalling the main application loop.
"""

import requests
import cv2
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Optional, Any, Callable
import socketio  # Add socketio for receiving commands

import config
from data_models import PayloadData

class GCSClient:
    """Manages network communication with the GCS server."""
    
    def __init__(self, lcd_callback: Optional[Callable[[int], None]] = None):
        """
        Initialises the GCS client.

        Args:
            max_workers: The number of background threads for network requests.
        """
        self.base_url = config.GCS_URL
        self.telemetry_url = f"{self.base_url}/telemetry"
        self.frame_url = f"{self.base_url}/frame"
        
        # Use a session object for connection pooling and performance
        self.session = requests.Session()
        
        # A thread pool to execute network requests asynchronously
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = True
        
        # SocketIO client for receiving commands
        self.sio = socketio.Client(reconnection=True, reconnection_attempts=0)
        self.lcd_callback = lcd_callback
        
        # Setup SocketIO event handlers
        self._setup_socketio_handlers()
        
        # Connect to GCS server
        try:
            self.sio.connect(config.GCS_URL, transports=['websocket', 'polling'])
            print(f"✓ Connected to GCS server for commands: {config.GCS_URL}")
        except Exception as e:
            print(f"Failed to connect to GCS for commands: {e}")

    def _setup_socketio_handlers(self):
        """Setup handlers for incoming SocketIO messages."""
        
        @self.sio.on('lcd_tab_command')
        def handle_lcd_tab_command(data):
            """Handle LCD tab change command from GCS."""
            tab_index = data.get('tab_index')
            if tab_index is not None and self.lcd_callback:
                print(f"Received LCD tab command: {tab_index}")
                self.lcd_callback(tab_index)
        
        @self.sio.on('connect')
        def on_connect():
            print("✓ GCS command channel connected")
        
        @self.sio.on('disconnect')
        def on_disconnect():
            print("✗ GCS command channel disconnected")

    def _send_post_request(self, url: str, **kwargs: Any) -> None:
        """Helper function to send a POST request and handle exceptions."""
        try:
            timeout = kwargs.pop("timeout", config.REQUEST_TIMEOUT)
            self.session.post(url, timeout=timeout, **kwargs)
        except requests.exceptions.RequestException:
            pass

    def send_data(self, payload_data: PayloadData) -> None:
        """
        Asynchronously sends a JSON data packet to the GCS.

        Args:
            payload_data: The PayloadData object to send.
        """
        # Convert dataclass to dictionary, then to JSON string
        json_data = json.dumps(asdict(payload_data))
        headers = {'Content-Type': 'application/json'}
        
        # Submit the network request to the thread pool
        self.executor.submit(self._send_post_request, self.telemetry_url,
                             data=json_data, headers=headers, timeout=config.REQUEST_TIMEOUT)

    def send_frame(self, frame: np.ndarray) -> None:
        """
        Asynchronously encodes and sends a video frame to the GCS.

        Args:
            frame: The video frame (NumPy array) to send.
        """
        if frame is None:
            return
        # Encode the frame as JPEG for efficient transmission
        is_success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not is_success:
            return

        headers = {'Content-Type': 'image/jpeg'}
        
        # Submit the network request to the thread pool
        self.executor.submit(self._send_post_request, self.frame_url,
                             data=buffer.tobytes(), headers=headers, timeout=config.REQUEST_TIMEOUT)

    def shutdown(self):
        """Gracefully shut down the GCS client."""
        self.running = False
        
        # Disconnect SocketIO
        if self.sio.connected:
            self.sio.disconnect()
        
        # Shuts down the thread pool executor
        print("Shutting down GCS client...")
        self.executor.shutdown(wait=True)
        print("GCS client shut down.")

# --- Standalone Test ---
if __name__ == '__main__':
    print("Testing GCS Client...")
    
       
    client = GCSClient()
    
    # Create mock data
    mock_environmental_data = {
        "temperature_c": 25.5,
        "pressure_hpa": 1013.2,
        "humidity_rh": 55.1,
        "light_lux": 500.0
    }
    
    mock_payload = PayloadData(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        gauge_pressure_bar=5.5,
        environmental_data=mock_environmental_data
    )
    
    # Create a mock frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(mock_frame, "GCS Client Test Frame", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    try:
        print("Sending 5 data packets and frames over 5 seconds...")
        for i in range(5):
            mock_payload.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            client.send_data(mock_payload)
            client.send_frame(mock_frame)
            print(f"Sent packet {i+1}")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        client.shutdown()
        print("Test finished.")