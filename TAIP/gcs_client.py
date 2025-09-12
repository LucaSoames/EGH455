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
from typing import Optional, Any

import config
from data_models import PayloadData

class GCSClient:
    """Manages network communication with the GCS server."""

    def __init__(self, max_workers: int = 2):
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
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        print(f"GCS Client initialised for server at {self.base_url}")

    def _send_post_request(self, url: str, **kwargs: Any) -> None:
        """Helper function to send a POST request and handle exceptions."""
        try:
            self.session.post(url, timeout=1.0, **kwargs)
        except requests.exceptions.RequestException as e:
            # Silently handle network errors to avoid crashing the main loop
            # In a production system, this could log to a file.
            # print(f"GCS connection error: {e}")
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
        self.executor.submit(self._send_post_request, self.telemetry_url, data=json_data, headers=headers)

    def send_frame(self, frame: np.ndarray) -> None:
        """
        Asynchronously encodes and sends a video frame to the GCS.

        Args:
            frame: The video frame (NumPy array) to send.
        """
        # Encode the frame as JPEG for efficient transmission
        is_success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not is_success:
            return

        headers = {'Content-Type': 'image/jpeg'}
        
        # Submit the network request to the thread pool
        self.executor.submit(self._send_post_request, self.frame_url, data=buffer.tobytes(), headers=headers)

    def shutdown(self) -> None:
        """Shuts down the thread pool executor."""
        print("Shutting down GCS client...")
        self.executor.shutdown(wait=True)
        print("GCS client shut down.")

# --- Standalone Test ---
if __name__ == '__main__':
    print("Testing GCS Client...")
    
    # Create a mock server URL (replace with a real test server if needed)
    # For this test, you can run the provided `ground_station_server.py`
    config.GCS_URL = "http://127.0.0.1:5000"
    
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