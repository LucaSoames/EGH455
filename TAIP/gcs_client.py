"""
Ground Control Station (GCS) client module for the TAIP subsystem.
Handles all communication with the GCS including JSON telemetry and video frame transmission.
"""

import requests
import cv2
import numpy as np
import json
import time
import threading
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
import config
from data_models import PayloadData


class GCSClient:
    """
    Client for communicating with the Ground Control Station.
    
    Handles asynchronous transmission of telemetry data and video frames
    to the GCS server with retry logic and error handling.
    """
    
    def __init__(self, base_url: str = None, max_workers: int = 3):
        """
        Initialize the GCS client.
        
        Args:
            base_url: Base URL of the GCS server
            max_workers: Maximum number of worker threads for async requests
        """
        self.base_url = base_url or config.GCS_BASE_URL
        self.telemetry_url = self.base_url + config.GCS_TELEMETRY_ENDPOINT
        self.frame_url = self.base_url + config.GCS_FRAME_ENDPOINT
        
        # Threading and async handling
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_requests: Dict[str, Future] = {}
        
        # Connection state
        self._connected = False
        self._last_connection_test = 0.0
        self._connection_test_interval = 30.0  # Test connection every 30s
        
        # Statistics
        self._telemetry_sent = 0
        self._frames_sent = 0
        self._failed_requests = 0
        self._last_error: Optional[str] = None
        self._total_bytes_sent = 0
        
        # Request session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TAIP-Subsystem/1.0',
            'Content-Type': 'application/json'
        })
        
        # Retry configuration
        self.max_retries = config.MAX_RETRIES
        self.retry_delay = config.RETRY_DELAY
        
        print(f"GCS Client initialized - Target: {self.base_url}")
    
    def test_connection(self, timeout: float = None) -> bool:
        """
        Test connection to the GCS server.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            True if connection successful, False otherwise
        """
        timeout = timeout or config.CONNECTION_TIMEOUT
        
        try:
            # Try to reach the base URL with a simple GET request
            response = self.session.get(
                self.base_url + "/health",  # Assume health endpoint exists
                timeout=timeout
            )
            
            if response.status_code == 200:
                self._connected = True
                self._last_connection_test = time.time()
                return True
            else:
                self._connected = False
                self._last_error = f"Health check failed: HTTP {response.status_code}"
                return False
                
        except requests.exceptions.RequestException as e:
            self._connected = False
            self._last_error = f"Connection test failed: {e}"
            return False
    
    def is_connected(self) -> bool:
        """
        Check if the client is connected to GCS.
        Performs periodic connection tests.
        
        Returns:
            True if connected, False otherwise
        """
        current_time = time.time()
        
        # Perform periodic connection test
        if (current_time - self._last_connection_test) > self._connection_test_interval:
            self.test_connection()
        
        return self._connected
    
    def send_data(self, payload_data: PayloadData, 
                 callback: Optional[Callable[[bool, str], None]] = None) -> Optional[Future]:
        """
        Send telemetry data to GCS asynchronously.
        
        Args:
            payload_data: PayloadData object to send
            callback: Optional callback function called with (success, message)
            
        Returns:
            Future object for the async request, or None if immediate failure
        """
        try:
            # Validate payload data
            payload_data.validate()
            
            # Convert to JSON
            json_data = payload_data.to_json()
            
            # Submit async request
            future = self.executor.submit(
                self._send_telemetry_sync, 
                json_data, 
                callback
            )
            
            # Store future for tracking
            request_id = f"telemetry_{int(time.time() * 1000)}"
            self.pending_requests[request_id] = future
            
            return future
            
        except Exception as e:
            self._last_error = f"Failed to submit telemetry request: {e}"
            print(f"Error: {self._last_error}")
            if callback:
                callback(False, self._last_error)
            return None
    
    def send_frame(self, frame: np.ndarray, 
                  callback: Optional[Callable[[bool, str], None]] = None) -> Optional[Future]:
        """
        Send video frame to GCS asynchronously.
        
        Args:
            frame: OpenCV frame (numpy array)
            callback: Optional callback function called with (success, message)
            
        Returns:
            Future object for the async request, or None if immediate failure
        """
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame data")
            
            # Resize frame if needed
            if frame.shape[1] > config.VIDEO_MAX_WIDTH or frame.shape[0] > config.VIDEO_MAX_HEIGHT:
                # Calculate new dimensions maintaining aspect ratio
                height, width = frame.shape[:2]
                if width > height:
                    new_width = config.VIDEO_MAX_WIDTH
                    new_height = int(height * (new_width / width))
                else:
                    new_height = config.VIDEO_MAX_HEIGHT
                    new_width = int(width * (new_height / height))
                
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode frame as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.VIDEO_QUALITY]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                raise ValueError("Failed to encode frame as JPEG")
            
            # Convert to bytes
            frame_bytes = buffer.tobytes()
            
            # Submit async request
            future = self.executor.submit(
                self._send_frame_sync, 
                frame_bytes, 
                callback
            )
            
            # Store future for tracking
            request_id = f"frame_{int(time.time() * 1000)}"
            self.pending_requests[request_id] = future
            
            return future
            
        except Exception as e:
            self._last_error = f"Failed to submit frame request: {e}"
            print(f"Error: {self._last_error}")
            if callback:
                callback(False, self._last_error)
            return None
    
    def _send_telemetry_sync(self, json_data: str, 
                           callback: Optional[Callable[[bool, str], None]] = None) -> bool:
        """
        Send telemetry data synchronously with retry logic.
        
        Args:
            json_data: JSON string to send
            callback: Optional callback function
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.telemetry_url,
                    data=json_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=config.REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    self._telemetry_sent += 1
                    self._total_bytes_sent += len(json_data)
                    if callback:
                        callback(True, "Telemetry sent successfully")
                    return True
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt == self.max_retries:
                        self._failed_requests += 1
                        self._last_error = error_msg
                        if callback:
                            callback(False, error_msg)
                    else:
                        print(f"Telemetry attempt {attempt + 1} failed: {error_msg}")
                        time.sleep(self.retry_delay)
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {e}"
                if attempt == self.max_retries:
                    self._failed_requests += 1
                    self._last_error = error_msg
                    if callback:
                        callback(False, error_msg)
                else:
                    print(f"Telemetry attempt {attempt + 1} failed: {error_msg}")
                    time.sleep(self.retry_delay)
        
        return False
    
    def _send_frame_sync(self, frame_bytes: bytes, 
                        callback: Optional[Callable[[bool, str], None]] = None) -> bool:
        """
        Send frame data synchronously with retry logic.
        
        Args:
            frame_bytes: Frame data as bytes
            callback: Optional callback function
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.frame_url,
                    data=frame_bytes,
                    headers={'Content-Type': 'image/jpeg'},
                    timeout=config.REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    self._frames_sent += 1
                    self._total_bytes_sent += len(frame_bytes)
                    if callback:
                        callback(True, "Frame sent successfully")
                    return True
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt == self.max_retries:
                        self._failed_requests += 1
                        self._last_error = error_msg
                        if callback:
                            callback(False, error_msg)
                    else:
                        print(f"Frame attempt {attempt + 1} failed: {error_msg}")
                        time.sleep(self.retry_delay)
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {e}"
                if attempt == self.max_retries:
                    self._failed_requests += 1
                    self._last_error = error_msg
                    if callback:
                        callback(False, error_msg)
                else:
                    print(f"Frame attempt {attempt + 1} failed: {error_msg}")
                    time.sleep(self.retry_delay)
        
        return False
    
    def wait_for_pending_requests(self, timeout: float = 5.0) -> None:
        """
        Wait for all pending requests to complete.
        
        Args:
            timeout: Maximum time to wait for completion
        """
        start_time = time.time()
        
        while self.pending_requests and (time.time() - start_time) < timeout:
            # Check for completed requests
            completed_keys = []
            for key, future in self.pending_requests.items():
                if future.done():
                    completed_keys.append(key)
            
            # Remove completed requests
            for key in completed_keys:
                del self.pending_requests[key]
            
            if self.pending_requests:
                time.sleep(0.1)  # Wait 100ms before checking again
    
    def cleanup_completed_requests(self) -> None:
        """Remove completed requests from the pending list."""
        completed_keys = [key for key, future in self.pending_requests.items() if future.done()]
        for key in completed_keys:
            del self.pending_requests[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with transmission statistics
        """
        return {
            'connected': self._connected,
            'telemetry_sent': self._telemetry_sent,
            'frames_sent': self._frames_sent,
            'failed_requests': self._failed_requests,
            'total_bytes_sent': self._total_bytes_sent,
            'pending_requests': len(self.pending_requests),
            'last_error': self._last_error,
            'base_url': self.base_url
        }
    
    def reset_statistics(self) -> None:
        """Reset transmission statistics."""
        self._telemetry_sent = 0
        self._frames_sent = 0
        self._failed_requests = 0
        self._total_bytes_sent = 0
        self._last_error = None
    
    def shutdown(self) -> None:
        """Shutdown the client and cleanup resources."""
        print("Shutting down GCS client...")
        
        # Wait for pending requests
        self.wait_for_pending_requests(timeout=2.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close session
        self.session.close()
        
        print("GCS client shutdown completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_test_request(gcs_url: str, endpoint: str, data: Dict[str, Any] = None) -> bool:
    """
    Test a specific GCS endpoint.
    
    Args:
        gcs_url: Base GCS URL
        endpoint: Endpoint path to test
        data: Optional data to send
        
    Returns:
        True if successful, False otherwise
    """
    try:
        url = gcs_url + endpoint
        
        if data:
            response = requests.post(
                url, 
                json=data, 
                timeout=config.CONNECTION_TIMEOUT
            )
        else:
            response = requests.get(url, timeout=config.CONNECTION_TIMEOUT)
        
        print(f"Test request to {url}: HTTP {response.status_code}")
        return response.status_code == 200
        
    except requests.exceptions.RequestException as e:
        print(f"Test request failed: {e}")
        return False


def validate_gcs_endpoints(gcs_url: str) -> Dict[str, bool]:
    """
    Validate all required GCS endpoints.
    
    Args:
        gcs_url: Base GCS URL
        
    Returns:
        Dictionary with endpoint validation results
    """
    results = {}
    
    # Test health endpoint
    results['health'] = create_test_request(gcs_url, "/health")
    
    # Test telemetry endpoint
    test_data = {"test": "data", "timestamp": "2023-01-01 00:00:00.000"}
    results['telemetry'] = create_test_request(
        gcs_url, 
        config.GCS_TELEMETRY_ENDPOINT, 
        test_data
    )
    
    # Test frame endpoint (would need actual image data)
    results['frame'] = create_test_request(gcs_url, config.GCS_FRAME_ENDPOINT)
    
    return results


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("Testing GCS client module...")
    
    # Test endpoint validation
    print(f"Validating GCS endpoints at {config.GCS_BASE_URL}...")
    endpoint_results = validate_gcs_endpoints(config.GCS_BASE_URL)
    
    for endpoint, result in endpoint_results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {endpoint}: {'OK' if result else 'FAILED'}")
    
    # Test client initialization
    try:
        with GCSClient() as client:
            print("✓ GCS client initialized successfully")
            
            # Test connection
            if client.test_connection():
                print("✓ GCS connection test passed")
            else:
                print("✗ GCS connection test failed")
            
            # Test with dummy data
            from data_models import create_test_payload
            
            test_payload = create_test_payload()
            print("Sending test telemetry...")
            
            # Send test data (non-blocking)
            future = client.send_data(test_payload)
            if future:
                print("✓ Test telemetry submitted")
            else:
                print("✗ Failed to submit test telemetry")
            
            # Test with dummy frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_frame, "TEST FRAME", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            print("Sending test frame...")
            frame_future = client.send_frame(test_frame)
            if frame_future:
                print("✓ Test frame submitted")
            else:
                print("✗ Failed to submit test frame")
            
            # Wait for completion
            client.wait_for_pending_requests()
            
            # Get statistics
            stats = client.get_statistics()
            print(f"Client statistics: {stats}")
            
    except Exception as e:
        print(f"✗ GCS client test failed: {e}")
    
    print("GCS client module testing completed.")
