"""
OAK-D Lite camera interface module for the TAIP subsystem.
Handles camera initialization, DepthAI pipeline setup, and neural network inference.
"""

import cv2
import depthai as dai
import numpy as np
import threading
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from queue import Queue
import config
from data_models import YoloDetection, parse_yolo_detection


class OakCamera:
    """
    OAK-D Lite camera interface with YOLO neural network inference.
    
    This class encapsulates all DepthAI pipeline setup and management,
    providing thread-safe access to camera frames and detection results.
    """
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the OAK camera with YOLO model.
        
        Args:
            model_path: Path to the .blob model file
            config_path: Path to the model configuration JSON
        """
        # Configuration
        self.model_path = Path(model_path or config.MODEL_BLOB_PATH)
        self.config_path = Path(config_path or config.MODEL_CONFIG_PATH)
        
        # DepthAI components
        self.device: Optional[dai.Device] = None
        self.pipeline: Optional[dai.Pipeline] = None
        self.q_rgb: Optional[dai.DataOutputQueue] = None
        self.q_detections: Optional[dai.DataOutputQueue] = None
        
        # Threading
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._detection_lock = threading.Lock()
        
        # Latest data (thread-safe storage)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: List[YoloDetection] = []
        self._frame_timestamp: float = 0.0
        self._detection_timestamp: float = 0.0
        
        # Statistics
        self._frames_processed = 0
        self._detections_processed = 0
        self._last_error: Optional[str] = None
        
        # Model configuration
        self._class_names: Dict[int, str] = {}
        self._model_loaded = False
        
        # Initialize the system
        self._load_model_config()
        self._setup_pipeline()
    
    def _load_model_config(self) -> None:
        """Load model configuration and class names."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Extract class names from config
                nn_config = config_data.get("nn_config", {})
                metadata = nn_config.get("NN_specific_metadata", {})
                class_names = metadata.get("classes", [])
                
                # Create class name mapping
                self._class_names = {i: name for i, name in enumerate(class_names)}
                print(f"Loaded {len(self._class_names)} class names from config")
                
            else:
                # Fallback to configured class names
                print(f"Warning: Config file not found at {self.config_path}")
                print("Using fallback class names from config.py")
                self._class_names = {v: k for k, v in config.YOLO_CLASSES.items()}
                
        except Exception as e:
            self._last_error = f"Failed to load model config: {e}"
            print(f"Error: {self._last_error}")
            # Use fallback class names
            self._class_names = {v: k for k, v in config.YOLO_CLASSES.items()}
    
    def _setup_pipeline(self) -> None:
        """Set up the DepthAI pipeline for camera and neural network."""
        try:
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Create RGB camera node
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(*config.CAMERA_PREVIEW_SIZE)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam_rgb.setFps(config.CAMERA_FPS)
            
            # Create YOLO detection network
            detection_nn = self.pipeline.create(dai.node.YoloDetectionNetwork)
            detection_nn.setConfidenceThreshold(config.CONFIDENCE_THRESHOLD)
            detection_nn.setNumClasses(len(self._class_names))
            detection_nn.setCoordinateSize(4)
            detection_nn.setAnchors([])  # YOLOv5 doesn't use anchors in the same way
            detection_nn.setAnchorMasks({})
            detection_nn.setIouThreshold(config.IOU_THRESHOLD)
            
            # Set blob path
            if self.model_path.exists():
                detection_nn.setBlobPath(str(self.model_path))
                self._model_loaded = True
                print(f"Loaded model from {self.model_path}")
            else:
                # Try backup path
                backup_path = Path(config.BACKUP_BLOB_PATH)
                if backup_path.exists():
                    detection_nn.setBlobPath(str(backup_path))
                    self._model_loaded = True
                    print(f"Loaded backup model from {backup_path}")
                else:
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Link camera to neural network
            cam_rgb.preview.link(detection_nn.input)
            
            # Create output queues
            rgb_out = self.pipeline.create(dai.node.XLinkOut)
            rgb_out.setStreamName("rgb")
            cam_rgb.preview.link(rgb_out.input)
            
            detection_out = self.pipeline.create(dai.node.XLinkOut)
            detection_out.setStreamName("detections")
            detection_nn.out.link(detection_out.input)
            
            print("DepthAI pipeline setup completed successfully")
            
        except Exception as e:
            self._last_error = f"Pipeline setup failed: {e}"
            print(f"Error: {self._last_error}")
            raise
    
    def start(self) -> bool:
        """
        Start the camera and begin capture thread.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self._running:
                print("Warning: Camera is already running")
                return True
            
            # Connect to device and start pipeline
            self.device = dai.Device(self.pipeline)
            
            # Get output queues
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=config.CAMERA_QUEUE_SIZE, blocking=False)
            self.q_detections = self.device.getOutputQueue(name="detections", maxSize=config.CAMERA_QUEUE_SIZE, blocking=False)
            
            # Start capture thread
            self._running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            
            print("OAK camera started successfully")
            return True
            
        except Exception as e:
            self._last_error = f"Failed to start camera: {e}"
            print(f"Error: {self._last_error}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop the camera and cleanup resources."""
        print("Stopping OAK camera...")
        
        # Stop capture thread
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        # Close device
        if self.device:
            try:
                self.device.close()
            except Exception as e:
                print(f"Warning: Error closing device: {e}")
            finally:
                self.device = None
        
        # Clear queues
        self.q_rgb = None
        self.q_detections = None
        
        print("OAK camera stopped")
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        print("Starting capture loop...")
        
        while self._running:
            try:
                # Get latest frame
                if self.q_rgb:
                    in_rgb = self.q_rgb.tryGet()
                    if in_rgb is not None:
                        # Convert to OpenCV format
                        frame = in_rgb.getCvFrame()
                        
                        # Store frame thread-safely
                        with self._frame_lock:
                            self._latest_frame = frame.copy()
                            self._frame_timestamp = time.time()
                            self._frames_processed += 1
                
                # Get latest detections
                if self.q_detections:
                    in_detections = self.q_detections.tryGet()
                    if in_detections is not None:
                        # Parse detections
                        detections = self._parse_detections(in_detections.detections)
                        
                        # Store detections thread-safely
                        with self._detection_lock:
                            self._latest_detections = detections
                            self._detection_timestamp = time.time()
                            self._detections_processed += 1
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                self._last_error = f"Capture loop error: {e}"
                print(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Longer sleep on error
    
    def _parse_detections(self, raw_detections) -> List[YoloDetection]:
        """
        Parse raw DepthAI detections into YoloDetection objects.
        
        Args:
            raw_detections: Raw detections from DepthAI
            
        Returns:
            List of parsed YoloDetection objects
        """
        parsed_detections = []
        
        try:
            for detection in raw_detections:
                # Convert DepthAI detection to dictionary format
                detection_dict = {
                    'label': detection.label,
                    'confidence': detection.confidence,
                    'xmin': detection.xmin,
                    'ymin': detection.ymin,
                    'xmax': detection.xmax,
                    'ymax': detection.ymax
                }
                
                # Parse using utility function
                yolo_detection = parse_yolo_detection(detection_dict, self._class_names)
                if yolo_detection and yolo_detection.confidence >= config.CONFIDENCE_THRESHOLD:
                    parsed_detections.append(yolo_detection)
                    
        except Exception as e:
            print(f"Warning: Failed to parse detections: {e}")
        
        return parsed_detections
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.
        
        Returns:
            Latest frame as numpy array, or None if no frame available
        """
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None
    
    def get_latest_detections(self) -> List[YoloDetection]:
        """
        Get the latest YOLO detections.
        
        Returns:
            List of latest detections
        """
        with self._detection_lock:
            return self._latest_detections.copy()
    
    def get_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the latest frame with its timestamp.
        
        Returns:
            Tuple of (frame, timestamp)
        """
        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            return frame, self._frame_timestamp
    
    def get_detections_with_timestamp(self) -> Tuple[List[YoloDetection], float]:
        """
        Get the latest detections with timestamp.
        
        Returns:
            Tuple of (detections, timestamp)
        """
        with self._detection_lock:
            return self._latest_detections.copy(), self._detection_timestamp
    
    def is_running(self) -> bool:
        """Check if the camera is currently running."""
        return self._running and self.device is not None
    
    def is_connected(self) -> bool:
        """Check if the camera is connected."""
        return self.device is not None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get camera and processing statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            'frames_processed': self._frames_processed,
            'detections_processed': self._detections_processed,
            'model_loaded': self._model_loaded,
            'is_running': self.is_running(),
            'is_connected': self.is_connected(),
            'class_count': len(self._class_names),
            'last_error': self._last_error,
            'frame_timestamp': self._frame_timestamp,
            'detection_timestamp': self._detection_timestamp
        }
    
    def get_class_names(self) -> Dict[int, str]:
        """Get the loaded class names mapping."""
        return self._class_names.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._frames_processed = 0
        self._detections_processed = 0
        self._last_error = None
    
    def capture_single_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Capture a single frame with timeout.
        
        Args:
            timeout: Maximum time to wait for frame (seconds)
            
        Returns:
            Captured frame or None if timeout
        """
        if not self.is_running():
            return None
        
        start_time = time.time()
        initial_count = self._frames_processed
        
        while (time.time() - start_time) < timeout:
            if self._frames_processed > initial_count:
                return self.get_latest_frame()
            time.sleep(0.01)  # 10ms polling
        
        return None
    
    # ==========================================================================
    # TEST MODE METHODS
    # ==========================================================================
    
    def _setup_test_pipeline(self) -> None:
        """Set up DepthAI pipeline for test mode (file input instead of camera)."""
        try:
            # Create pipeline for test mode
            self.pipeline = dai.Pipeline()
            
            # Create XLinkIn node for frame input
            xinFrame = self.pipeline.create(dai.node.XLinkIn)
            xinFrame.setStreamName("inFrame")
            
            # Create YOLO detection network (same as live mode)
            detection_nn = self.pipeline.create(dai.node.YoloDetectionNetwork)
            detection_nn.setConfidenceThreshold(config.CONFIDENCE_THRESHOLD)
            detection_nn.setNumClasses(len(self._class_names))
            detection_nn.setCoordinateSize(4)
            detection_nn.setAnchors([])
            detection_nn.setAnchorMasks({})
            detection_nn.setIouThreshold(config.IOU_THRESHOLD)
            
            # Set blob path
            if self.model_path.exists():
                detection_nn.setBlobPath(str(self.model_path))
                self._model_loaded = True
                print(f"Loaded model for test mode from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Link input to neural network
            xinFrame.out.link(detection_nn.input)
            
            # Create output queue for detections
            detection_out = self.pipeline.create(dai.node.XLinkOut)
            detection_out.setStreamName("detections")
            detection_nn.out.link(detection_out.input)
            
            print("DepthAI test mode pipeline setup completed successfully")
            
        except Exception as e:
            self._last_error = f"Test pipeline setup failed: {e}"
            print(f"Error: {self._last_error}")
            raise
    
    def start_test_mode(self) -> bool:
        """
        Start the camera in test mode for processing individual frames.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self._running:
                print("Warning: Camera is already running")
                return True
            
            # Setup test pipeline
            self._setup_test_pipeline()
            
            # Connect to device and start pipeline
            self.device = dai.Device(self.pipeline)
            
            # Get queues for test mode
            self.q_input = self.device.getInputQueue("inFrame")
            self.q_detections = self.device.getOutputQueue("detections", maxSize=4, blocking=False)
            
            self._running = True
            print("OAK camera started in test mode successfully")
            return True
            
        except Exception as e:
            self._last_error = f"Failed to start camera in test mode: {e}"
            print(f"Error: {self._last_error}")
            self.stop()
            return False
    
    def process_test_frame(self, frame: np.ndarray) -> List[YoloDetection]:
        """
        Process a single frame in test mode and return detections.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of YOLO detections
        """
        if not self._running or not hasattr(self, 'q_input'):
            print("Warning: Camera not started in test mode")
            return []
        
        try:
            # Convert frame to planar format for DepthAI
            frame_resized = cv2.resize(frame, config.CAMERA_PREVIEW_SIZE)
            
            # Convert to planar format (CHW)
            frame_planar = frame_resized.transpose(2, 0, 1).flatten()
            
            # Create DepthAI frame
            img_frame = dai.ImgFrame()
            img_frame.setData(frame_planar)
            img_frame.setType(dai.ImgFrame.Type.BGR888p)
            img_frame.setWidth(config.CAMERA_PREVIEW_SIZE[0])
            img_frame.setHeight(config.CAMERA_PREVIEW_SIZE[1])
            
            # Send frame for processing
            self.q_input.send(img_frame)
            
            # Get detections (with timeout)
            in_det = self.q_detections.get()  # Blocking call
            
            if in_det is not None:
                detections = self._parse_detections(in_det.detections)
                return detections
            else:
                return []
                
        except Exception as e:
            self._last_error = f"Test frame processing failed: {e}"
            print(f"Error processing test frame: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        if not self.start():
            raise RuntimeError("Failed to start OAK camera")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_oak_devices() -> List[Dict[str, str]]:
    """
    List all available OAK devices.
    
    Returns:
        List of device information dictionaries
    """
    try:
        devices = dai.Device.getAllAvailableDevices()
        device_list = []
        
        for device_info in devices:
            device_dict = {
                'name': device_info.name,
                'mxid': device_info.getMxId(),
                'state': device_info.state.name,
                'protocol': device_info.protocol.name
            }
            device_list.append(device_dict)
        
        return device_list
        
    except Exception as e:
        print(f"Error listing OAK devices: {e}")
        return []


def test_camera_connection() -> bool:
    """
    Test if an OAK camera can be connected.
    
    Returns:
        True if camera can be connected, False otherwise
    """
    try:
        # Try to create a simple pipeline and connect
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(320, 240)
        
        with dai.Device(pipeline) as device:
            print("✓ OAK camera connection test passed")
            return True
            
    except Exception as e:
        print(f"✗ OAK camera connection test failed: {e}")
        return False


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("Testing OAK camera module...")
    
    # Test device listing
    devices = list_oak_devices()
    print(f"Found {len(devices)} OAK devices:")
    for device in devices:
        print(f"  - {device['name']} ({device['mxid']})")
    
    # Test camera connection
    if test_camera_connection():
        print("Camera connection test passed")
    else:
        print("Camera connection test failed - check hardware connection")
        exit(1)
    
    # Test camera initialization and capture
    try:
        print("Testing camera initialization...")
        camera = OakCamera()
        
        if camera.start():
            print("✓ Camera started successfully")
            
            # Wait for some frames
            time.sleep(2)
            
            # Get statistics
            stats = camera.get_statistics()
            print(f"Statistics: {stats}")
            
            # Test frame capture
            frame = camera.get_latest_frame()
            if frame is not None:
                print(f"✓ Frame captured: {frame.shape}")
            else:
                print("✗ No frame captured")
            
            # Test detection capture
            detections = camera.get_latest_detections()
            print(f"✓ Detections captured: {len(detections)}")
            
            camera.stop()
            print("✓ Camera stopped successfully")
            
        else:
            print("✗ Failed to start camera")
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
    
    print("OAK camera module testing completed.")
