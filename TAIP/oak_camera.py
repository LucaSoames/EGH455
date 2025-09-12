# /home/pi/EGH455/TAIP/oak_camera.py

"""
OAK-D Lite Camera and Inference Module for EGH455 TAIP Subsystem

This module encapsulates all interactions with the Luxonis OAK-D Lite camera.
It sets up a single DepthAI pipeline for both RGB video streaming and YOLOv8s
neural network inference. A dedicated background thread continuously fetches
data from the device, ensuring the main application loop is non-blocking.
It provides synchronised RGB frames, monochrome frames (for ArUco), and YOLO detections. 
It also includes a MockCamera class to enable file-based testing.
"""

import depthai as dai
import numpy as np
import threading
import json
import time
import cv2
from pathlib import Path
from typing import List, Tuple, Optional

import config
from data_models import YoloDetection

class OakCamera:
    """Manages the OAK-D Lite camera and the YOLOv8s neural network pipeline."""

    def __init__(self):
        self.latest_rgb_frame: Optional[np.ndarray] = None
        self.latest_mono_frame: Optional[np.ndarray] = None
        self.latest_detections: List[YoloDetection] = []
        self._load_config()
        self.pipeline = self._create_pipeline()
        self._lock = threading.Lock()
        self._running = False
        try:
            self.device = dai.Device(self.pipeline)
            print("OAK-D Lite connected successfully.")
            self._rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self._mono_queue = self.device.getOutputQueue(name="mono", maxSize=4, blocking=False)
            self._nn_queue = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)
            self._running = True
            self._thread = threading.Thread(target=self._thread_loop, daemon=True)
            self._thread.start()
            print("Camera and inference thread started.")
        except Exception as e:
            print(f"FATAL: Failed to initialize OAK-D Lite device: {e}")
            raise

    def _load_config(self):
        """Loads model configuration from the JSON file."""
        try:
            with open(config.CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            self.class_names = cfg["mappings"]["labels"]
            self.model_input_size = tuple(map(int, cfg["nn_config"]["input_size"].split('x')))
            if not self.class_names or not self.model_input_size: raise ValueError("Config invalid")
        except Exception as e:
            print(f"FATAL: Could not load model config at {config.CONFIG_PATH}: {e}")
            raise

    def _create_pipeline(self) -> dai.Pipeline:
        """Builds the DepthAI pipeline with RGB, Mono, and YOLO nodes."""
        pipeline = dai.Pipeline()

        # --- Nodes ---
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        mono_cam = pipeline.create(dai.node.MonoCamera)
        detection_network = pipeline.create(dai.node.YoloDetectionNetwork)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_mono = pipeline.create(dai.node.XLinkOut)
        xout_nn = pipeline.create(dai.node.XLinkOut)
        
        xout_rgb.setStreamName("rgb")
        xout_mono.setStreamName("mono")
        xout_nn.setStreamName("nn")

        # --- Properties ---
        cam_rgb.setPreviewSize(self.model_input_size)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        mono_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B) # Left stereo camera
        mono_cam.setFps(30)

        # --- YOLO Network Configuration (loaded from JSON) ---
        nn_meta = self.nn_config['nn_config']['NN_specific_metadata']
        detection_network.setBlobPath(config.BLOB_PATH)
        detection_network.setConfidenceThreshold(nn_meta['confidence_threshold'])
        detection_network.setNumClasses(nn_meta['classes'])
        detection_network.setCoordinateSize(nn_meta['coordinates'])
        detection_network.setIouThreshold(nn_meta['iou_threshold'])

        # --- Linking ---
        cam_rgb.preview.link(detection_network.input)
        mono_cam.out.link(xout_mono.input)
        detection_network.passthrough.link(xout_rgb.input)
        detection_network.out.link(xout_nn.input)
        
        return pipeline

    def _thread_loop(self):
        """The main loop for the background thread to fetch data from the camera."""
        while self._running:
            # Get the raw data from the queues
            in_rgb = self._rgb_queue.tryGet()
            in_mono = self._mono_queue.tryGet()
            in_nn = self._nn_queue.tryGet()

            new_rgb = None
            new_mono = None
            new_detections = []

            if in_rgb: 
                new_rgb = in_rgb.getCvFrame()
            if in_mono: 
                new_mono = in_mono.getCvFrame()
            if in_nn:
                new_detections = [YoloDetection(self.class_names[d.label], d.confidence, (d.xmin, d.ymin, d.xmax, d.ymax)) for d in in_nn.detections]
            
            # Atomically update the shared state
            with self._lock:
                if new_rgb is not None: 
                    self.latest_rgb_frame = new_rgb
                if new_mono is not None: 
                    self.latest_mono_frame = new_mono
                # Always update detections, even if empty, to clear old ones
                self.latest_detections = new_detections
            
            # Yield CPU to other processes
            time.sleep(0.001)
            
            
    def get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest RGB frame from the camera.

        Returns:
            A NumPy array of the RGB frame, or None if no frame is available.
        """
        with self._lock: 
            return self.latest_rgb_frame.copy() if self.latest_rgb_frame is not None else None
        
    def get_latest_mono_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest monochrome frame from the left camera.

        Returns:
            A NumPy array of the monochrome frame, or None if no frame is available.
        """
        with self._lock: 
            return self.latest_mono_frame.copy() if self.latest_mono_frame is not None else None

    def get_latest_detections(self) -> List[YoloDetection]:
        """
        Get the latest list of detected objects.

        Returns:
            A list of YoloDetection dataclass objects.
        """
        with self._lock:
            return list(self.latest_detections)

    def close(self):
        """Stops the thread and closes the device to release resources."""
        print("Closing OAK-D Lite...")
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=2.0)
        if hasattr(self, 'device'):
            self.device.close()
        print("OAK-D Lite closed.")

class MockCamera:
    """A mock camera class for testing the system with local image/video files."""
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.files = []
        self.cap = None
        self.model_input_size = (640, 640)
        try:
            with open(config.CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            self.class_names = cfg["mappings"]["labels"]
        except Exception:
            self.class_names = [f"class_{i}" for i in range(80)]
        
        if self.input_path.is_dir():
            self.files = sorted([p for p in self.input_path.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        elif self.input_path.is_file():
            self.cap = cv2.VideoCapture(str(self.input_path))
        print(f"MockCamera initialized with path: {input_path}")
    
    def get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        if self.files:
            if not hasattr(self, 'image_idx') or self.image_idx >= len(self.files): return None
            frame = cv2.imread(str(self.files[self.image_idx]))
            self.image_idx += 1
            return frame
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def get_latest_mono_frame(self) -> Optional[np.ndarray]:
        # In test mode, we simulate the mono frame by converting the RGB frame
        frame = self.get_latest_rgb_frame()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
    
    def get_latest_detections(self) -> List[YoloDetection]:
        # This mock doesn't run inference; it will be done in the main testing loop.
        return []

    def close(self):
        if self.cap: 
            self.cap.release()

