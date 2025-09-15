# /home/pi/EGH455/TAIP/oak_camera.py

"""
OAK-D Lite Camera and Inference Module for EGH455 TAIP Subsystem

This module encapsulates all interactions with the Luxonis OAK-D Lite camera.
It sets up a single DepthAI pipeline for both RGB video streaming and YOLOv8s
neural network inference. A dedicated background thread continuously fetches
data from the device, ensuring the main application loop is non-blocking.
It provides synchronised RGB frames, monochrome frames (for ArUco), and YOLO detections.
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
            print(f"FATAL: Failed to initialise OAK-D Lite device: {e}")
            raise

    def _load_config(self):
        """Loads model configuration from the JSON file."""
        try:
            with open(config.CONFIG_PATH, 'r') as f:
                self.model_config = json.load(f)
            self.class_names = self.model_config["mappings"]["labels"]
            self.model_input_size = tuple(map(int, self.model_config["nn_config"]["input_size"].split('x')))
            if not self.class_names or not self.model_input_size: 
                raise ValueError("Config invalid")
        except Exception as e:
            print(f"FATAL: Could not load model config at {config.CONFIG_PATH}: {e}")
            raise

    def _create_pipeline(self) -> dai.Pipeline:
        """Builds the DepthAI pipeline with RGB, Mono, and YOLO nodes."""
        pipeline = dai.Pipeline()

        cam_rgb = pipeline.createColorCamera()
        mono_cam = pipeline.createMonoCamera()
        yolo_net = pipeline.createYoloDetectionNetwork()
        xout_rgb = pipeline.createXLinkOut()
        xout_mono = pipeline.createXLinkOut()
        xout_nn = pipeline.createXLinkOut()

        xout_rgb.setStreamName("rgb")
        xout_mono.setStreamName("mono")
        xout_nn.setStreamName("nn")

        # RGB camera → YOLO input + passthrough for visualisation
        cam_rgb.setPreviewSize(self.model_input_size)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(config.CAMERA_FPS)
        cam_rgb.preview.link(yolo_net.input)
        yolo_net.passthrough.link(xout_rgb.input)

        # Mono camera for ArUco
        mono_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono_cam.setFps(config.CAMERA_FPS)
        mono_cam.out.link(xout_mono.input)

        # On-device YOLOv8 inference
        nn_meta = self.model_config['nn_config']['NN_specific_metadata']
        yolo_net.setBlobPath(str(config.BLOB_PATH))
        yolo_net.setConfidenceThreshold(config.CONFIDENCE_THRESHOLD)
        yolo_net.setIouThreshold(config.IOU_THRESHOLD)
        yolo_net.setNumClasses(nn_meta['classes'])
        yolo_net.setCoordinateSize(nn_meta['coordinates'])
        yolo_net.setNumInferenceThreads(2)
        yolo_net.out.link(xout_nn.input)

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

