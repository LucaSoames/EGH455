"""
File Processing Module for the EGH455 TAIP Subsystem

This module combines file reading and inference capabilities for test mode.
It handles video files, image directories, and runs neural network inference
using the same DepthAI pipeline as the live camera mode.
"""

import cv2
import depthai as dai
import json
import numpy as np
from pathlib import Path
from typing import Optional, List

import config
from data_models import YoloDetection

class FileProcessor:
    """
    Combined file processor and inference engine for test mode.
    Handles both file/image loading and on-device neural network inference.
    """
    
    def __init__(self, input_path):
        # File handling attributes
        self.input_path = Path(input_path) if not isinstance(input_path, Path) else input_path
        self.cap = None
        self.image_files = []
        self.current_index = 0
        
        # Verify path exists first
        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {self.input_path}")
        
        # Set up file source
        if self.input_path.is_file():
            # Video file
            self.cap = cv2.VideoCapture(str(self.input_path))
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video file: {self.input_path}")
            print(f"✓ Loaded video: {self.input_path.name}")
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"  Frames: {frame_count}, FPS: {fps:.1f}")
        elif self.input_path.is_dir():
            # Image directory
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            self.image_files = sorted([
                p for p in self.input_path.glob('*') 
                if p.suffix.lower() in extensions
            ])
            if not self.image_files:
                raise ValueError(f"No images found in: {self.input_path}")
            print(f"✓ Loaded {len(self.image_files)} images from {self.input_path.name}")
        else:
            raise ValueError(f"Input path is neither file nor directory: {self.input_path}")
        
        # Set up inference pipeline
        self._load_model_config()
        self.pipeline = self._create_pipeline()
        try:
            self.device = dai.Device(self.pipeline)
            self.q_in = self.device.getInputQueue("host_in")
            self.q_nn = self.device.getOutputQueue("nn_out", maxSize=4, blocking=True)
            print("✓ File processor device connected")
        except Exception as e:
            if self.cap:
                self.cap.release()
            raise RuntimeError(f"Failed to create file processor device: {e}")
    
    def _load_model_config(self):
        """Load neural network configuration from JSON file."""
        try:
            with open(config.CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            self.class_names = cfg["mappings"]["labels"]
            self.model_input = tuple(map(int, cfg["nn_config"]["input_size"].split('x')))  # (W, H)
            self.nn_meta = cfg["nn_config"]["NN_specific_metadata"]
        except Exception as e:
            raise RuntimeError(f"Failed to load model configuration: {e}")

    def _create_pipeline(self):
        """Create a DepthAI pipeline for file-based inference."""
        p = dai.Pipeline()
        xin = p.createXLinkIn()
        nn = p.createYoloDetectionNetwork()
        xout = p.createXLinkOut()

        xin.setStreamName("host_in")
        xout.setStreamName("nn_out")

        # Use on-device decoding via the Myriad X
        nn.setBlobPath(str(config.BLOB_PATH))
        nn.setConfidenceThreshold(config.CONFIDENCE_THRESHOLD)
        nn.setIouThreshold(config.IOU_THRESHOLD)
        nn.setNumClasses(self.nn_meta["classes"])
        nn.setCoordinateSize(self.nn_meta["coordinates"])
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        
        # The neural network input node will automatically resize the frame from the host.
        # This is more efficient than resizing on the Pi's CPU.
        nn.input.setQueueSize(1)

        xin.out.link(nn.input)
        nn.out.link(xout.input)
        return p
    
    def get_next_frame(self) -> Optional[np.ndarray]:
        """Get next frame from video or next image."""
        if self.cap:
            # Video mode
            ret, frame = self.cap.read()
            if ret:
                # Correction for test videos that are read with an incorrect 90-degree rotation.
                return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            return None
        elif self.image_files:
            # Image directory mode
            if self.current_index < len(self.image_files):
                frame = cv2.imread(str(self.image_files[self.current_index]))
                self.current_index += 1
                return frame
            return None
        return None
    
    def process_frame(self, frame) -> List[YoloDetection]:
        """Process a single frame through the neural network."""
        if frame is None:
            return []
        
        # Resize frame to model input size to avoid exceeding XLinkIn buffer
        # The DepthAI device expects frames at the model's input resolution
        frame_resized = cv2.resize(frame, self.model_input)
        
        # Create a DepthAI ImgFrame
        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888p)
        img.setFrame(frame_resized)
        img.setWidth(frame_resized.shape[1])
        img.setHeight(frame_resized.shape[0])
        self.q_in.send(img)

        # Block until results ready
        nn_packet = self.q_nn.get()
        dets = []
        for d in nn_packet.detections:
            if 0 <= d.label < len(self.class_names):
                dets.append(
                    YoloDetection(
                        class_name=self.class_names[d.label],
                        confidence=d.confidence,
                        box=(d.xmin, d.ymin, d.xmax, d.ymax)
                    )
                )
        return dets
    
    @property
    def is_video(self) -> bool:
        """Return True if processing a video file."""
        return self.cap is not None
    
    def close(self):
        """Release all resources."""
        if self.cap:
            self.cap.release()
        
        try:
            if hasattr(self, "device"):
                self.device.close()
                print("DepthAI device for file processing closed")
        except Exception as e:
            print(f"Warning: Error while closing DepthAI device: {e}")