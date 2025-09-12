"""Simple file processor for TAIP input."""

import cv2
from pathlib import Path
from typing import Optional, List
import numpy as np
import config

class FileProcessor:
    """Simple processor for video files and image directories."""
    
    def __init__(self, input_path: Path):
        self.input_path = Path(input_path)
        self.cap = None
        self.image_files = []
        self.current_index = 0
        
        if self.input_path.is_file():
            # Video file
            self.cap = cv2.VideoCapture(str(self.input_path))
            print(f"Loaded video: {self.input_path.name}")
        elif self.input_path.is_dir():
            # Image directory
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            self.image_files = sorted([
                p for p in self.input_path.glob('*') 
                if p.suffix.lower() in extensions
            ])
            print(f"Loaded {len(self.image_files)} images from {self.input_path.name}")
    
    def get_next_frame(self) -> Optional[np.ndarray]:
        """Get next frame from video or next image."""
        if self.cap:
            # Video mode
            ret, frame = self.cap.read()
            if ret:
                return cv2.resize(frame, config.CAMERA_PREVIEW_SIZE)
            return None
        elif self.image_files:
            # Image directory mode
            if self.current_index < len(self.image_files):
                frame = cv2.imread(str(self.image_files[self.current_index]))
                self.current_index += 1
                if frame is not None:
                    return cv2.resize(frame, config.CAMERA_PREVIEW_SIZE)
            return None
        return None
    
    def close(self):
        if self.cap:
            self.cap.release()