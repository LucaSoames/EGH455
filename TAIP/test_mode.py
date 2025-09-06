"""
Test mode module for the TAIP subsystem.
Handles processing of video files and image directories for testing detection algorithms.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, List, Generator, Tuple
import config
from data_models import YoloDetection
from vision_processing import draw_detections_on_frame, detect_aruco_markers, calculate_gauge_reading


class TestModeProcessor:
    """
    Processor for test mode operations on videos and images.
    Replaces live camera feed with file-based input for testing.
    """
    
    def __init__(self, input_path: Optional[Path] = None):
        """
        Initialize test mode processor.
        
        Args:
            input_path: Path to video file or image directory
        """
        self.input_path = Path(input_path) if input_path else None
        self.is_video = False
        self.is_image_dir = False
        self.is_single_image = False
        
        if self.input_path and self.input_path.exists():
            if self.input_path.is_file():
                # Check if it's a video file
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
                if self.input_path.suffix.lower() in video_extensions:
                    self.is_video = True
                else:
                    # Assume it's a single image
                    self.is_single_image = True
            elif self.input_path.is_dir():
                self.is_image_dir = True
        
        self.current_frame_index = 0
        self.total_frames = 0
        
    def is_test_mode(self) -> bool:
        """Check if test mode is active."""
        return self.input_path is not None and self.input_path.exists()
    
    def get_frame_generator(self) -> Generator[Tuple[np.ndarray, str], None, None]:
        """
        Generate frames from the test input source.
        
        Yields:
            Tuple of (frame, description) where frame is numpy array and description is string
        """
        if not self.is_test_mode():
            return
        
        if self.is_video:
            yield from self._process_video()
        elif self.is_image_dir:
            yield from self._process_image_directory()
        elif self.is_single_image:
            yield from self._process_single_image()
    
    def _process_video(self) -> Generator[Tuple[np.ndarray, str], None, None]:
        """Process video file frame by frame."""
        cap = cv2.VideoCapture(str(self.input_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.input_path}")
            return
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing video: {self.input_path.name}")
        print(f"Total frames: {self.total_frames}, FPS: {fps:.2f}")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.current_frame_index = frame_count
            
            # Resize frame to match camera input size
            frame = cv2.resize(frame, config.CAMERA_PREVIEW_SIZE)
            
            description = f"Video: {self.input_path.name} - Frame {frame_count}/{self.total_frames}"
            yield frame, description
        
        cap.release()
        print(f"Video processing completed: {frame_count} frames processed")
    
    def _process_image_directory(self) -> Generator[Tuple[np.ndarray, str], None, None]:
        """Process all images in a directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = sorted([
            p for p in self.input_path.glob('*') 
            if p.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            print(f"No images found in {self.input_path}")
            return
        
        self.total_frames = len(image_files)
        print(f"Processing {self.total_frames} images from: {self.input_path}")
        
        for i, image_file in enumerate(image_files, 1):
            frame = cv2.imread(str(image_file))
            if frame is None:
                print(f"Warning: Could not load image {image_file}")
                continue
            
            self.current_frame_index = i
            
            # Resize frame to match camera input size
            frame = cv2.resize(frame, config.CAMERA_PREVIEW_SIZE)
            
            description = f"Image {i}/{self.total_frames}: {image_file.name}"
            yield frame, description
        
        print(f"Image directory processing completed: {len(image_files)} images processed")
    
    def _process_single_image(self) -> Generator[Tuple[np.ndarray, str], None, None]:
        """Process a single image file."""
        frame = cv2.imread(str(self.input_path))
        if frame is None:
            print(f"Error: Could not load image {self.input_path}")
            return
        
        self.total_frames = 1
        self.current_frame_index = 1
        
        # Resize frame to match camera input size
        frame = cv2.resize(frame, config.CAMERA_PREVIEW_SIZE)
        
        description = f"Single image: {self.input_path.name}"
        yield frame, description


class TestModeDisplay:
    """
    Display manager for test mode visualization.
    Handles window creation and user interaction.
    """
    
    def __init__(self):
        """Initialize test mode display."""
        self.window_name = config.TEST_MODE_WINDOW_NAME
        self.display_time = config.TEST_MODE_DISPLAY_TIME
        self.auto_advance = config.TEST_MODE_AUTO_ADVANCE
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        print("\n" + "="*60)
        print("TAIP TEST MODE CONTROLS:")
        print("  'q' - Quit test mode")
        print("  'n' - Next frame/image (manual mode)")
        print("  ' ' - Pause/Resume (video mode)")
        print("  'a' - Toggle auto-advance")
        print("  's' - Save current frame")
        print("="*60 + "\n")
    
    def display_frame(self, frame: np.ndarray, detections: List[YoloDetection], 
                     aruco_markers: List, gauge_pressure: Optional[float],
                     description: str) -> str:
        """
        Display frame with detection results and handle user input.
        
        Args:
            frame: Input frame
            detections: YOLO detections
            aruco_markers: ArUco markers
            gauge_pressure: Gauge pressure reading
            description: Frame description
            
        Returns:
            User command ('continue', 'quit', 'next', 'pause')
        """
        # Create annotated frame
        display_frame = frame.copy()
        
        # Draw detections
        if detections or aruco_markers:
            display_frame = draw_detections_on_frame(
                display_frame, detections, aruco_markers
            )
        
        # Add information overlay
        self._add_info_overlay(display_frame, detections, gauge_pressure, description)
        
        # Display frame
        cv2.imshow(self.window_name, display_frame)
        
        # Handle user input
        return self._handle_user_input()
    
    def _add_info_overlay(self, frame: np.ndarray, detections: List[YoloDetection],
                         gauge_pressure: Optional[float], description: str) -> None:
        """Add information overlay to frame."""
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text information
        y_pos = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Description
        cv2.putText(frame, description, (15, y_pos), font, 0.6, (255, 255, 255), 2)
        y_pos += 25
        
        # Detection count
        det_text = f"Detections: {len(detections)}"
        cv2.putText(frame, det_text, (15, y_pos), font, 0.6, (0, 255, 0), 2)
        y_pos += 20
        
        # Gauge pressure
        if gauge_pressure is not None:
            pressure_text = f"Pressure: {gauge_pressure:.2f} bar"
            color = (0, 0, 255) if gauge_pressure < config.DRILL_PRESSURE_THRESHOLD else (0, 255, 255)
            cv2.putText(frame, pressure_text, (15, y_pos), font, 0.6, color, 2)
        else:
            cv2.putText(frame, "Pressure: N/A", (15, y_pos), font, 0.6, (128, 128, 128), 2)
        y_pos += 20
        
        # Detection details
        if detections:
            for i, det in enumerate(detections[:3]):  # Show first 3
                det_text = f"  {det.class_name}: {det.confidence:.2f}"
                cv2.putText(frame, det_text, (200, 30 + i*20), font, 0.5, (255, 255, 0), 1)
    
    def _handle_user_input(self) -> str:
        """Handle user keyboard input."""
        if self.auto_advance:
            # Auto advance mode
            key = cv2.waitKey(max(1, self.display_time)) & 0xFF
        else:
            # Manual advance mode
            key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('n'):
            return 'next'
        elif key == ord(' '):
            return 'pause'
        elif key == ord('a'):
            self.auto_advance = not self.auto_advance
            print(f"Auto-advance: {'ON' if self.auto_advance else 'OFF'}")
            return 'continue'
        elif key == ord('s'):
            return 'save'
        else:
            return 'continue'
    
    def cleanup(self) -> None:
        """Cleanup display resources."""
        cv2.destroyWindow(self.window_name)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_test_mode_enabled() -> bool:
    """Check if test mode is enabled in configuration."""
    return (config.TEST_INPUT_PATH is not None and 
            Path(config.TEST_INPUT_PATH).exists())


def create_test_processor() -> Optional[TestModeProcessor]:
    """Create test mode processor if test mode is enabled."""
    if is_test_mode_enabled():
        return TestModeProcessor(config.TEST_INPUT_PATH)
    return None


def run_test_mode(oak_camera, gcs_client=None) -> None:
    """
    Run the TAIP system in test mode.
    
    Args:
        oak_camera: OAK camera instance (not used in test mode)
        gcs_client: GCS client for sending results (optional)
    """
    processor = create_test_processor()
    if not processor:
        print("Test mode not enabled or invalid input path")
        return
    
    display = TestModeDisplay()
    
    try:
        frame_count = 0
        for frame, description in processor.get_frame_generator():
            frame_count += 1
            
            # Process frame with OAK camera's detection pipeline
            # Note: In test mode, we'd need to adapt this to use the model directly
            # For now, we'll simulate detections or use a mock approach
            detections = []  # TODO: Implement detection on test frames
            aruco_markers = detect_aruco_markers(frame)
            gauge_pressure = calculate_gauge_reading(detections)
            
            # Display results
            command = display.display_frame(
                frame, detections, aruco_markers, gauge_pressure, description
            )
            
            if command == 'quit':
                break
            elif command == 'save':
                save_path = config.TAIP_ROOT / f"test_frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"Frame saved: {save_path}")
            
            # Send to GCS if enabled
            if gcs_client and detections:
                # TODO: Create and send payload data
                pass
        
        print(f"\nTest mode completed: {frame_count} frames processed")
        
    except KeyboardInterrupt:
        print("\nTest mode interrupted by user")
    except Exception as e:
        print(f"Error in test mode: {e}")
    finally:
        display.cleanup()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the test mode processor
    print("Testing test mode processor...")
    
    # Test with images directory
    test_path = config.PROJECT_ROOT / "models/testing/images"
    if test_path.exists():
        processor = TestModeProcessor(test_path)
        print(f"Test mode enabled: {processor.is_test_mode()}")
        print(f"Is image directory: {processor.is_image_dir}")
        
        # Process a few frames
        frame_count = 0
        for frame, description in processor.get_frame_generator():
            print(f"Frame {frame_count + 1}: {description}")
            frame_count += 1
            if frame_count >= 3:  # Just test first 3
                break
    else:
        print(f"Test images directory not found: {test_path}")
    
    print("Test mode processor testing completed.")
