# /home/pi/EGH455/TAIP/main.py

"""
Main Application for the EGH455 TAIP Subsystem

This script orchestrates the entire payload's functionality. It initialises all
hardware components, runs a continuous processing loop to gather and analyse
data, communicates with the Ground Control Station (GCS), and manages the
local LCD display.
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb' # Set the Qt platform plugin before importing cv2 to supress Wayland/XCB warnings

import time
import cv2
import socket
import traceback
from datetime import datetime
from typing import Optional
from pathlib import Path

# Import our custom modules
import config
from data_models import PayloadData, EnvironmentalData
from oak_camera import OakCamera
from vision_processing import (calculate_gauge_reading,
                               show_inference_visualisation,
                               draw_detections_on_frame,
                               ArucoWorker)
from file_processing import FileProcessor
from gcs_client import GCSClient
from enviro_lcd import EnvironmentalSensors, LCDDisplay
from drilling import DrillController


class MainApp:
    """The main application class for the TAIP subsystem."""

    def __init__(self):
        self.running = True
        self.file_processor = None
        self.camera = None
        self.is_video_mode = True
        
        # Hardware components
        self.gcs_client = None
        self.env_sensors = None
        self.lcd_display = None
        self.drill_controller = None
        # ArUco worker
        self.aruco_worker: Optional[ArucoWorker] = None
        
        # System state
        self.ip_address = "N/A"
        self.last_telem_time = 0
        self.last_frame_time = 0

    def setup(self):
        """Initialise all hardware and software components."""
        print("Initialising TAIP Subsystem...")
        
        # Determine input mode and initialise accordingly
        if config.INPUT_PATH:
            # File input mode
            print(f"File input mode: {config.INPUT_PATH}")
            self.file_processor = FileProcessor(config.INPUT_PATH)
            # Determine if auto-advance or manual based on input type
            self.is_video_mode = self.file_processor.is_video
            mode_text = "video (auto-advance)" if self.is_video_mode else "images (manual advance)"
            print(f"Input type: {mode_text}")
            # In test mode, ArUco runs on RGB frames using RGB intrinsics
            K = config.CAMERA_MATRIX_RGB
            D = config.DISTORTION_COEFFS_RGB
        else:
            # Live camera mode
            print("Live camera mode")
            self.camera = OakCamera()
            self.is_video_mode = True
            # In live mode, ArUco runs on LEFT mono using LEFT intrinsics
            K = config.CAMERA_MATRIX_LEFT
            D = config.DISTORTION_COEFFS_LEFT

        # Start non-blocking ArUco worker
        self.aruco_worker = ArucoWorker(camera_matrix=K, dist_coeffs=D, marker_size_m=config.ARUCO_MARKER_SIZE_M, max_hz=30.0)
        self.aruco_worker.start()
        
        # Initialise hardware components
        # GCS client with LCD callback
        self.gcs_client = GCSClient(lcd_callback=self._handle_lcd_command)
        self.env_sensors = EnvironmentalSensors()
        self.lcd_display = LCDDisplay()
        self.drill_controller = DrillController()
        
        # Get system IP address
        self.ip_address = self._get_ip_address()
        print(f"System IP: {self.ip_address}")
        print(f"GCS Server URL: {config.GCS_URL}")

    def run_loop(self):
        """Main processing loop for both camera and file input."""
        frame_count = 0
        print("Starting main processing loop...")
        
        while self.running:
            rgb_frame, mono_frame, yolo_detections = self._get_frame_and_detections()
            if rgb_frame is None:
                if self.file_processor:
                    print(f"File processing completed. Processed {frame_count} frames.")
                    break
                time.sleep(0.01)
                continue

            if mono_frame is None:
                mono_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

            frame_count += 1

            # Non-blocking ArUco: feed the worker and read last result
            try:
                if self.file_processor:
                    # Test mode: worker configured for RGB intrinsics
                    self.aruco_worker.update_frame(rgb_frame)
                else:
                    # Live mode: worker configured for LEFT intrinsics
                    self.aruco_worker.update_frame(mono_frame)
                aruco_detections, aruco_corners, aruco_ids, _, _, aruco_vis = self.aruco_worker.get_latest()
                if aruco_detections is None:
                    aruco_detections, aruco_corners, aruco_ids, aruco_vis = [], None, None, None
            except Exception as e:
                print(f"ArUco error: {e}")
                aruco_detections, aruco_corners, aruco_ids, aruco_vis = [], None, None, None

            gauge_reading = calculate_gauge_reading(yolo_detections)
            env_data = self.env_sensors.get_readings()
            
            # Drill control
            if self.drill_controller:
                self.drill_controller.control_drill(gauge_reading)
            
            # GCS communication - send telemetry and frames to remote GCS server
            self._handle_gcs_communication(rgb_frame, yolo_detections, aruco_detections, gauge_reading, env_data)

            # Update LCD display - pass frame WITH detections drawn
            proximity = self.env_sensors.get_proximity()
            self.lcd_display.update_mode(proximity)
            
            # Draw detections on frame for LCD display
            frame_with_detections = draw_detections_on_frame(rgb_frame, yolo_detections, aruco_detections)
            self.lcd_display.update_display(self.ip_address, frame_with_detections, env_data)

            # Show visualisation if enabled (works for live camera, video, or images)
            if config.SHOW_LIVE_VISUALISATION:
                # Determine camera matrix for the RGB frame
                if self.file_processor:
                    K = config.CAMERA_MATRIX_RGB
                    D = config.DISTORTION_COEFFS_RGB
                else:
                    K = config.CAMERA_MATRIX_RGB
                    D = config.DISTORTION_COEFFS_RGB
                
                key = show_inference_visualisation(
                    rgb_frame, yolo_detections, aruco_detections, aruco_corners, aruco_ids, gauge_reading,
                    camera_matrix=K, dist_coeffs=D,
                    aruco_inset_bgr=aruco_vis,
                    is_video_mode=self.is_video_mode  # Pass the mode flag
                )
                
                # Handle quit key (q or ESC)
                if key == ord('q') or key == 27:
                    print("Quit requested by user")
                    break
                if frame_count % 1 == 0:
                    pressure_txt = f"{gauge_reading:.2f}" if gauge_reading is not None else "N/A"
                    print(f"Frame {frame_count}: {len(yolo_detections)} detections, {len(aruco_detections)} markers, pressure: {pressure_txt} bar")
    
            # Reset drill state if needed (when pressure is back above threshold + margin)
            if (self.drill_controller.drilling_complete and 
                gauge_reading is not None and 
                gauge_reading >= config.DRILL_PRESSURE_THRESHOLD + 2.0):
                self.drill_controller.reset_drill_state()
                print(f"Drill reset: pressure now {gauge_reading:.2f} bar (above threshold)")
                
            time.sleep(0.01)  # Small sleep to reduce CPU hogging

    def _get_frame_and_detections(self):
        """Get frame and detections from the appropriate input source."""
        if self.file_processor:
            # File mode
            rgb_frame = self.file_processor.get_next_frame()
            if rgb_frame is None:
                return None, None, []
                
            mono_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            yolo_detections = self.file_processor.process_frame(rgb_frame)
            return rgb_frame, mono_frame, yolo_detections
        else:
            # Live camera mode
            rgb_frame = self.camera.get_latest_rgb_frame()
            mono_frame = self.camera.get_latest_mono_frame()
            yolo_detections = self.camera.get_latest_detections()
            return rgb_frame, mono_frame, yolo_detections

    def _handle_gcs_communication(self, rgb_frame, yolo_detections, aruco_detections, 
                                gauge_reading, env_data):
        """Handle GCS communication at controlled rates."""
        now = time.time()
        
        # Send telemetry data
        if (now - self.last_telem_time) >= (1.0 / config.POST_TELEM_HZ):
            payload = PayloadData(
                timestamp=datetime.now().isoformat(),
                yolo_detections=yolo_detections,
                aruco_markers=aruco_detections,
                gauge_pressure_bar=gauge_reading,
                environmental_data=env_data
            )
            self.gcs_client.send_data(payload)
            self.last_telem_time = now

        # Send video frame
        if (now - self.last_frame_time) >= (1.0 / config.POST_FRAME_FPS):
            self.gcs_client.send_frame(rgb_frame)
            self.last_frame_time = now

    def _get_ip_address(self) -> str:
        """Get the primary IP address of the device."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "No IP Found"
        return ip

    def _handle_lcd_command(self, tab_index: int):
        """Callback for handling LCD tab change commands from GCS."""
        if self.lcd_display:
            self.lcd_display.set_tab(tab_index)

    def shutdown(self):
        """Clean up all resources."""
        print("Shutting down TAIP Subsystem...")
        self.running = False
        
        # Close OpenCV windows first
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process window events
        except Exception as e:
            print(f"Error closing CV windows: {e}")
        
        # Stop ArUco worker
        if self.aruco_worker:
            self.aruco_worker.stop()
        
        if self.file_processor:
            self.file_processor.close()
        
        if self.camera:
            self.camera.close()
        
        if self.gcs_client:
            self.gcs_client.shutdown()
        
        if self.drill_controller:
            self.drill_controller.close()
        
        if self.lcd_display:
            self.lcd_display.close()
        
        print("Shutdown complete.")

if __name__ == '__main__':
    app = MainApp()
    try:
        app.setup()
        app.run_loop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"FATAL ERROR in main application: {e}")
        traceback.print_exc()
    finally:
        app.shutdown()