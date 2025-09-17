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
from vision_processing import (calculate_gauge_reading, detect_aruco_markers,
                               show_inference_visualisation)
from file_processing import FileProcessor
from gcs_client import GCSClient
from drilling import DrillController  # Updated import from new module
from enviro_lcd import EnvironmentalSensors, LCDDisplay  # Moved here


class MainApp:
    """The main application class for the TAIP subsystem."""

    def __init__(self):
        self.running = True
        self.file_processor = None
        self.camera = None
        
        # Hardware components
        self.gcs_client = None
        self.env_sensors = None
        self.lcd_display = None
        self.drill_controller = None
        
        # System state
        self.ip_address = "N/A"
        self.last_telem_time = 0
        self.last_frame_time = 0

    def setup(self):
        """Initialise all hardware and software components."""
        print("Initialising TAIP Subsystem...")
        
        # Determine input mode and initialise accordingly
        if config.INPUT_PATH and Path(config.INPUT_PATH).exists():
            print(f"File input mode: {config.INPUT_PATH}")
            self.file_processor = FileProcessor(config.INPUT_PATH)
        else:
            print("Live camera mode")
            self.camera = OakCamera()
            
        # Initialise hardware components
        self.gcs_client = GCSClient()
        self.env_sensors = EnvironmentalSensors()
        self.lcd_display = LCDDisplay()
        self.drill_controller = DrillController()
        
        # Get system IP address
        self.ip_address = self._get_ip_address()
        print(f"System IP: {self.ip_address}")

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

            # ArUco detection (isolated to avoid fatal loop errors)
            try:
                # When visualising, we MUST detect on the same frame being displayed.
                # The rgb_frame is now the passthrough frame from the YOLO node, which is perfect.
                if (config.SHOW_LIVE_VISUALISATION or self.file_processor) and rgb_frame is not None:
                    aruco_frame = rgb_frame
                    matrix = config.CAMERA_MATRIX_RGB
                    coeffs = config.DISTORTION_COEFFS_RGB
                # Headless operation can use the configured source
                else:
                    aruco_frame = mono_frame if config.CAMERA_ARUCO_SOURCE.upper() == 'LEFT' else rgb_frame
                    matrix = config.CAMERA_MATRIX
                    coeffs = config.DISTORTION_COEFFS
                    
                if aruco_frame is not None:
                    aruco_detections, aruco_corners, aruco_ids = detect_aruco_markers(
                        aruco_frame, matrix, coeffs
                    )
                else:
                    aruco_detections, aruco_corners, aruco_ids = [], None, None

            except Exception as e:
                print(f"ArUco error: {e}")
                aruco_detections, aruco_corners, aruco_ids = [], None, None

            gauge_reading = calculate_gauge_reading(yolo_detections)
            env_data = self.env_sensors.get_readings()
            
            # Control drill based on gauge reading
            self.drill_controller.control_drill(gauge_reading)
            
            # Handle GCS communication
            self._handle_gcs_communication(rgb_frame, yolo_detections, aruco_detections,
                                           gauge_reading, env_data)

            # Update LCD display
            proximity = self.env_sensors.get_proximity()
            self.lcd_display.update_mode(proximity)
            self.lcd_display.update_display(self.ip_address, rgb_frame, yolo_detections,
                                            env_data, gauge_reading, bool(self.file_processor))

            # Show visualisation for file processing mode or if live visualisation enabled
            if self.file_processor or (config.SHOW_LIVE_VISUALISATION and self.camera):
                # The frame for visualisation is the passthrough frame, which matches YOLO and ArUco coordinates.
                key = show_inference_visualisation(
                    rgb_frame, yolo_detections, aruco_detections, aruco_corners, aruco_ids, gauge_reading,
                    config.CAMERA_MATRIX_RGB, config.DISTORTION_COEFFS_RGB
                )
                if key == ord('q'):
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

    def shutdown(self):
        """Clean up all resources."""
        print("Shutting down TAIP Subsystem...")
        self.running = False
        
        if self.file_processor:
            self.file_processor.close()
        if self.camera:
            self.camera.close()
        if self.gcs_client:
            self.gcs_client.shutdown()
        if self.lcd_display:
            self.lcd_display.close()
        if self.drill_controller:
            self.drill_controller.close()
            
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
            
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