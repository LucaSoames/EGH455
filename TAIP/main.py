# /home/pi/EGH455/TAIP/main.py

"""
Main Application for the EGH455 TAIP Subsystem

This script orchestrates the entire payload's functionality. It initialises all
hardware components, runs a continuous processing loop to gather and analyse
data, communicates with the Ground Control Station (GCS), and manages the
local LCD display.
"""

import time
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List
import socket
from pathlib import Path

# Import our custom modules
import config
from data_models import PayloadData, YoloDetection, ArucoDetection, EnvironmentalData
from oak_camera import OakCamera
from vision_processing import (calculate_gauge_reading,
                               detect_aruco_markers,
                               show_inference_visualisation)
from file_inference import FileInferenceProcessor
from gcs_client import GCSClient
from file_processor import FileProcessor

# Conditional import for RPi.GPIO
try:
    import RPi.GPIO as GPIO
    IS_GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    print("WARNING: RPi.GPIO not found. Running without GPIO support.")
    IS_GPIO_AVAILABLE = False

# Conditional import for Pimoroni libraries
try:
    from bme280 import BME280
    from ltr559 import LTR559
    import st7735
    from PIL import Image, ImageDraw, ImageFont
    from fonts.ttf import RobotoMedium as UserFont
    IS_ENVIRO_AVAILABLE = True
except ImportError:
    print("WARNING: Enviro+ libraries not found. Running without LCD/sensor support.")
    IS_ENVIRO_AVAILABLE = False


class EnvironmentalSensors:
    """Handles Enviro+ board sensors."""
    
    def __init__(self):
        self.bme = None
        self.ltr = None
        if IS_ENVIRO_AVAILABLE:
            try:
                self.bme = BME280()
                self.ltr = LTR559()
                print("✓ Environmental sensors initialised")
            except Exception as e:
                print(f"Environmental sensor init failed: {e}")
    
    def get_readings(self) -> Optional[EnvironmentalData]:
        """Read data from the Enviro+ sensors."""
        if not self.bme or not self.ltr:
            return None
        
        try:
            return EnvironmentalData(
                temperature_c=self.bme.get_temperature(),
                pressure_hpa=self.bme.get_pressure(),
                humidity_rh=self.bme.get_humidity(),
                light_lux=self.ltr.get_lux()
            )
        except Exception as e:
            print(f"Environmental sensor error: {e}")
            return None
    
    def get_proximity(self) -> int:
        """Get proximity sensor reading for LCD mode switching."""
        return self.ltr.get_proximity() if self.ltr else 0


class LCDDisplay:
    """Handles the ST7735 LCD display on the Enviro+ board."""
    
    def __init__(self):
        self.lcd = None
        self.current_mode = 0
        self.last_tap_time = 0
        
        if IS_ENVIRO_AVAILABLE:
            try:
                self.lcd = st7735.ST7735(port=0, cs=1, dc="GPIO9", 
                                       backlight="GPIO12", rotation=270)
                self.lcd.begin()
                self.image = Image.new("RGB", (self.lcd.width, self.lcd.height), color=(0, 0, 0))
                self.draw = ImageDraw.Draw(self.image)
                self.font = ImageFont.truetype(UserFont, 14)
                print("✓ LCD display initialised")
            except Exception as e:
                print(f"LCD setup failed: {e}")
                self.lcd = None
    
    def update_mode(self, proximity: int):
        """Update LCD mode based on proximity sensor."""
        now = time.time()
        if proximity > 1500 and (now - self.last_tap_time) > 0.5:
            self.current_mode = (self.current_mode + 1) % 3
            self.last_tap_time = now
    
    def update_display(self, ip_address: str, frame, detections,
                       env_data, gauge_pressure, is_file_mode: bool):
        """
        Update LCD content.
        gauge_pressure may be None.
        """
        if not self.lcd:
            return
        try:
            self.draw.rectangle((0, 0, self.lcd.width, self.lcd.height), (0, 0, 0))

            if self.current_mode == 0:
                self.draw.text((5, 5), "IP:", fill=(255, 255, 0), font=self.font)
                self.draw.text((5, 23), ip_address, fill=(255, 255, 255), font=self.font)
                mode_text = "File" if is_file_mode else "Live"
                self.draw.text((5, 41), f"Mode: {mode_text}", fill=(0, 255, 255), font=self.font)
                if gauge_pressure is not None:
                    self.draw.text((5, 59), f"P:{gauge_pressure:.2f} bar",
                                   fill=(0, 255, 0), font=self.font)

            elif self.current_mode == 1 and frame is not None:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_pil = img_pil.resize((self.lcd.width, self.lcd.height))
                d2 = ImageDraw.Draw(img_pil)
                for det in detections:
                    x1 = int(det.box[0] * self.lcd.width)
                    y1 = int(det.box[1] * self.lcd.height)
                    x2 = int(det.box[2] * self.lcd.width)
                    y2 = int(det.box[3] * self.lcd.height)
                    d2.rectangle((x1, y1, x2, y2), outline="lime", width=1)
                if gauge_pressure is not None:
                    d2.text((2, 2), f"{gauge_pressure:.1f} bar",
                            fill=(255, 255, 0), font=self.font)
                self.image.paste(img_pil)

            elif self.current_mode == 2 and env_data:
                self.draw.text((2, 2),  f"T {env_data.temperature_c:.1f}C", fill=(255,255,255), font=self.font)
                self.draw.text((2, 20), f"P {env_data.pressure_hpa:.0f}hPa", fill=(255,255,255), font=self.font)
                self.draw.text((2, 38), f"H {env_data.humidity_rh:.0f}%", fill=(255,255,255), font=self.font)
                self.draw.text((2, 56), f"L {env_data.light_lux:.0f}lx", fill=(255,255,255), font=self.font)
                if gauge_pressure is not None:
                    self.draw.text((2, 74), f"G {gauge_pressure:.1f}b", fill=(0,255,0), font=self.font)

            self.lcd.display(self.image)
        except Exception as e:
            print(f"LCD update error: {e}")
    
    def close(self):
        """Turn off LCD backlight."""
        if self.lcd:
            self.lcd.set_backlight(0)


class DrillController:
    """Handles drill activation via GPIO (graceful fallback if GPIO unsupported)."""
    
    def __init__(self):
        self.gpio_available = IS_GPIO_AVAILABLE
        self.drill_active = False
        if self.gpio_available:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(config.DRILL_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
                print(f"✓ GPIO pin {config.DRILL_GPIO_PIN} initialised")
            except RuntimeError as e:
                print(f"GPIO unavailable ({e}). Continuing without drill control.")
                self.gpio_available = False
            except Exception as e:
                print(f"GPIO init error: {e}")
                self.gpio_available = False
    
    def control_drill(self, gauge_reading: Optional[float]):
        """Control drill activation based on pressure reading."""
        should_activate = (gauge_reading is not None and 
                          gauge_reading < config.DRILL_PRESSURE_THRESHOLD)
        
        # Only update GPIO if state changed
        if should_activate != self.drill_active:
            self.drill_active = should_activate
            if self.gpio_available:
                GPIO.output(config.DRILL_GPIO_PIN, GPIO.HIGH if should_activate else GPIO.LOW)
                status = "ACTIVATED" if should_activate else "DEACTIVATED"
                pressure_txt = f"{gauge_reading:.2f}" if gauge_reading is not None else "N/A"
                print(f"DRILL {status} - Pressure: {pressure_txt} bar")
    
    def close(self):
        """Clean up GPIO."""
        if self.gpio_available:
            GPIO.cleanup()


class MainApp:
    """The main application class for the TAIP subsystem."""

    def __init__(self):
        self.running = True
        
        # Input sources
        self.camera = None
        self.file_processor = None
        self.file_inference = None
        
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
            self.file_inference = FileInferenceProcessor()
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
                aruco_detections = detect_aruco_markers(
                    mono_frame, config.CAMERA_MATRIX, config.DISTORTION_COEFFS
                )
            except Exception as e:
                print(f"ArUco error: {e}")
                aruco_detections = []

            gauge_reading = calculate_gauge_reading(yolo_detections)
            env_data = self.env_sensors.get_readings()
            self.drill_controller.control_drill(gauge_reading)
            self._handle_gcs_communication(rgb_frame, yolo_detections, aruco_detections,
                                           gauge_reading, env_data)

            proximity = self.env_sensors.get_proximity()
            self.lcd_display.update_mode(proximity)
            self.lcd_display.update_display(self.ip_address, rgb_frame, yolo_detections,
                                            env_data, gauge_reading, bool(self.file_processor))

            if self.file_processor:
                key = show_inference_visualisation(
                    rgb_frame, yolo_detections, aruco_detections, gauge_reading,
                    config.CAMERA_MATRIX, config.DISTORTION_COEFFS
                )
                if key == ord('q'):
                    print("Quit requested by user")
                    break
                elif key == ord('s'):
                    self._save_debug_frame(rgb_frame, yolo_detections, frame_count)
                # 'n' or any key (except handled above) just advances when manual mode
                if frame_count % 5 == 0:
                    pressure_txt = f"{gauge_reading:.2f}" if gauge_reading is not None else "N/A"
                    print(f"Frame {frame_count}: {len(yolo_detections)} detections, pressure: {pressure_txt} bar")

            time.sleep(0.01)

    def _get_frame_and_detections(self):
        """Get frame and detections from the appropriate input source."""
        if self.file_processor:
            # File mode
            rgb_frame = self.file_processor.get_next_frame()
            if rgb_frame is None:
                return None, None, []
            mono_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            yolo_detections = self.file_inference.run_inference(rgb_frame) if self.file_inference else []
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

    def _save_debug_frame(self, frame, detections, frame_count):
        """Save current frame for debugging."""
        save_path = config.TAIP_ROOT / f"saved_frame_{frame_count}_{int(time.time())}.jpg"
        cv2.imwrite(str(save_path), frame)
        print(f"Frame saved: {save_path}")

    def shutdown(self):
        """Clean up all resources."""
        print("Shutting down TAIP Subsystem...")
        self.running = False
        if self.camera:
            self.camera.close()
        if self.file_processor:
            self.file_processor.close()
        if self.file_inference:
            self.file_inference.close()
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
        import traceback
        traceback.print_exc()
    finally:
        app.shutdown()