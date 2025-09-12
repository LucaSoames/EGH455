# /home/pi/EGH455/TAIP/main.py

"""
Main Application for the EGH455 TAIP Subsystem

This script orchestrates the entire payload's functionality. It initializes all
hardware components, runs a continuous processing loop to gather and analyze
data, communicates with the Ground Control Station (GCS), and manages the
local LCD display.
"""

import time
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List
import socket

# Import our custom modules
import config
from data_models import PayloadData, YoloDetection, ArucoDetection, EnvironmentalData
from oak_camera import OakCamera, MockCamera
from vision_processing import calculate_gauge_reading, detect_aruco_markers
from gcs_client import GCSClient

# Conditional import for RPi.GPIO
try:
    import RPi.GPIO as GPIO
    IS_GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    print("WARNING: RPi.GPIO not found. Running without GPIO support (drilling disabled).")
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


class MainApp:
    """The main application class for the TAIP subsystem."""

    def __init__(self, use_live_camera: bool):
        self.running = True
        self.use_live_camera = use_live_camera
        self.camera = None
        self.gcs_client = None
        self.enviro_bme = None
        self.enviro_ltr = None
        self.lcd = None
        self.current_lcd_mode = 0  # 0: IP, 1: Live Feed, 2: Sensors
        self.last_lcd_tap = 0
        self.ip_address = "N/A"

    def setup(self):
        """Initializes all hardware and software components."""
        print("Initializing TAIP Subsystem...")
        
        # --- Initialize GPIO ---
        if IS_GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(config.DRILL_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
            print(f"GPIO pin {config.DRILL_GPIO_PIN} for drilling initialized.")

        # --- Initialize GCS Client ---
        self.gcs_client = GCSClient()

        # --- Initialize Enviro+ Board and LCD ---
        if IS_ENVIRO_AVAILABLE:
            self.enviro_bme = BME280()
            self.enviro_ltr = LTR559()
            self._setup_lcd()
            self.ip_address = self._get_ip_address()
            print(f"Enviro+ board initialized. IP Address: {self.ip_address}")
        
        # --- Initialize OAK-D Lite Camera (must be last hardware init) ---
        # This will start its own background processing thread.
        if self.use_live_camera:
            self.camera = OakCamera()
        else:
            self.camera = MockCamera(config.INPUT_PATH)
            # When using MockCamera, we need a separate inference engine
            self._setup_local_inference()
    
    def _setup_local_inference(self):
        """Sets up a local DepthAI pipeline for processing files."""
        self.local_pipeline = dai.Pipeline()
        self.xinFrame = self.local_pipeline.create(dai.node.XLinkIn)
        self.xinFrame.setStreamName("inFrame")
        self.detection_network = self.local_pipeline.create(dai.node.YoloDetectionNetwork)
        self.detection_network.setBlobPath(config.BLOB_PATH)
        self.detection_network.setConfidenceThreshold(config.CONFIDENCE_THRESHOLD)
        # Load other NN params from config as needed
        self.xout_nn = self.local_pipeline.create(dai.node.XLinkOut)
        self.xout_nn.setStreamName("nn")
        self.xinFrame.out.link(self.detection_network.input)
        self.detection_network.out.link(self.xout_nn.input)
        
        self.local_device = dai.Device(self.local_pipeline)
        self.qIn = self.local_device.getInputQueue("inFrame")
        self.qDet = self.local_device.getOutputQueue("nn")

    def _run_local_inference(self, frame):
        """Runs inference on a single frame for file-based testing."""
        img = dai.ImgFrame()
        img.setData(cv2.resize(frame, self.camera.model_input_size).transpose(2,0,1).flatten())
        img.setType(dai.ImgFrame.Type.BGR888p)
        img.setWidth(self.camera.model_input_size[0])
        img.setHeight(self.camera.model_input_size[1])
        self.qIn.send(img)
        
        in_det = self.qDet.get()
        detections = []
        if in_det:
            detections = [YoloDetection(self.camera.class_names[d.label], d.confidence, (d.xmin, d.ymin, d.xmax, d.ymax)) for d in in_det.detections]
        return detections

    def run_loop(self):
        last_frame_post_time = 0
        last_telem_post_time = 0
        while self.running:
            # 1. Acquire Data
            rgb_frame = self.camera.get_latest_rgb_frame()
            if rgb_frame is None: 
                if not self.use_live_camera: self.running = False # End of files
                time.sleep(0.1)
                continue

            if self.use_live_camera:
                mono_frame = self.camera.get_latest_mono_frame()
                yolo_detections = self.camera.get_latest_detections()
            else: # File-based testing requires local inference
                mono_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                yolo_detections = self._run_local_inference(rgb_frame)

            environmental_data = self._get_environmental_data()
            
            # 2. Process Data
            aruco_detections = detect_aruco_markers(mono_frame, config.CAMERA_MATRIX, config.DIST_COEFFS)
            gauge_reading = calculate_gauge_reading(yolo_detections)

            # 3. Control Logic
            if gauge_reading is not None and gauge_reading < config.DRILL_THRESHOLD_BAR and IS_GPIO_AVAILABLE:
                GPIO.output(config.DRILL_GPIO_PIN, GPIO.HIGH)
            elif IS_GPIO_AVAILABLE:
                GPIO.output(config.DRILL_GPIO_PIN, GPIO.LOW)
            
            # 4. Package & Communicate Data
            now = time.time()
            if (now - last_telem_post_time) >= (1.0 / config.POST_TELEM_HZ):
                payload = PayloadData(datetime.now().isoformat(), yolo_detections, aruco_detections, gauge_reading, environmental_data)
                self.gcs_client.send_data(payload)
                last_telem_post_time = now

            if (now - last_frame_post_time) >= (1.0 / config.POST_FRAME_FPS):
                self.gcs_client.send_frame(rgb_frame)
                last_frame_post_time = now
            
            # 5. Update Local Display
            self._update_lcd(rgb_frame, yolo_detections, environmental_data)
            
            if not self.use_live_camera: # For viewing file test output
                display_frame = rgb_frame.copy()
                # Draw detections for display
                cv2.imshow("Test Mode", display_frame)
                if cv2.waitKey(1) == ord('q'): break
            
            # Yield CPU
            time.sleep(0.01)
    
    def _setup_lcd(self):
        """Sets up the ST7735 LCD display."""
        self.lcd = st7735.ST7735(port=0, cs=1, dc="GPIO9", backlight="GPIO12", rotation=270)
        self.lcd.begin()
        self.lcd_image = Image.new("RGB", (self.lcd.width, self.lcd.height), color=(0, 0, 0))
        self.lcd_draw = ImageDraw.Draw(self.lcd_image)
        self.lcd_font = ImageFont.truetype(UserFont, 14)

    def _get_ip_address(self) -> str:
        """Retrieves the primary IP address of the device."""
        # Implementation from your display_ip.py prototype
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = "No IP Found"
        finally:
            s.close()
        return ip

    def _get_environmental_data(self) -> Optional[EnvironmentalData]:
        """Reads data from the Enviro+ sensors."""
        if not IS_ENVIRO_AVAILABLE or self.enviro_bme is None or self.enviro_ltr is None:
            return None
        return EnvironmentalData(
            temperature_c=self.enviro_bme.get_temperature(),
            pressure_hpa=self.enviro_bme.get_pressure(),
            humidity_rh=self.enviro_bme.get_humidity(),
            light_lux=self.enviro_ltr.get_lux()
        )

    def _update_lcd(self, frame: np.ndarray, detections: List[YoloDetection], env_data: Optional[EnvironmentalData]):
        """Manages the content displayed on the Enviro+ LCD."""
        if not IS_ENVIRO_AVAILABLE or not self.lcd or not self.enviro_ltr: 
            return
        now = time.time()

        # Check proximity sensor to cycle through modes
        if self.enviro_ltr.get_proximity() > 1500 and (now - self.last_lcd_tap) > 0.5:
            self.current_lcd_mode = (self.current_lcd_mode + 1) % 3
            self.last_lcd_tap = now

        # Clear display
        self.lcd_draw.rectangle((0, 0, self.lcd.width, self.lcd.height), (0, 0, 0))
        
        # IP Address tab
        if self.current_lcd_mode == 0:
            self.lcd_draw.text((5, 5), "IP Address:", fill=(255, 255, 0), font=self.lcd_font)
            self.lcd_draw.text((5, 25), self.ip_address, fill=(255, 255, 255), font=self.lcd_font)
        
        # Live Feed tab
        elif self.current_lcd_mode == 1:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_pil = img_pil.resize((self.lcd.width, self.lcd.height))
            # Create a new draw object for the resized image before drawing
            draw_on_resized = ImageDraw.Draw(img_pil)
            for det in detections:
                # Use frame shape to ensure coordinates are scaled correctly regardless of the frame's resolution
                box = [int(c * dim) for c, dim in zip(det.box, [self.lcd.width, self.lcd.height, self.lcd.width, self.lcd.height])]
                draw_on_resized.rectangle(box, outline="lime", width=1)
            self.lcd_image.paste(img_pil)
        
        # Sensor Data tab
        elif self.current_lcd_mode == 2 and env_data:
            self.lcd_draw.text((5, 5), f"Temp: {env_data.temperature_c:.1f} C", fill=(255, 255, 255), font=self.lcd_font)
            self.lcd_draw.text((5, 25), f"Pres: {env_data.pressure_hpa:.1f} hPa", fill=(255, 255, 255), font=self.lcd_font)
            self.lcd_draw.text((5, 45), f"Humid: {env_data.humidity_rh:.1f} %", fill=(255, 255, 255), font=self.lcd_font)
            self.lcd_draw.text((5, 65), f"Light: {env_data.light_lux:.1f} Lux", fill=(255, 255, 255), font=self.lcd_font)
            
        # Display the composed image
        self.lcd.display(self.lcd_image)


    def shutdown(self):
        """Cleans up all resources."""
        print("Shutting down TAIP Subsystem...")
        self.running = False
        if self.camera: 
            self.camera.close()
        if self.gcs_client: 
            self.gcs_client.shutdown()
        if hasattr(self, 'local_device'): 
            self.local_device.close()
        
        # Turn off LCD backlight
        if self.lcd: 
            self.lcd.set_backlight(0)
        if IS_GPIO_AVAILABLE: 
            GPIO.cleanup()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == '__main__':
    is_live = config.INPUT_PATH is None
    app = MainApp(use_live_camera=is_live)
    try:
        app.setup()
        app.run_loop()
    except Exception as e:
        print(f"FATAL ERROR in main application: {e}")
    finally:
        app.shutdown()