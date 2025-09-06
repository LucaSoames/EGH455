"""
Main application for the TAIP (Target Acquisition and Image Processing) subsystem.
Orchestrates all components and manages the main processing loop.
"""

import cv2
import numpy as np
import time
import signal
import sys
import logging
import argparse
from typing import Optional, Dict, Any
from pathlib import Path
import RPi.GPIO as GPIO
from threading import Event
import traceback

# Import TAIP modules
import config
from data_models import PayloadData, EnvironmentalData, SystemStatus
from oak_camera import OakCamera
from gcs_client import GCSClient
from test_mode import TestModeProcessor, TestModeDisplay
from vision_processing import (
    calculate_gauge_reading, 
    detect_aruco_markers, 
    filter_detections_by_confidence,
    draw_detections_on_frame
)

# Import Pimoroni Enviro+ modules
try:
    from enviroplus import gas, noise, motion
    from enviroplus.noise import Noise
    import ST7735
    from PIL import Image, ImageDraw, ImageFont
    import ltr559
    import bme280
    ENVIRO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enviro+ modules not available: {e}")
    ENVIRO_AVAILABLE = False


class TAIPSystem:
    """
    Main TAIP subsystem coordinator.
    
    Manages all hardware components and orchestrates the main processing loop
    for target acquisition, image processing, and GCS communication.
    """
    
    def __init__(self, test_mode: bool = False, test_input: Optional[str] = None):
        """Initialize the TAIP system with all components."""
        # System state
        self.running = False
        self.shutdown_event = Event()
        
        # Test mode configuration
        self.test_mode = test_mode
        self.test_input = test_input
        self.test_processor: Optional[TestModeProcessor] = None
        self.test_display: Optional[TestModeDisplay] = None
        
        # Hardware components
        self.camera: Optional[OakCamera] = None
        self.gcs_client: Optional[GCSClient] = None
        self.lcd: Optional[ST7735.ST7735] = None
        
        # Display state
        self.display_mode = 0  # 0: IP, 1: Live feed, 2: Sensor data, 3: Status
        self.proximity_sensor = None
        
        # Environmental sensors
        self.bme280_sensor = None
        self.light_sensor = None
        
        # Processing state
        self.frames_processed = 0
        self.last_gauge_reading: Optional[float] = None
        self.drill_active = False
        self.last_environmental_update = 0.0
        self.last_environmental_data: Optional[EnvironmentalData] = None
        
        # Performance tracking
        self.loop_times = []
        self.last_error: Optional[str] = None
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        if self.test_mode:
            self.logger.info(f"TAIP System initializing in test mode with input: {test_input}")
        else:
            self.logger.info("TAIP System initializing in live mode...")
    
    def _setup_logging(self) -> None:
        """Configure logging system."""
        # Create logs directory
        config.LOG_FILE.parent.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format=config.LOG_FORMAT,
            handlers=[
                logging.FileHandler(config.LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def initialize_hardware(self) -> bool:
        """
        Initialize all hardware components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing hardware components...")
            
            # Initialize GPIO (only in live mode)
            if not self.test_mode:
                self._setup_gpio()
            
            # Initialize camera
            if not self._initialize_camera():
                return False
            
            # Initialize GCS client (skip in test mode if desired)
            if not self.test_mode or config.TEST_MODE_SEND_TO_GCS:
                if not self._initialize_gcs():
                    return False
            
            # Initialize Enviro+ components (only in live mode)
            if not self.test_mode and ENVIRO_AVAILABLE:
                self._initialize_enviro()
            elif not self.test_mode:
                self.logger.warning("Enviro+ not available - using dummy sensors")
            
            self.logger.info("Hardware initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def _setup_gpio(self) -> None:
        """Setup GPIO for drill trigger."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(config.DRILL_TRIGGER_PIN, GPIO.OUT)
        GPIO.output(config.DRILL_TRIGGER_PIN, GPIO.LOW)
        self.logger.info(f"GPIO initialized - Drill trigger on pin {config.DRILL_TRIGGER_PIN}")
    
    def _initialize_camera(self) -> bool:
        """Initialize OAK camera."""
        try:
            self.camera = OakCamera()
            
            if self.test_mode:
                # Start camera in test mode
                if self.camera.start_test_mode():
                    self.logger.info("OAK camera initialized in test mode")
                    
                    # Initialize test mode processor
                    if self.test_input:
                        self.test_processor = TestModeProcessor(self.test_input)
                        self.test_display = TestModeDisplay()
                        self.logger.info(f"Test mode processor initialized with input: {self.test_input}")
                    
                    return True
                else:
                    self.logger.error("Failed to start camera in test mode")
                    return False
            else:
                # Start camera in live mode
                if self.camera.start():
                    self.logger.info("OAK camera initialized successfully")
                    return True
                else:
                    self.logger.error("Failed to start OAK camera")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False
    
    def _initialize_gcs(self) -> bool:
        """Initialize GCS client."""
        try:
            self.gcs_client = GCSClient()
            if self.gcs_client.test_connection():
                self.logger.info("GCS client initialized and connected")
                return True
            else:
                self.logger.warning("GCS client initialized but connection failed")
                return True  # Continue without GCS connection
        except Exception as e:
            self.logger.error(f"GCS client initialization failed: {e}")
            return False
    
    def _initialize_enviro(self) -> None:
        """Initialize Enviro+ sensors and display."""
        try:
            # Initialize BME280 sensor
            self.bme280_sensor = bme280
            
            # Initialize light sensor  
            self.light_sensor = ltr559
            
            # Initialize proximity sensor for display switching
            self.proximity_sensor = ltr559
            
            # Initialize LCD display
            self.lcd = ST7735.ST7735(
                port=0, cs=1, dc=9, backlight=12, rotation=270, spi_speed_hz=10000000
            )
            self.lcd.begin()
            
            self.logger.info("Enviro+ sensors and display initialized")
            
        except Exception as e:
            self.logger.error(f"Enviro+ initialization failed: {e}")
            # Continue without Enviro+ functionality
    
    def read_environmental_data(self) -> Optional[EnvironmentalData]:
        """
        Read current environmental sensor data.
        
        Returns:
            EnvironmentalData object with current readings
        """
        if not ENVIRO_AVAILABLE or self.test_mode:
            # Return dummy data in test mode or when Enviro+ unavailable
            return EnvironmentalData(
                timestamp=time.time(),
                temperature=20.0,
                humidity=50.0,
                pressure=1013.25,
                light=500.0,
                proximity=0,
                gas_readings={
                    'oxidising': 16000.0,
                    'reducing': 16000.0,
                    'nh3': 16000.0
                }
            )
        
        try:
            # Read BME280 data
            temp = self.bme280_sensor.get_temperature()
            pressure = self.bme280_sensor.get_pressure()
            humidity = self.bme280_sensor.get_humidity()
            
            # Read light and proximity
            light = self.light_sensor.get_lux()
            proximity = self.proximity_sensor.get_proximity()
            
            # Read gas sensor data
            gas_data = gas.read_all()
            
            return EnvironmentalData(
                timestamp=time.time(),
                temperature=temp,
                humidity=humidity,
                pressure=pressure,
                light=light,
                proximity=proximity,
                gas_readings=gas_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to read environmental data: {e}")
            return None
    
    def update_display_mode(self) -> None:
        """Update display mode based on proximity sensor."""
        if not self.proximity_sensor or self.test_mode:
            return
        
        try:
            proximity = self.proximity_sensor.get_proximity()
            if proximity > 1000:  # Object detected close to sensor
                self.display_mode = (self.display_mode + 1) % 4
                time.sleep(0.5)  # Debounce
        except Exception as e:
            self.logger.error(f"Proximity sensor error: {e}")
    
    def update_lcd_display(self, frame: Optional[np.ndarray] = None) -> None:
        """
        Update LCD display based on current mode.
        
        Args:
            frame: Current camera frame for display mode 1
        """
        if not self.lcd or self.test_mode:
            return
        
        try:
            # Create blank image
            img = Image.new('RGB', (160, 128), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            if self.display_mode == 0:
                # Display IP address
                draw.text((5, 5), "TAIP System", fill=(255, 255, 255))
                draw.text((5, 25), f"IP: {self._get_ip_address()}", fill=(255, 255, 255))
                draw.text((5, 45), f"Frames: {self.frames_processed}", fill=(255, 255, 255))
                
            elif self.display_mode == 1:
                # Display live camera feed (thumbnail)
                if frame is not None:
                    frame_small = cv2.resize(frame, (160, 120))
                    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    img_frame = Image.fromarray(frame_rgb)
                    img.paste(img_frame, (0, 4))
                    
            elif self.display_mode == 2:
                # Display sensor data
                if self.last_environmental_data:
                    env = self.last_environmental_data
                    draw.text((5, 5), f"T: {env.temperature:.1f}°C", fill=(255, 255, 255))
                    draw.text((5, 25), f"H: {env.humidity:.1f}%", fill=(255, 255, 255))
                    draw.text((5, 45), f"P: {env.pressure:.1f}hPa", fill=(255, 255, 255))
                    draw.text((5, 65), f"L: {env.light:.0f}lux", fill=(255, 255, 255))
                    
            elif self.display_mode == 3:
                # Display system status
                draw.text((5, 5), "System Status", fill=(255, 255, 255))
                draw.text((5, 25), f"Camera: {'OK' if self.camera else 'ERR'}", fill=(0, 255, 0) if self.camera else (255, 0, 0))
                draw.text((5, 45), f"GCS: {'OK' if self.gcs_client else 'ERR'}", fill=(0, 255, 0) if self.gcs_client else (255, 0, 0))
                if self.last_gauge_reading is not None:
                    draw.text((5, 65), f"Gauge: {self.last_gauge_reading:.1f}", fill=(255, 255, 255))
            
            # Display the image
            self.lcd.display(img)
            
        except Exception as e:
            self.logger.error(f"LCD display update failed: {e}")
    
    def _get_ip_address(self) -> str:
        """Get the device's IP address."""
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "Unknown"
    
    def trigger_drill(self) -> None:
        """Trigger the drill mechanism."""
        if self.test_mode:
            self.logger.info("TEST MODE: Drill trigger simulated")
            return
            
        try:
            self.logger.info("Triggering drill...")
            GPIO.output(config.DRILL_TRIGGER_PIN, GPIO.HIGH)
            time.sleep(config.DRILL_TRIGGER_DURATION)
            GPIO.output(config.DRILL_TRIGGER_PIN, GPIO.LOW)
            self.drill_active = True
            self.logger.info("Drill triggered successfully")
        except Exception as e:
            self.logger.error(f"Drill trigger failed: {e}")
    
    def process_frame(self) -> Optional[PayloadData]:
        """
        Process current camera frame and return payload data.
        
        Returns:
            PayloadData with detection results or None if processing failed
        """
        try:
            start_time = time.time()
            
            if self.test_mode and self.test_processor:
                # Process test mode frame
                frame = self.test_processor.get_next_frame()
                if frame is None:
                    self.logger.info("Test mode: No more frames to process")
                    self.running = False
                    return None
                
                # Get detections using test mode
                detections = self.camera.process_test_frame(frame)
                
                # Display test results
                if self.test_display:
                    processed_frame = draw_detections_on_frame(frame, detections)
                    self.test_display.show_frame(processed_frame, detections)
                    
                    # Check for exit key
                    if self.test_display.should_exit():
                        self.running = False
                        return None
                
            else:
                # Live mode processing
                frame = self.camera.get_latest_frame()
                if frame is None:
                    return None
                
                detections = self.camera.get_latest_detections()
            
            # Filter detections by confidence
            filtered_detections = filter_detections_by_confidence(
                detections, config.CONFIDENCE_THRESHOLD
            )
            
            # Calculate gauge reading
            gauge_reading = calculate_gauge_reading(filtered_detections)
            if gauge_reading is not None:
                self.last_gauge_reading = gauge_reading
            
            # Detect ArUco markers
            aruco_markers = detect_aruco_markers(frame)
            
            # Read environmental data (throttled)
            environmental_data = None
            current_time = time.time()
            if current_time - self.last_environmental_update > config.ENVIRONMENTAL_UPDATE_INTERVAL:
                environmental_data = self.read_environmental_data()
                if environmental_data:
                    self.last_environmental_data = environmental_data
                self.last_environmental_update = current_time
            
            # Update LCD display
            if not self.test_mode:
                self.update_lcd_display(frame)
            
            # Performance tracking
            processing_time = (time.time() - start_time) * 1000
            self.loop_times.append(processing_time)
            if len(self.loop_times) > 100:
                self.loop_times.pop(0)
            
            self.frames_processed += 1
            
            # Create payload data
            payload = PayloadData(
                timestamp=time.time(),
                frame_id=self.frames_processed,
                detections=filtered_detections,
                gauge_reading=self.last_gauge_reading,
                aruco_markers=aruco_markers,
                environmental_data=self.last_environmental_data,
                drill_active=self.drill_active,
                system_status=SystemStatus(
                    camera_active=True,
                    gcs_connected=self.gcs_client.is_connected() if self.gcs_client else False,
                    frames_processed=self.frames_processed,
                    average_processing_time=sum(self.loop_times) / len(self.loop_times) if self.loop_times else 0,
                    last_error=self.last_error
                )
            )
            
            return payload
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Frame processing failed: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def run(self) -> None:
        """Run the main TAIP processing loop."""
        try:
            mode_str = "test mode" if self.test_mode else "live mode"
            self.logger.info(f"Starting TAIP main loop in {mode_str}...")
            self.running = True
            self.start_time = time.time()
            
            # Main processing loop
            while self.running and not self.shutdown_event.is_set():
                loop_start = time.time()
                
                # Update display mode (only in live mode)
                if not self.test_mode:
                    self.update_display_mode()
                
                # Process current frame
                payload = self.process_frame()
                
                if payload and not self.test_mode:
                    # Send telemetry data to GCS (only in live mode)
                    if self.gcs_client:
                        self.gcs_client.send_data(payload)
                        
                        # Send frame to GCS
                        frame = self.camera.get_latest_frame()
                        if frame is not None:
                            self.gcs_client.send_frame(frame)
                
                # Maintain loop rate (only in live mode)
                if not self.test_mode:
                    loop_time = time.time() - loop_start
                    sleep_time = config.FRAME_INTERVAL - loop_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Cleanup completed GCS requests periodically
                if self.gcs_client and self.frames_processed % 10 == 0:
                    self.gcs_client.cleanup_completed_requests()
                
                # Log performance every 100 frames
                if self.frames_processed % 100 == 0:
                    avg_time = sum(self.loop_times) / len(self.loop_times) if self.loop_times else 0
                    self.logger.info(f"Performance: {self.frames_processed} frames, avg: {avg_time:.1f}ms")
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            self.logger.debug(traceback.format_exc())
        finally:
            self.running = False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
        self.running = False
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        try:
            self.logger.info("Cleaning up TAIP system...")
            
            # Stop test mode display
            if self.test_display:
                self.test_display.cleanup()
            
            # Stop camera
            if self.camera:
                self.camera.stop()
                self.logger.info("Camera stopped")
            
            # Stop GCS client
            if self.gcs_client:
                self.gcs_client.stop()
                self.logger.info("GCS client stopped")
            
            # Cleanup GPIO (only in live mode)
            if not self.test_mode:
                try:
                    GPIO.cleanup()
                    self.logger.info("GPIO cleaned up")
                except:
                    pass
            
            self.logger.info("TAIP system cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TAIP System - Target Acquisition and Image Processing')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--input', type=str, help='Test input path (video file or image directory)')
    return parser.parse_args()


def main():
    """Main entry point for TAIP system."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("EGH455 UAVPayloadTAQ - TAIP Subsystem")
    print("Target Acquisition and Image Processing System")
    if args.test:
        print(f"Running in TEST MODE with input: {args.input}")
    else:
        print("Running in LIVE MODE")
    print("=" * 60)
    
    # Validate configuration
    try:
        config.validate_config()
        print("✓ Configuration validation passed")
    except ValueError as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)
    
    # Create TAIP system
    taip_system = TAIPSystem(test_mode=args.test, test_input=args.input)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, taip_system.signal_handler)
    signal.signal(signal.SIGTERM, taip_system.signal_handler)
    
    try:
        # Initialize hardware
        if not taip_system.initialize_hardware():
            print("✗ Hardware initialization failed")
            sys.exit(1)
        
        print("✓ TAIP system initialized successfully")
        if not args.test:
            print(f"✓ GCS Target: {config.GCS_BASE_URL}")
            print(f"✓ Processing rate: {config.MAIN_LOOP_RATE} Hz")
        print("✓ System ready - Starting main loop...")
        
        # Run main processing loop
        taip_system.run()
        
    except Exception as e:
        print(f"✗ TAIP system error: {e}")
        logging.error(f"System error: {e}")
        logging.debug(traceback.format_exc())
        sys.exit(1)
    
    finally:
        taip_system.cleanup()
        print("TAIP system shutdown completed")


if __name__ == "__main__":
    main()
