"""
Configuration file for the TAIP (Target Acquisition and Image Processing) subsystem.
Contains all constants and configuration variables for the EGH455 UAVPayloadTAQ project.
"""

import os
from pathlib import Path

# =============================================================================
# SYSTEM PATHS
# =============================================================================
# Base project directory (TAIP folder is now the working directory)
PROJECT_ROOT = Path("/home/pi/EGH455")
TAIP_ROOT = Path("/home/pi/EGH455/TAIP")

# Model paths
MODEL_BLOB_PATH = PROJECT_ROOT / "models" / "blobs" / "YOLOv8s.blob"
MODEL_CONFIG_PATH = PROJECT_ROOT / "models" / "blobs" / "YOLOv8s.json"


# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================
# OAK-D Lite camera settings
CAMERA_PREVIEW_SIZE = (640, 640)  # Size for YOLO input
CAMERA_FPS = 30
CAMERA_QUEUE_SIZE = 4

# Video encoding settings for GCS transmission
VIDEO_QUALITY = 70  # JPEG quality (0-100)
VIDEO_MAX_WIDTH = 640
VIDEO_MAX_HEIGHT = 480

# =============================================================================
# TEST MODE CONFIGURATION
# =============================================================================
# Set TEST_INPUT_PATH to enable test mode instead of live camera
# Simply comment/uncomment the line you want to use:

# Live camera mode (default)
TEST_INPUT_PATH = None

# Test with all images in folder
# TEST_INPUT_PATH = PROJECT_ROOT / "models/testing/images"

# Test with specific videos (uncomment one to use)
# TEST_INPUT_PATH = PROJECT_ROOT / "models/testing/videos/far_blue.mp4"
# TEST_INPUT_PATH = PROJECT_ROOT / "models/testing/videos/far_silver_A.mp4"
# TEST_INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_blue_A.mp4"
# TEST_INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_blue_B.mp4"
# TEST_INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_silver_A.mp4"
# TEST_INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_silver_B.mp4"
# TEST_INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_silver_C.mp4"

# Test mode display settings
TEST_MODE_WINDOW_NAME = "TAIP Detection Test"
TEST_MODE_DISPLAY_TIME = 1000  # ms to display each image (0 = wait for keypress)
TEST_MODE_AUTO_ADVANCE = False  # True = auto advance, False = manual (press key)

# =============================================================================
# YOLO MODEL CONFIGURATION
# =============================================================================
# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.01

# Expected YOLO classes for the pressure gauge system (from YOLOv8s.json)
YOLO_CLASSES = {
    'Gauge_Centre': 0,
    'Needle_Tip': 1,
    'Valve_Closed': 2,
    'Valve_Open': 3
}

# =============================================================================
# GAUGE CALIBRATION
# =============================================================================
# Pressure gauge calibration parameters
# Angle range: -45° to 225° (270° total range)
# Pressure range: 10 to 0 bar (decreasing with clockwise rotation)
GAUGE_MIN_ANGLE = -45.0      # degrees (10 bar position)
GAUGE_MAX_ANGLE = 225.0      # degrees (0 bar position)
GAUGE_MIN_PRESSURE = 0.0     # bar (at max angle)
GAUGE_MAX_PRESSURE = 10.0    # bar (at min angle)

# Drilling threshold
DRILL_PRESSURE_THRESHOLD = 2.0  # bar - below this triggers drilling

# =============================================================================
# GPIO CONFIGURATION
# =============================================================================
# GPIO pin assignments
DRILL_TRIGGER_PIN = 18  # GPIO pin to send HIGH signal to DE subsystem
GPIO_MODE = "BCM"       # Use BCM pin numbering

# =============================================================================
# ARUCO MARKER CONFIGURATION
# =============================================================================
# ArUco detection settings
ARUCO_DICT = "DICT_4X4_250"  # Dictionary type
ARUCO_MARKER_SIZE = 0.05     # Marker size in meters (5cm)

# TODO: Camera calibration matrix for ArUco pose prediction (REQUIRES CALIBRATION)
CAMERA_MATRIX = [
    [640.0, 0.0, 320.0],
    [0.0, 640.0, 240.0],
    [0.0, 0.0, 1.0]
]

DISTORTION_COEFFICIENTS = [0.0, 0.0, 0.0, 0.0, 0.0]

# =============================================================================
# ENVIRO+ HAT CONFIGURATION
# =============================================================================
# LCD display settings
LCD_WIDTH = 160
LCD_HEIGHT = 80
LCD_ROTATION = 90

# Display modes for proximity sensor
DISPLAY_MODES = {
    'ip_address': 0,
    'live_feed': 1,
    'sensor_data': 2,
    'system_status': 3
}

# Proximity sensor thresholds for mode switching
PROXIMITY_THRESHOLD_NEAR = 1000   # Raw sensor value
PROXIMITY_THRESHOLD_FAR = 2000    # Raw sensor value

# Environmental sensor update intervals (seconds)
SENSOR_UPDATE_INTERVAL = 1.0
ENVIRONMENTAL_UPDATE_INTERVAL = 1.0  # How often to read environmental sensors

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
# Ground Control Station (GCS) settings
GCS_BASE_URL = "http://192.168.1.100:5000"  # Default GCS IP and port
GCS_TELEMETRY_ENDPOINT = "/telemetry"
GCS_FRAME_ENDPOINT = "/frame"

# Network timeouts
REQUEST_TIMEOUT = 2.0  # seconds
CONNECTION_TIMEOUT = 1.0  # seconds

# Data transmission settings
MAX_RETRIES = 3
RETRY_DELAY = 0.5  # seconds between retries

# =============================================================================
# SYSTEM TIMING
# =============================================================================
# Main loop timing
MAIN_LOOP_RATE = 10.0  # Hz (10 FPS for main processing)
FRAME_INTERVAL = 1.0 / MAIN_LOOP_RATE

# Performance requirements
MAX_PROCESSING_TIME = 0.4  # seconds (within 4-second requirement)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Log levels and formatting
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = TAIP_ROOT / "logs" / "taip_system.log"

# Create logs directory if it doesn't exist
LOG_FILE.parent.mkdir(exist_ok=True)

# =============================================================================
# DEVELOPMENT/DEBUG SETTINGS
# =============================================================================
# Debug flags
DEBUG_MODE = False
SAVE_DEBUG_IMAGES = False
DEBUG_IMAGE_PATH = TAIP_ROOT / "debug_images"

# Visualization settings
DRAW_BOUNDING_BOXES = True
DRAW_ARUCO_MARKERS = True
BBOX_COLOR = (0, 255, 0)  # Green for bounding boxes
BBOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White for text
TEXT_SCALE = 2.0

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def validate_config():
    """Validate configuration settings and file paths."""
    errors = []
    
    # Check if model files exist
    if not MODEL_BLOB_PATH.exists():
        errors.append(f"Blob file not found")
    
    # Validate angle ranges
    if GAUGE_MIN_ANGLE >= GAUGE_MAX_ANGLE:
        errors.append("GAUGE_MIN_ANGLE must be less than GAUGE_MAX_ANGLE")
    
    # Validate pressure ranges
    if GAUGE_MIN_PRESSURE >= GAUGE_MAX_PRESSURE:
        errors.append("GAUGE_MIN_PRESSURE must be less than GAUGE_MAX_PRESSURE")
    
    # Validate thresholds
    if not (0.0 <= CONFIDENCE_THRESHOLD <= 1.0):
        errors.append("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
    
    if not (0.0 <= IOU_THRESHOLD <= 1.0):
        errors.append("IOU_THRESHOLD must be between 0.0 and 1.0")
    
    # Validate network settings
    if REQUEST_TIMEOUT <= 0:
        errors.append("REQUEST_TIMEOUT must be positive")
    
    if MAX_RETRIES < 0:
        errors.append("MAX_RETRIES must be non-negative")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Auto-validate configuration on import
if __name__ != "__main__":
    validate_config()
