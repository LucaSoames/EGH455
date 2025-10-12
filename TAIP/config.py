# /home/pi/EGH455/TAIP/config.py

"""
Configuration File for the EGH455 TAIP Subsystem

This file contains all the static configuration variables for the project,
including model paths, network settings, GPIO pins, and vision processing
calibration values. Centralising these values makes the application easier
to manage, test, and deploy.
"""

import cv2
import numpy as np
from pathlib import Path

# --- Project Structure ---
# Use relative paths that work on both Windows and Linux
TAIP_ROOT = Path(__file__).parent
PROJECT_ROOT = TAIP_ROOT.parent

# --- Mode Configuration ---
# Set to None to use the live OAK-D camera feed.
# Set path to a directory of images or a video file to run in testing mode.

# Production (live camera feed)
INPUT_PATH = None
SHOW_LIVE_VISUALISATION = False # Show live camera feed with detections overlayed on Pi

# Testing (images)
# INPUT_PATH = PROJECT_ROOT / "models/testing/images"

# Testing (video)
# INPUT_PATH = PROJECT_ROOT / "models/testing/videos/far_blue.mp4"
# INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_blue_B.mp4"
# INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_silver_A.mp4"

# --- Camera Configuration ---
CAMERA_PREVIEW_SIZE = (640, 640)
CAMERA_FPS = 10
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.01

# --- Model Configuration ---
MODEL_DIR = PROJECT_ROOT / "models/blobs"
BLOB_NAME = "YOLOv8n"
BLOB_PATH = MODEL_DIR / f"{BLOB_NAME}.blob"
CONFIG_PATH = MODEL_DIR / f"{BLOB_NAME}.json"

# --- Visualisation Configuration ---
DETECTION_TEXT_SIZE = 1.0  # Font scale for detection labels
DETECTION_TEXT_THICKNESS = 2  # Text thickness for better visibility
DETECTION_COLOURS = {
    "Valve_Open": (0, 255, 0),      # Green
    "Valve_Closed": (0, 0, 255),    # Red
    "Needle_Tip": (0, 255, 255),    # Yellow
    "Gauge_Centre": (255, 0, 0),    # Blue
    "default": (0, 0, 0)            # Black 
}

# Trained YOLO classes
# YOLO_CLASSES = ["Gauge_Centre", "Needle_Tip", "Valve_Closed", "Valve_Open"]

# --- GCS (Ground Control Station) Communication ---
# The IP address of the laptop running the ground_station_server.py
GCS_LAPTOP_IP = "192.168.86.24"
# GCS_LAPTOP_IP = "127.0.0.1"
GCS_URL = f"http://{GCS_LAPTOP_IP}:3000"
POST_FRAME_FPS = 10
POST_TELEM_HZ = 5
REQUEST_TIMEOUT = 2.0

# --- Gauge Calibration ---
# These values map needle angles to pressure readings.
# Convention:
#   GAUGE_MIN_ANGLE_DEG -> GAUGE_MIN_PRESSURE_BAR (e.g. 225° = 0 bar)
#   GAUGE_MAX_ANGLE_DEG -> GAUGE_MAX_PRESSURE_BAR (e.g. -45° (315°) = 10 bar)
# The needle sweeps CLOCKWISE from min->max over (min - max) % 360 degrees.
GAUGE_READING_OFFSET = 0.045 # Offset to add to gauge reading (bar) 
GAUGE_MIN_ANGLE_DEG = 222.0
GAUGE_MAX_ANGLE_DEG = -48.0
GAUGE_MIN_PRESSURE_BAR = 0.0
GAUGE_MAX_PRESSURE_BAR = 10.0
GAUGE_SWEEP_DEG = ( (GAUGE_MIN_ANGLE_DEG % 360) - (GAUGE_MAX_ANGLE_DEG % 360) ) % 360  # Expect 270°

SHOW_GAUGE_OVERLAY = True  # Overlay gauge calibration on output image

# --- Drilling Subsystem Configuration ---
DRILL_GPIO_PIN = 13
DRILL_PRESSURE_THRESHOLD = 2.0  # Activate drill below this pressure
DRILL_DURATION_SEC = 45.0       # Duration to run drill once activated
DRILL_TRIGGER_COUNT = 3         # Hysteresis parameter (consecutive readings required to trigger)
PWM_FREQUENCY = 50              # Hz
STOP_DUTY = 7.5                 # ~1.5ms pulse - stop
CCW_DUTY = 2.0                  # ~0.5ms pulse - drilling (CCW)
CW_DUTY = 13.0                  # ~2.5 ms pulse — full clockwise

# --- ArUco Marker Configuration ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
ARUCO_MARKER_SIZE_M = 0.20  # Physical size of markers in metres

# We now always use the LEFT mono camera (CAM_B) for ArUco pose estimation.
# Keep both intrinsics for correctness in test mode (file inputs use RGB frames).
# --- Camera Calibration (RGB camera, CAM_A socket) ---
CAMERA_MATRIX_RGB = np.array([
    [2968.344482421875,    0.0,               1898.425048828125],
    [   0.0,             2968.344482421875,   1158.2735595703125],
    [   0.0,                0.0,                 1.0]
], dtype=np.float32)

DISTORTION_COEFFS_RGB = np.array([
    -3.0139496326446533,
    -1.5687170028686523,
    -0.0007319296128116548,
    -0.0005127631011418998,
     33.57282257080078
], dtype=np.float32).reshape(-1, 1)

# --- Camera Calibration (left mono camera, CAM_B socket) ---
CAMERA_MATRIX_LEFT = np.array([
    [452.614501953125,   0.0,              307.8687438964844],
    [0.0,                452.614501953125, 233.67922973632813],
    [0.0,                0.0,              1.0]
], dtype=np.float32)

DISTORTION_COEFFS_LEFT = np.array([
    3.8201379776000977,
   -52.87614059448242,
   -0.0017299839528277516,
    0.0011204829206690192,
  218.44808959960938
], dtype=np.float32).reshape(-1, 1)

# Removed CAMERA_ARUCO_SOURCE flag and derived CAMERA_MATRIX/DISTORTION_COEFFS
# to avoid runtime camera switching ambiguity. The app will use:
#  - LEFT intrinsics for live ArUco (mono frame)
#  - RGB intrinsics automatically in test/file mode.

# --- Test Mode Display ---
TEST_MODE_WINDOW_NAME = "TAIP Test Mode Visualisation"
TEST_MODE_DISPLAY_TIME = 100  # ms

# --- Gas Sensor Calibration (from display_ip.py) ---
# Baseline resistance values in clean air (Ohms)
RO_RED = 451379.96      # Reducing gases (CO, H2S, NH3)
RO_OX = 11485.55        # Oxidising gases (NO2, NO, O3)
RO_NH3 = 347942.92      # Ammonia (NH3)

# Linear coefficients for PPM conversion: ppm = A * (Rs/Ro) + B
# Derived from MiCS-6814 datasheet graphs
A_RED, B_RED = 300.0,  -300.0   # CO estimation
A_OX,  B_OX  = 0.25,   -0.25    # NO2 estimation
A_NH3, B_NH3 = -3.0,     3.0    # NH3 estimation

def validate_config():
    """Basic configuration validation for test scripts."""
    errors = []
    if not BLOB_PATH.exists():
        errors.append(f"Model blob missing: {BLOB_PATH}")
    if not CONFIG_PATH.exists():
        errors.append(f"Model config JSON missing: {CONFIG_PATH}")
    if GAUGE_MIN_PRESSURE_BAR >= GAUGE_MAX_PRESSURE_BAR:
        errors.append("Gauge pressure range invalid")
    if DRILL_PRESSURE_THRESHOLD < 0 or DRILL_PRESSURE_THRESHOLD > GAUGE_MAX_PRESSURE_BAR:
        errors.append("Drill pressure threshold out of range")
    # Basic intrinsics sanity
    for name, K in [("RGB", CAMERA_MATRIX_RGB), ("LEFT", CAMERA_MATRIX_LEFT)]:
        if K.shape != (3,3):
            errors.append(f"Camera matrix shape invalid for {name}")

    return True