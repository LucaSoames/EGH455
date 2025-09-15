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
PROJECT_ROOT = Path("/home/pi/EGH455")
TAIP_ROOT = PROJECT_ROOT / "TAIP"

# --- Mode Configuration ---
# Set to None to use the live OAK-D camera feed.
# Set path to a directory of images or a video file to run in testing mode.

# Production (live camera feed)
# INPUT_PATH = None

# Testing (images)
INPUT_PATH = PROJECT_ROOT / "models/testing/images"

# Testing (video)
# INPUT_PATH = PROJECT_ROOT / "models/testing/videos/far_blue.mp4"
# INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_blue_A.mp4"
# INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_silver_A.mp4"

# --- Camera Configuration ---
CAMERA_PREVIEW_SIZE = (640, 640)
CAMERA_FPS = 30  # Added: used by OAK pipeline
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# --- Model Configuration ---
MODEL_DIR = PROJECT_ROOT / "models/blobs"
BLOB_NAME = "YOLOv8n"
BLOB_PATH = MODEL_DIR / f"{BLOB_NAME}.blob"
CONFIG_PATH = MODEL_DIR / f"{BLOB_NAME}.json"

# Trained YOLO classes
YOLO_CLASSES = ["Gauge_Centre", "Needle_Tip", "Valve_Closed", "Valve_Open"]

# --- GCS (Ground Control Station) Communication ---
# The IP address of the laptop running the ground_station_server.py
# GCS_LAPTOP_IP = "192.168.1.100"
GCS_LAPTOP_IP = "127.0.0.1"
GCS_URL = f"http://{GCS_LAPTOP_IP}:5000"
POST_FRAME_FPS = 10
POST_TELEM_HZ = 5
REQUEST_TIMEOUT = 2.0

# --- GPIO Configuration ---
DRILL_GPIO_PIN = 18

# --- Gauge Calibration ---
# These values map needle angles to pressure readings.
# Convention:
#   GAUGE_MIN_ANGLE_DEG -> GAUGE_MIN_PRESSURE_BAR (e.g. 225° = 0 bar)
#   GAUGE_MAX_ANGLE_DEG -> GAUGE_MAX_PRESSURE_BAR (e.g. -45° (315°) = 10 bar)
# The needle sweeps CLOCKWISE from min->max over (min - max) % 360 degrees.
GAUGE_MIN_ANGLE_DEG = 225.0
GAUGE_MAX_ANGLE_DEG = -45.0
GAUGE_MIN_PRESSURE_BAR = 0.0
GAUGE_MAX_PRESSURE_BAR = 10.0
GAUGE_SWEEP_DEG = ( (GAUGE_MIN_ANGLE_DEG % 360) - (GAUGE_MAX_ANGLE_DEG % 360) ) % 360  # Expect 270°
if GAUGE_SWEEP_DEG == 0:
    GAUGE_SWEEP_DEG = 1e-6  # avoid divide by zero
DRILL_PRESSURE_THRESHOLD = 2.0   # Activate drill below this pressure

# --- ArUco Configuration ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
ARUCO_MARKER_SIZE_M = 0.20  # Physical size of markers in metres

# --- Select which camera sensor to use for ArUco pose estimation ---
#   'RGB'  → use the OAK-D colour camera (CAM_A)
#   'LEFT' → use the left mono camera (CAM_B)
CAMERA_ARUCO_SOURCE = 'RGB'   # 'LEFT' or 'RGB'

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

if CAMERA_ARUCO_SOURCE.upper() == 'RGB':
    CAMERA_MATRIX = CAMERA_MATRIX_RGB
    DISTORTION_COEFFS = DISTORTION_COEFFS_RGB
else:
    CAMERA_MATRIX = CAMERA_MATRIX_LEFT
    DISTORTION_COEFFS = DISTORTION_COEFFS_LEFT

# --- Test Mode Display ---
TEST_MODE_WINDOW_NAME = "TAIP Test Mode"
TEST_MODE_DISPLAY_TIME = 100  # ms
TEST_MODE_AUTO_ADVANCE = False

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
    if errors:
        print("Configuration issues:")
        for e in errors:
            print(" -", e)
        return False
    print("✓ Configuration valid")

    # Additional gauge sanity check
    if not (150.0 <= GAUGE_SWEEP_DEG <= 315.0):
        print(f"Warning: Unusual gauge sweep: {GAUGE_SWEEP_DEG:.1f} deg")
    return True