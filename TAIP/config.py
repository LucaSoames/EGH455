# /home/pi/EGH455/TAIP/config.py

"""
Configuration File for the EGH455 TAIP Subsystem

This file contains all the static configuration variables for the project,
including model paths, network settings, GPIO pins, and vision processing
calibration values. Centralizing these values makes the application easier
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

# --- Camera Configuration ---
CAMERA_PREVIEW_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# --- Model Configuration ---
MODEL_DIR = PROJECT_ROOT / "models/blobs"
BLOB_NAME = "YOLOv8s"
BLOB_PATH = MODEL_DIR / f"{BLOB_NAME}.blob"
CONFIG_PATH = MODEL_DIR / f"{BLOB_NAME}.json"

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
GAUGE_MIN_ANGLE_DEG = 225.0
GAUGE_MAX_ANGLE_DEG = -45.0
GAUGE_MIN_PRESSURE_BAR = 0.0
GAUGE_MAX_PRESSURE_BAR = 10.0
DRILL_PRESSURE_THRESHOLD = 2.0

# --- ArUco Configuration ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
ARUCO_MARKER_SIZE_M = 0.20

# --- Camera Calibration ---
CAMERA_MATRIX = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 320.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

# --- Test Mode Display ---
TEST_MODE_WINDOW_NAME = "TAIP Test Mode"
TEST_MODE_DISPLAY_TIME = 100  # ms
TEST_MODE_AUTO_ADVANCE = True