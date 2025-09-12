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
import os

# --- GCS (Ground Control Station) Communication ---
# The IP address of the laptop running the ground_station_server.py
# It's recommended to set this as an environment variable for flexibility.
GCS_LAPTOP_IP = os.environ.get("GCS_LAPTOP_IP", "192.168.1.100")
GCS_URL = f"http://{GCS_LAPTOP_IP}:5000"
POST_FRAME_FPS: int = 10  # Max frames per second to send to GCS
POST_TELEM_HZ: int = 5    # Max telemetry packets per second to send

# --- AI Model Configuration ---
MODEL_DIR = "/home/pi/EGH455/models/blobs/"
# The name of the model files (without extension).
# This allows for easy switching between different trained models.
BLOB_NAME = "YOLOv8s"
BLOB_PATH = os.path.join(MODEL_DIR, f"{BLOB_NAME}.blob")
CONFIG_PATH = os.path.join(MODEL_DIR, f"{BLOB_NAME}.json")

# --- GPIO Configuration ---
# The BCM pin number that will send a HIGH signal to the Drilling & Enclosure subsystem.
DRILL_GPIO_PIN: int = 17

# --- Vision Processing Calibration ---

# Gauge Reading Calibration
# Angle of the needle at the minimum and maximum pressure readings.
# 0 degrees is horizontal-right, positive is counter-clockwise.
# Refactored from drone_client.py and Roboflow blog.
GAUGE_MIN_ANGLE_DEG: float = 225.0  # Angle for min pressure (e.g., 0 bar)
GAUGE_MAX_ANGLE_DEG: float = -45.0  # Angle for max pressure (e.g., 10 bar)
GAUGE_MIN_PRESSURE_BAR: float = 0.0
GAUGE_MAX_PRESSURE_BAR: float = 10.0
DRILL_THRESHOLD_BAR: float = 2.0  # Pressure threshold to trigger drill signal

# ArUco Marker Configuration
# The dictionary should match the markers used in the environment.
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_200)
# Physical size of the ArUco marker in meters. Required for accurate pose estimation.
ARUCO_MARKER_SIZE_M: float = 0.20 # As per project spec: "exactly 200x200mm"

# --- Camera Intrinsics (IMPORTANT: CALIBRATE YOUR CAMERA) ---
# These are placeholder values. You MUST replace them with the actual calibration
# results for your specific OAK-D Lite camera to get accurate ArUco pose estimation.
#
# HOW TO CALIBRATE:
# 1. Print a chessboard pattern (e.g., 9x6 squares). You can find patterns online.
# 2. Using a script, capture 15-20 images of the chessboard from your OAK-D Lite
#    at various angles and distances.
# 3. Use OpenCV's `cv2.findChessboardCorners()` and `cv2.calibrateCamera()` functions
#    with these images to compute the camera matrix and distortion coefficients.
# 4. For a detailed guide, follow the official OpenCV tutorial:
#    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
#
CAMERA_MATRIX = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 320.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)  # Assuming no/minimal lens distortion for this application

# --- Development & Testing Configuration ---
# Set to a path to a directory of images, a single image file, or a video file
# to run in testing mode. Set to None to use the live OAK-D camera feed.
# This logic is refactored from your object_detection.py prototype.
# Example: "/home/pi/EGH455/testing/images/"
# Example: "/home/pi/EGH455/testing/videos/near_blue_A.mp4"
INPUT_PATH: str | None = None