# /home/pi/EGH455/TAIP/vision_processing.py

"""
Vision Processing Module for the EGH455 TAIP Subsystem

This module contains pure functions for processing video frames and raw
model outputs to extract meaningful information like gauge pressure and
ArUco marker poses.
"""

import cv2
import numpy as np
import math
import json
import depthai as dai
from typing import List, Optional

# Import our custom data models and configuration
import config
from data_models import YoloDetection, ArucoDetection

# Initialise the ArUco detector once as a module-level object for efficiency
_aruco_detector = cv2.aruco.ArucoDetector(config.ARUCO_DICT)

def calculate_gauge_reading(detections: List[YoloDetection]) -> Optional[float]:
    """
    Calculates the pressure from a gauge based on YOLO detections.

    This function finds the center of the 'Gauge_Centre' and 'Needle_Tip'
    bounding boxes, calculates the angle of the needle, and maps this
    angle to a pressure value using pre-defined calibration constants.

    Args:
        detections: A list of YoloDetection objects for a single frame.

    Returns:
        The calculated pressure in bar as a float, or None if the required
        markers are not found.
    """
    # Find the highest confidence needle tip and gauge center
    needle_tip = max([d for d in detections if d.class_name == 'Needle_Tip'], 
                     key=lambda x: x.confidence, default=None)
    gauge_centre = max([d for d in detections if d.class_name == 'Gauge_Centre'], 
                       key=lambda x: x.confidence, default=None)

    if not needle_tip or not gauge_centre:
        return None

    # Calculate the center point of each bounding box
    tip_x = (needle_tip.box[0] + needle_tip.box[2]) / 2.0
    tip_y = (needle_tip.box[1] + needle_tip.box[3]) / 2.0
    centre_x = (gauge_centre.box[0] + gauge_centre.box[2]) / 2.0
    centre_y = (gauge_centre.box[1] + gauge_centre.box[3]) / 2.0

    # Calculate angle of the needle vector (tip -> centre)
    # Note: atan2(y, x) but image y is inverted, so we use (centre_y - tip_y)
    angle_rad = math.atan2(centre_y - tip_y, tip_x - centre_x)
    angle_deg = math.degrees(angle_rad)

    # --- Angle to Pressure Mapping ---
    # Normalise angles to be consistently positive (0-360)
    a0 = (config.GAUGE_MIN_ANGLE_DEG + 360) % 360
    a1 = (config.GAUGE_MAX_ANGLE_DEG + 360) % 360
    angle_deg_norm = (angle_deg + 360) % 360
    
    # Handle angle wrapping (e.g., span from 315 deg to 45 deg)
    span = (a1 - a0 + 360) % 360
    pos = (angle_deg_norm - a0 + 360) % 360

    if span == 0:
        return config.GAUGE_MIN_PRESSURE_BAR

    # Calculate interpolation factor (t) and clamp between 0 and 1
    t = max(0.0, min(1.0, pos / span))
    
    pressure = config.GAUGE_MIN_PRESSURE_BAR + t * (config.GAUGE_MAX_PRESSURE_BAR - config.GAUGE_MIN_PRESSURE_BAR)
    
    return pressure

def detect_aruco_markers(
    frame: np.ndarray, 
    camera_matrix: np.ndarray, 
    dist_coeffs: np.ndarray
) -> List[ArucoDetection]:
    """
    Detects ArUco markers in a frame and estimates their pose.

    Args:
        frame: The input video frame (as a NumPy array).
        camera_matrix: The camera's intrinsic calibration matrix.
        dist_coeffs: The camera's distortion coefficients.

    Returns:
        A list of ArucoDetection objects, one for each marker found.
    """    
    # Detect markers
    corners, ids, _ = _aruco_detector.detectMarkers(frame)

    detected_markers: List[ArucoDetection] = []
    if ids is not None and len(ids) > 0:
        # Pose estimation requires float32 corners
        corners_float = [c.astype(np.float32) for c in corners]
        # Estimate pose for each detected marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners_float, 
            config.ARUCO_MARKER_SIZE_M, 
            camera_matrix, 
            dist_coeffs
        )

        # Populate the list of detected markers
        for i, marker_id in enumerate(ids):
            detected_markers.append(
                ArucoDetection(
                    marker_id=int(marker_id[0]),
                    position=tuple(tvecs[i][0]),
                    orientation=tuple(rvecs[i][0])
                )
            )
            
    return detected_markers

def draw_detections_on_frame(frame: np.ndarray, 
                           detections: List[YoloDetection], 
                           aruco_markers: List[ArucoDetection]) -> np.ndarray:
    """
    Draw detection results on a frame for visualization.
    
    Args:
        frame: Input frame
        detections: YOLO detections to draw
        aruco_markers: ArUco markers to draw
        
    Returns:
        Frame with detections drawn
    """
    result_frame = frame.copy()
    
    # Draw YOLO detections
    for detection in detections:
        # Convert relative coordinates to absolute
        h, w = frame.shape[:2]
        x1 = int(detection.box[0] * w)
        y1 = int(detection.box[1] * h)
        x2 = int(detection.box[2] * w)
        y2 = int(detection.box[3] * h)
        
        # Draw bounding box
        color = (0, 255, 0) if detection.class_name in ['Gauge_Centre', 'Needle_Tip'] else (255, 0, 0)
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        cv2.putText(result_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw ArUco markers
    for marker in aruco_markers:
        # Draw marker ID
        cv2.putText(result_frame, f"ArUco {marker.marker_id}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return result_frame

def validate_gauge_calibration() -> bool:
    """Validate gauge calibration parameters."""
    try:
        # Check angle range
        angle_range = abs(config.GAUGE_MAX_ANGLE_DEG - config.GAUGE_MIN_ANGLE_DEG)
        if angle_range < 90 or angle_range > 300:
            print(f"Warning: Unusual gauge angle range: {angle_range:.1f} degrees")
            return False
            
        # Check pressure range
        pressure_range = config.GAUGE_MAX_PRESSURE_BAR - config.GAUGE_MIN_PRESSURE_BAR
        if pressure_range <= 0:
            print("Error: Invalid pressure range")
            return False
            
        print("✓ Gauge calibration valid")
        return True
        
    except Exception as e:
        print(f"Error validating gauge calibration: {e}")
        return False

def show_inference_visualisation(frame, detections, aruco_markers, gauge_pressure):
    """Show visualisation window for file mode."""
    display_frame = draw_detections_on_frame(frame, detections, aruco_markers)
    
    # Add gauge reading overlay
    if gauge_pressure is not None:
        cv2.putText(display_frame, f"Pressure: {gauge_pressure:.2f} bar", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add detection count
    cv2.putText(display_frame, f"Detections: {len(detections)}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow("TAIP File Mode Visualisation", display_frame)
    key = cv2.waitKey(config.TEST_MODE_DISPLAY_TIME) & 0xFF
    return key