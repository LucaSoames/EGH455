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
from typing import List, Optional, Tuple

import config
from data_models import YoloDetection, ArucoDetection

# Create ArUco dictionary and detector with default parameters
_aruco_params = cv2.aruco.DetectorParameters()
_aruco_detector = cv2.aruco.ArucoDetector(config.ARUCO_DICT, _aruco_params)

def calculate_gauge_reading(detections: List[YoloDetection]) -> Optional[float]:
    """
    Derive gauge pressure from needle + centre detections.
    Expects relative (0..1) bounding boxes; angular maths is scale independent.
    Returns pressure in bar or None if required detections missing.
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
    Detect ArUco markers and estimate pose.
    Falls back to manual solvePnP if cv2.aruco.estimatePoseSingleMarkers
    is not available in the installed OpenCV build.
    """
    if frame is None:
        return []

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    corners, ids, _ = _aruco_detector.detectMarkers(gray)
    detections: List[ArucoDetection] = []
    if ids is None or len(ids) == 0:
        return detections

    # Marker 3D corner coordinates (centre at origin, Z=0 plane)
    s = float(config.ARUCO_MARKER_SIZE_M)
    obj_pts = np.array([
        [-s/2,  s/2, 0],
        [ s/2,  s/2, 0],
        [ s/2, -s/2, 0],
        [-s/2, -s/2, 0],
    ], dtype=np.float32)

    has_builtin = hasattr(cv2.aruco, "estimatePoseSingleMarkers")

    for i, marker_id in enumerate(ids):
        c = corners[i].astype(np.float32).reshape(-1, 2)

        try:
            if has_builtin:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [c], s, camera_matrix, dist_coeffs
                )
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
            else:
                # Manual PnP
                # Order: must match obj_pts ordering (OpenCV corner order is TL, TR, BR, BL)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    c,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                    if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    continue
            detections.append(
                ArucoDetection(
                    marker_id=int(marker_id[0]),
                    position=tuple(map(float, tvec)),
                    orientation=tuple(map(float, rvec))
                )
            )
        except Exception:
            continue

    return detections

def draw_detections_on_frame(frame: np.ndarray,
                             detections: List[YoloDetection],
                             aruco_markers: List[ArucoDetection]) -> np.ndarray:
    """
    Visual overlay for debugging. Bounding boxes assume relative coords [0,1].
    """
    out = frame.copy()
    h, w = frame.shape[:2]

    for det in detections:
        x1 = int(det.box[0] * w)
        y1 = int(det.box[1] * h)
        x2 = int(det.box[2] * w)
        y2 = int(det.box[3] * h)
        colour = (0, 255, 0) if det.class_name in ('Gauge_Centre', 'Needle_Tip') else (255, 0, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, f"{det.class_name}:{det.confidence:.2f}",
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    for idx, marker in enumerate(aruco_markers):
        text = f"ID {marker.marker_id}"
        cv2.putText(out, text,
                    (10, 20 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out

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
    """Show visualisation window for file mode (blocks if TEST_MODE_AUTO_ADVANCE=False)."""
    display_frame = draw_detections_on_frame(frame, detections, aruco_markers)
    if gauge_pressure is not None:
        cv2.putText(display_frame, f"Pressure: {gauge_pressure:.2f} bar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Detections: {len(detections)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("TAIP File Mode Visualisation", display_frame)
    wait_ms = 0 if not config.TEST_MODE_AUTO_ADVANCE else config.TEST_MODE_DISPLAY_TIME
    key = cv2.waitKey(wait_ms) & 0xFF
    return key