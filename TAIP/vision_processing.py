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
    Compute gauge pressure from needle and centre detections.
    Uses clockwise sweep from GAUGE_MIN_ANGLE_DEG (min pressure) to GAUGE_MAX_ANGLE_DEG (max pressure).
    Returns pressure in bar, or None if required detections are missing.
    """
    if not detections:
        return None

    # Pick highest-confidence centre and needle
    centres = sorted((d for d in detections if d.class_name == "Gauge_Centre"),
                     key=lambda d: d.confidence, reverse=True)
    tips = sorted((d for d in detections if d.class_name == "Needle_Tip"),
                  key=lambda d: d.confidence, reverse=True)
    if not centres or not tips:
        return None

    centre = centres[0]
    tip = tips[0]

    cx = (centre.box[0] + centre.box[2]) / 2.0
    cy = (centre.box[1] + centre.box[3]) / 2.0
    tx = (tip.box[0] + tip.box[2]) / 2.0
    ty = (tip.box[1] + tip.box[3]) / 2.0

    # Angle in image plane: 0° along +X, increasing counter‑clockwise
    angle_deg = (math.degrees(math.atan2(cy - ty, tx - cx)) + 360.0) % 360.0

    a_min = config.GAUGE_MIN_ANGLE_DEG % 360.0
    a_max = config.GAUGE_MAX_ANGLE_DEG % 360.0

    # Clockwise progress from min to current
    sweep_deg = config.GAUGE_SWEEP_DEG
    progress_deg = (a_min - angle_deg) % 360.0
    progress = progress_deg / sweep_deg
    # Clamp
    if progress < 0.0:
        progress = 0.0
    elif progress > 1.0:
        progress = 1.0

    return (config.GAUGE_MIN_PRESSURE_BAR +
            progress * (config.GAUGE_MAX_PRESSURE_BAR - config.GAUGE_MIN_PRESSURE_BAR))

def detect_aruco_markers(frame: np.ndarray,
                         camera_matrix: np.ndarray,
                         dist_coeffs: np.ndarray) -> List[ArucoDetection]:
    """
    Detect ArUco markers and estimate pose.
    Reuses the pre-initialised global detector to avoid per-frame allocations.
    Falls back to manual solvePnP if built-in pose estimation not available.
    """
    if frame is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    try:
        corners, ids, _ = _aruco_detector.detectMarkers(gray)
    except Exception:
        # Fallback old API
        corners, ids, _ = cv2.aruco.detectMarkers(gray, config.ARUCO_DICT)

    detections: List[ArucoDetection] = []
    if ids is None or len(ids) == 0:
        return detections

    s = float(config.ARUCO_MARKER_SIZE_M)
    obj_pts = np.array([
        [-s/2,  s/2, 0],
        [ s/2,  s/2, 0],
        [ s/2, -s/2, 0],
        [-s/2, -s/2, 0],
    ], dtype=np.float32)

    use_builtin = hasattr(cv2.aruco, "estimatePoseSingleMarkers")

    for idx, marker_id in enumerate(ids):
        c = corners[idx].reshape(-1, 2).astype(np.float32)
        try:
            if use_builtin:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([c], s, camera_matrix, dist_coeffs)
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
            else:
                flag = (cv2.SOLVEPNP_IPPE_SQUARE
                        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
                        else cv2.SOLVEPNP_ITERATIVE)
                ok, rvec, tvec = cv2.solvePnP(obj_pts, c, camera_matrix, dist_coeffs, flags=flag)
                if not ok:
                    continue
            detections.append(ArucoDetection(
                marker_id=int(marker_id[0]),
                position=tuple(map(float, tvec)),
                orientation=tuple(map(float, rvec))
            ))
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

def show_inference_visualisation(frame, detections, aruco_markers, gauge_pressure,
                                 camera_matrix=None, dist_coeffs=None):
    """Show visualisation window for file mode; draws pose axes if intrinsics given."""
    display_frame = draw_detections_on_frame(frame, detections, aruco_markers)
    if gauge_pressure is not None:
        cv2.putText(display_frame, f"Pressure: {gauge_pressure:.2f} bar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Detections: {len(detections)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Optional pose axes
    if camera_matrix is not None and dist_coeffs is not None and len(aruco_markers):
        axis_len = config.ARUCO_MARKER_SIZE_M * 0.5  # half marker size
        axis_obj = np.float32([
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len]
        ])
        for m in aruco_markers:
            try:
                rvec = np.array(m.orientation, dtype=np.float32).reshape(3, 1)
                tvec = np.array(m.position, dtype=np.float32).reshape(3, 1)
                imgpts, _ = cv2.projectPoints(axis_obj, rvec, tvec, camera_matrix, dist_coeffs)
                imgpts = imgpts.reshape(-1, 2).astype(int)
                o = tuple(imgpts[0])
                cv2.line(display_frame, o, tuple(imgpts[1]), (0, 0, 255), 2)   # X (red)
                cv2.line(display_frame, o, tuple(imgpts[2]), (0, 255, 0), 2)   # Y (green)
                cv2.line(display_frame, o, tuple(imgpts[3]), (255, 0, 0), 2)   # Z (blue)
            except Exception:
                continue

    cv2.imshow(config.TEST_MODE_WINDOW_NAME, display_frame)
    wait_ms = 0 if not config.TEST_MODE_AUTO_ADVANCE else config.TEST_MODE_DISPLAY_TIME
    key = cv2.waitKey(wait_ms) & 0xFF
    return key