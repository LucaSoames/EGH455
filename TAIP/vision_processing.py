# /home/pi/EGH455/TAIP/vision_processing.py

"""
Vision Processing Module for the EGH455 TAIP Subsystem

This module contains pure functions for processing video frames and raw
model outputs to extract meaningful information like gauge pressure and
ArUco marker poses.
"""

import collections
import cv2
import numpy as np
import math
from typing import List, Optional

import config
from data_models import YoloDetection, ArucoDetection

# Store last 5 gauge readings for smoothing
_last_readings = collections.deque(maxlen=5)

# Create ArUco dictionary and detector with default parameters
_aruco_params = cv2.aruco.DetectorParameters()
_aruco_detector = cv2.aruco.ArucoDetector(config.ARUCO_DICT, _aruco_params)

_ARUCO_OBJ_PTS = None

def calculate_gauge_reading(detections: List[YoloDetection]) -> Optional[float]:
    """
    Computes gauge pressure from needle and centre detections.

    Coordinate system:
      Image +X → right, +Y → down.
      We compute the angle with atan2(cy - ty, tx - cx) so that:
        - 0° points to the right (+X)
        - angles increase counter-clockwise
      The gauge is defined (in config.py) with a CLOCKWISE sweep from GAUGE_MIN_ANGLE_DEG
      (minimum pressure) to GAUGE_MAX_ANGLE_DEG (maximum pressure).

    This function converts the current absolute angle of the needle into a clockwise
    progress percentage along the calibrated sweep to determine the pressure.
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

    # Centre of Gauge_Centre bounding boxe
    cx = (centre.box[0] + centre.box[2]) / 2.0
    cy = (centre.box[1] + centre.box[3]) / 2.0
    
    # Centre of Needle_Tip bounding box
    tx = (tip.box[0] + tip.box[2]) / 2.0
    ty = (tip.box[1] + tip.box[3]) / 2.0
    
    # TESTING: Top-left corner of Needle_Tip bounding box
    # tx = tip.box[0]  # Left edge (xmin)
    # ty = tip.box[1]  # Top edge (ymin)

    # Angle in image plane: 0° along +X, increasing counter‑clockwise
    angle_deg = (math.degrees(math.atan2(cy - ty, tx - cx)) + 360.0) % 360.0

    a_min = config.GAUGE_MIN_ANGLE_DEG % 360.0
    a_max = config.GAUGE_MAX_ANGLE_DEG % 360.0

    # Clockwise progress from min to current
    sweep_deg = config.GAUGE_SWEEP_DEG
    progress_deg = (a_min - angle_deg) % 360.0
    progress = progress_deg / sweep_deg
    progress = max(0.0, min(1.0, progress))  # clamp
    
    raw_pressure = (config.GAUGE_MIN_PRESSURE_BAR +
                progress * (config.GAUGE_MAX_PRESSURE_BAR - config.GAUGE_MIN_PRESSURE_BAR))
        
    # Apply smoothing (average of 5 most recent readings)
    global _last_readings
    _last_readings.append(raw_pressure)

    return sum(_last_readings) / len(_last_readings) + config.GAUGE_READING_OFFSET


def detect_aruco_markers(frame: np.ndarray,
                         camera_matrix: np.ndarray,
                         dist_coeffs: np.ndarray) -> List[ArucoDetection]:
    """
    Detects ArUco markers and estimates their pose.
    Reuses the pre-initialised global detector to avoid per-frame allocations.
    """
    global _ARUCO_OBJ_PTS
    if _ARUCO_OBJ_PTS is None:
        s = float(config.ARUCO_MARKER_SIZE_M)
        _ARUCO_OBJ_PTS = np.array([
            [-s/2,  s/2, 0],
            [ s/2,  s/2, 0],
            [ s/2, -s/2, 0],
            [-s/2, -s/2, 0],
        ], dtype=np.float32)

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
                ok, rvec, tvec = cv2.solvePnP(_ARUCO_OBJ_PTS, c, camera_matrix, dist_coeffs, flags=flag)
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
    Creates a visual overlay for debugging purposes. Bounding boxes assume relative coords [0,1].
    """
    out = frame.copy()
    h, w = frame.shape[:2]

    for det in detections:
        x1 = int(det.box[0] * w)
        y1 = int(det.box[1] * h)
        x2 = int(det.box[2] * w)
        y2 = int(det.box[3] * h)
        
        # Get the colour from config, with a fallback to the default colour
        colour = config.DETECTION_COLOURS.get(det.class_name, config.DETECTION_COLOURS['default'])
        
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, f"{det.class_name}:{det.confidence:.2f}",
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 
                    config.DETECTION_TEXT_SIZE, colour, config.DETECTION_TEXT_THICKNESS, cv2.LINE_AA)

    for idx, marker in enumerate(aruco_markers):
        text = f"ID {marker.marker_id}"
        cv2.putText(out, text,
                    (10, 20 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    config.DETECTION_TEXT_SIZE * 0.3, (0, 255, 255), config.DETECTION_TEXT_THICKNESS, cv2.LINE_AA)
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
    """Shows a visualisation window for test mode; draws pose axes if intrinsics are given."""
    display_frame = draw_detections_on_frame(frame, detections, aruco_markers)
    
    if config.SHOW_GAUGE_OVERLAY:
        display_frame = draw_angle_debug(display_frame, detections)
    
    h, w = frame.shape[:2]
    
    if gauge_pressure is not None:
        # Position gauge reading in top right corner
        text = f"Pressure: {gauge_pressure:.2f} bar"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, config.DETECTION_TEXT_THICKNESS)[0]
        text_x = w - text_size[0] - 10  # 10 pixels from right edge
        cv2.putText(display_frame, text,
                    (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, 
                    (0, 255, 0), config.DETECTION_TEXT_THICKNESS)
    
    # Keep detection count in top left corner
    cv2.putText(display_frame, f"Detections: {len(detections)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, 
                (255, 255, 255), config.DETECTION_TEXT_THICKNESS)

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


# VISUALISE GAUGE ANGLE CALCULATION AND MIN/MAX LIMITS (set SHOW_GAUGE_OVERLAY=True in config.py)
def debug_gauge_angles(detections: List[YoloDetection]) -> Optional[dict]:
    """Debug function to analyse gauge angle calculations."""
    centres = sorted((d for d in detections if d.class_name == "Gauge_Centre"),
                     key=lambda d: d.confidence, reverse=True)
    tips = sorted((d for d in detections if d.class_name == "Needle_Tip"),
                  key=lambda d: d.confidence, reverse=True)
    
    if not centres or not tips:
        return None
    
    centre, tip = centres[0], tips[0]
    cx = (centre.box[0] + centre.box[2]) / 2.0
    cy = (centre.box[1] + centre.box[3]) / 2.0
    tx = (tip.box[0] + tip.box[2]) / 2.0
    ty = (tip.box[1] + tip.box[3]) / 2.0
    
    angle_deg = (math.degrees(math.atan2(cy - ty, tx - cx)) + 360.0) % 360.0
    
    return {
        "needle_angle": angle_deg,
        "min_angle": config.GAUGE_MIN_ANGLE_DEG,
        "max_angle": config.GAUGE_MAX_ANGLE_DEG,
        "sweep_deg": config.GAUGE_SWEEP_DEG,
        "centre_pos": (cx, cy),
        "tip_pos": (tx, ty)
    }

def draw_angle_debug(frame: np.ndarray, detections: List[YoloDetection]) -> np.ndarray:
    """Draw angle lines for calibration debugging."""
    debug_info = debug_gauge_angles(detections)
    if not debug_info:
        return frame
    
    h, w = frame.shape[:2]
    cx = int(debug_info["centre_pos"][0] * w)
    cy = int(debug_info["centre_pos"][1] * h)
    tx = int(debug_info["tip_pos"][0] * w)
    ty = int(debug_info["tip_pos"][1] * h)
    
    # Draw needle line
    cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 255), 3)
    
    # Draw lines to min max points
    radius = 300
    
    # Draw min angle line (0 bar)
    min_rad = math.radians(config.GAUGE_MIN_ANGLE_DEG)
    min_x = int(cx + radius * math.cos(min_rad))
    min_y = int(cy - radius * math.sin(min_rad))  # Note: Y is flipped
    cv2.line(frame, (cx, cy), (min_x, min_y), (0, 255, 0), 2)
    
    # Position text below the end of the min angle line
    text_offset_y = 40  # Pixels below the line end
    cv2.putText(frame, "0 bar", (min_x - 60, min_y + text_offset_y), 
                cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, (0, 255, 0), 2)
    
    # Draw max angle line (10 bar)
    max_rad = math.radians(config.GAUGE_MAX_ANGLE_DEG)
    max_x = int(cx + radius * math.cos(max_rad))
    max_y = int(cy - radius * math.sin(max_rad))  # Note: Y is flipped
    cv2.line(frame, (cx, cy), (max_x, max_y), (0, 0, 255), 2)
    
    # Position text below the end of the max angle line
    cv2.putText(frame, "10 bar", (max_x, max_y + text_offset_y), 
                cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, (0, 0, 255), 2)
    
    # Show current needle angle in top right corner
    text = f"Angle: {debug_info['needle_angle']:.1f} deg"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, config.DETECTION_TEXT_THICKNESS)[0]
    text_x = w - text_size[0] - 10  # 10 pixels from right edge
    cv2.putText(frame, text,
                (text_x, 150), cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, 
                (0, 0, 0), config.DETECTION_TEXT_THICKNESS)
    
    return frame