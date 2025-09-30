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

# Create ArUco dictionary and detector with enhanced parameters
_aruco_params = cv2.aruco.DetectorParameters()
# Improve ArUco detection parameters
_aruco_params.adaptiveThreshWinSizeMin = 3
_aruco_params.adaptiveThreshWinSizeMax = 23
_aruco_params.adaptiveThreshWinSizeStep = 2
_aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_aruco_params.cornerRefinementWinSize = 5
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
    average_pressure = sum(_last_readings) / len(_last_readings)
    
    return average_pressure + config.GAUGE_READING_OFFSET


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

    # The detectMarkers function signature changed in OpenCV 4.7.
    # We use a try-except block to handle both new and old versions.
    try:
        # OpenCV 4.7+ returns corners, ids, rejected
        corners, ids, _ = _aruco_detector.detectMarkers(gray)
    except ValueError:
        # Older OpenCV versions returned corners, ids, rejected, recovered
        corners, ids, _, _ = _aruco_detector.detectMarkers(gray)
    except Exception:
        # Fallback to the older cv2.aruco.detectMarkers function if the detector object fails
        corners, ids, _ = cv2.aruco.detectMarkers(gray, config.ARUCO_DICT, parameters=_aruco_params)


    detections: List[ArucoDetection] = []
    if ids is None or len(ids) == 0:
        return detections, None, None

    # The estimatePoseSingleMarkers function is reliable and preferred.
    # We solvePnP manually only as a fallback.
    use_builtin_pose_estimator = hasattr(cv2.aruco, "estimatePoseSingleMarkers")
    s = float(config.ARUCO_MARKER_SIZE_M)

    for i, marker_id in enumerate(ids):
        marker_corners = corners[i].reshape(-1, 2).astype(np.float32)
        try:
            if use_builtin_pose_estimator:
                # Use the recommended built-in function for pose estimation
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [marker_corners], s, camera_matrix, dist_coeffs
                )
                rvec, tvec = rvecs[0][0], tvecs[0][0]
            else:
                # Manual fallback using solvePnP if the built-in is not available
                ret, rvec, tvec = cv2.solvePnP(
                    _ARUCO_OBJ_PTS, marker_corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not ret:
                    continue
            
            detections.append(ArucoDetection(
                marker_id=int(marker_id[0]),
                position=tuple(map(float, tvec)),
                orientation=tuple(map(float, rvec))
            ))
        except Exception:
            continue

    return detections, corners, ids

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

    # Display ArUco marker ID and position
    for idx, marker in enumerate(aruco_markers):
        pos = marker.position
        text = f"ID {marker.marker_id}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m"
        cv2.putText(out, text, (10, 120 + idx * 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    config.DETECTION_TEXT_SIZE * 0.5, (0, 255, 255), config.DETECTION_TEXT_THICKNESS, cv2.LINE_AA)
    
    return out


def show_inference_visualisation(frame, detections, aruco_markers, aruco_corners, aruco_ids, gauge_pressure,
                                 camera_matrix=None, dist_coeffs=None):
    """Shows a visualisation window for test mode; draws pose axes if intrinsics are given."""
    display_frame = draw_detections_on_frame(frame, detections, aruco_markers)
    
    if config.SHOW_GAUGE_OVERLAY:
        display_frame = draw_gauge_debug(display_frame, detections)
    
    h, w = frame.shape[:2]
    
    if gauge_pressure is not None:
        # Position gauge reading in top right corner
        text = f"Pressure: {gauge_pressure:.2f} bar"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, config.DETECTION_TEXT_THICKNESS)[0]
        text_x = w - text_size[0] - 10  # 10 pixels from right edge
        cv2.putText(display_frame, text,
                    (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, 
                    (255, 255, 255), config.DETECTION_TEXT_THICKNESS)
    
    # Keep detection count in top left corner
    cv2.putText(display_frame, f"Detections: {len(detections)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, 
                (255, 255, 255), config.DETECTION_TEXT_THICKNESS)

    # Draw ArUco bounding boxes and pose axes
    if aruco_corners is not None and aruco_ids is not None:
        # The detection and display frames are now the same, so no scaling is needed.
        cv2.aruco.drawDetectedMarkers(display_frame, aruco_corners, aruco_ids)
        
        if camera_matrix is not None and dist_coeffs is not None and len(aruco_markers) > 0:
            # Get rvecs and tvecs for drawing the axes
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                aruco_corners, config.ARUCO_MARKER_SIZE_M, camera_matrix, dist_coeffs
            )
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(display_frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1, 3)

    cv2.imshow(config.TEST_MODE_WINDOW_NAME, display_frame)
    wait_ms = 0 if not config.TEST_MODE_AUTO_ADVANCE else config.TEST_MODE_DISPLAY_TIME
    key = cv2.waitKey(wait_ms) & 0xFF
    
    return key


# VISUALISE GAUGE ANGLE CALCULATION AND MIN/MAX LIMITS (set SHOW_GAUGE_OVERLAY=True in config.py)
def draw_gauge_debug(frame: np.ndarray, detections: List[YoloDetection]) -> np.ndarray:
    """Draw gauge debugging information including angle lines and current pressure."""
    centres = sorted((d for d in detections if d.class_name == "Gauge_Centre"),
                     key=lambda d: d.confidence, reverse=True)
    tips = sorted((d for d in detections if d.class_name == "Needle_Tip"),
                  key=lambda d: d.confidence, reverse=True)
    
    if not centres or not tips:
        return frame
    
    # Copy frame to avoid modifying the original
    debug_frame = frame.copy()
    h, w = debug_frame.shape[:2]
    
    # Calculate positions and angles
    centre = centres[0]
    tip = tips[0]
    cx = int((centre.box[0] + centre.box[2]) / 2.0 * w)
    cy = int((centre.box[1] + centre.box[3]) / 2.0 * h)
    tx = int((tip.box[0] + tip.box[2]) / 2.0 * w)
    ty = int((tip.box[1] + tip.box[3]) / 2.0 * h)
    
    angle_deg = (math.degrees(math.atan2(cy - ty, tx - cx)) + 360.0) % 360.0
    
    # Draw needle line
    cv2.line(debug_frame, (cx, cy), (tx, ty), (0, 255, 255), 3)
    
    # Draw min angle line (0 bar)
    min_rad = math.radians(config.GAUGE_MIN_ANGLE_DEG)
    min_x = int(cx + 300 * math.cos(min_rad))
    min_y = int(cy - 300 * math.sin(min_rad))  # Note: Y is flipped
    cv2.line(debug_frame, (cx, cy), (min_x, min_y), (0, 255, 0), 2)
    cv2.putText(debug_frame, "0 bar", (min_x - 60, min_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, (0, 255, 0), 2)
    
    # Draw max angle line (10 bar)
    max_rad = math.radians(config.GAUGE_MAX_ANGLE_DEG)
    max_x = int(cx + 300 * math.cos(max_rad))
    max_y = int(cy - 300 * math.sin(max_rad))  # Note: Y is flipped
    cv2.line(debug_frame, (cx, cy), (max_x, max_y), (0, 0, 255), 2)
    cv2.putText(debug_frame, "10 bar", (max_x, max_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, (0, 0, 255), 2)
    
    # Show current needle angle in top right corner
    text = f"Angle: {angle_deg:.1f} deg"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, config.DETECTION_TEXT_THICKNESS)[0]
    text_x = w - text_size[0] - 10  # 10 pixels from right edge
    cv2.putText(debug_frame, text, (text_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                config.DETECTION_TEXT_SIZE, (0, 0, 0), config.DETECTION_TEXT_THICKNESS)
    
    return debug_frame