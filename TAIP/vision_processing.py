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
import threading
import time
from typing import List, Optional, Tuple

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

class ArucoWorker:
    """
    Non-blocking ArUco detection worker.
    Processes frames in a background thread and exposes the latest results.
    """

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 marker_size_m: float = None, max_hz: float = 30.0):
        self.K = camera_matrix.astype(np.float32)
        self.D = dist_coeffs.astype(np.float32)
        self.marker_size = float(marker_size_m or config.ARUCO_MARKER_SIZE_M)
        self.max_dt = 0.0 if max_hz <= 0 else 1.0 / max_hz

        self._lock = threading.Lock()
        self._running = False
        self._latest_frame = None  # Gray or BGR
        self._result = ([], None, None, None, None, None)  # dets, corners, ids, rvecs, tvecs, vis_img

        # Precompute object points for manual fallback
        global _ARUCO_OBJ_PTS
        if _ARUCO_OBJ_PTS is None:
            s = self.marker_size
            _ARUCO_OBJ_PTS = np.array(
                [[-s/2,  s/2, 0],
                 [ s/2,  s/2, 0],
                 [ s/2, -s/2, 0],
                 [-s/2, -s/2, 0]], dtype=np.float32
            )

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def update_frame(self, frame: np.ndarray):
        if frame is None:
            return
        with self._lock:
            self._latest_frame = frame.copy()

    def get_latest(self):
        with self._lock:
            return self._result

    def _loop(self):
        last_time = 0.0
        while self._running:
            now = time.time()
            if self.max_dt and (now - last_time) < self.max_dt:
                time.sleep(0.001)
                continue
            last_time = now

            frame = None
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()

            if frame is None:
                time.sleep(0.001)
                continue

            try:
                dets, corners, ids, rvecs, tvecs, vis = self._process(frame)
                with self._lock:
                    self._result = (dets, corners, ids, rvecs, tvecs, vis)
            except Exception:
                # Never block main loop due to exceptions
                pass

            time.sleep(0.0005)

    def _process(self, frame: np.ndarray):
        # Convert to gray if needed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # Detect markers
        try:
            corners, ids, _ = _aruco_detector.detectMarkers(gray)
        except Exception:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, config.ARUCO_DICT, parameters=_aruco_params)

        detections: List[ArucoDetection] = []
        rvecs = None
        tvecs = None
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if ids is None or len(ids) == 0:
            return detections, None, None, None, None, vis

        # Robust pose estimation using IPPE square with reprojection selection
        r_list, t_list = [], []
        for i in range(len(ids)):
            pts_img = corners[i].reshape(-1, 2).astype(np.float32)
            try:
                retval, rvecs_cand, tvecs_cand, reprojErrs = cv2.solvePnPGeneric(
                    _ARUCO_OBJ_PTS, pts_img, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not retval or len(rvecs_cand) == 0:
                    # Fallback to built-in if IPPE fails
                    rvecs_f, tvecs_f, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [pts_img], self.marker_size, self.K, self.D
                    )
                    rvec = rvecs_f[0][0].reshape(3, 1)
                    tvec = tvecs_f[0][0].reshape(3, 1)
                else:
                    # Choose best candidate: prefer positive Z and smallest reprojection error
                    best_idx = 0
                    best_score = float("inf")
                    for j, (rv, tv) in enumerate(zip(rvecs_cand, tvecs_cand)):
                        proj, _ = cv2.projectPoints(_ARUCO_OBJ_PTS, rv, tv, self.K, self.D)
                        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - pts_img, axis=1)))
                        z = float(tv[2])
                        # Penalise negative Z to avoid axis flipping
                        score = err + (0.5 if z > 0 else 1000.0)
                        if score < best_score:
                            best_score = score
                            best_idx = j
                    rvec = rvecs_cand[best_idx]
                    tvec = tvecs_cand[best_idx]
                r_list.append(rvec.reshape(1, 1, 3))
                t_list.append(tvec.reshape(1, 1, 3))
            except Exception:
                continue

        if r_list:
            rvecs = np.vstack(r_list)
            tvecs = np.vstack(t_list)

        # Draw markers and axes on the worker's own image (mono or input)
        try:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            if rvecs is not None and tvecs is not None:
                for rvec, tvec in zip(rvecs, tvecs):
                    cv2.drawFrameAxes(vis, self.K, self.D, rvec, tvec, 0.1, 3)
        except Exception:
            pass

        # Build typed detections with correct field names
        if rvecs is not None and tvecs is not None:
            for i, marker_id in enumerate(ids.flatten()):
                tvec = tvecs[i][0]
                rvec = rvecs[i][0]
                
                # Calculate distance
                distance_m = float(np.linalg.norm(tvec))
                
                # Get corner points
                corner_pts = corners[i].reshape(-1, 2)
                corner_list = [(float(pt[0]), float(pt[1])) for pt in corner_pts]
                
                detections.append(ArucoDetection(
                    marker_id=int(marker_id),
                    tvec=[float(tvec[0]), float(tvec[1]), float(tvec[2])],
                    rvec=[float(rvec[0]), float(rvec[1]), float(rvec[2])],
                    distance_m=distance_m,
                    corners=corner_list
                ))

        return detections, corners, ids, rvecs, tvecs, vis

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
    Legacy synchronous detector (kept for standalone/testing).
    Prefer using ArucoWorker in main loop to avoid blocking.
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
                             aruco_markers: List[ArucoDetection] = None) -> np.ndarray:
    """
    Draw YOLO detections and ArUco markers on a frame (for LCD display).
    Returns a new frame with drawings, does not modify original.
    
    Args:
        frame: Input BGR frame
        detections: List of YOLO detections
        aruco_markers: Optional list of ArUco markers
    
    Returns:
        Frame with detections drawn
    """
    output = frame.copy()
    h, w = output.shape[:2]
    
    # Draw YOLO bounding boxes
    for det in detections:
        x1 = int(det.x_min * w)
        y1 = int(det.y_min * h)
        x2 = int(det.x_max * w)
        y2 = int(det.y_max * h)
        
        # Color based on class_name
        color = config.DETECTION_COLOURS.get(det.class_name, config.DETECTION_COLOURS["default"])
        
        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw label text
        cv2.putText(output, f"{det.class_name}:{det.confidence:.2f}",
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 
                    config.DETECTION_TEXT_SIZE, color, config.DETECTION_TEXT_THICKNESS, cv2.LINE_AA)
    
    # Display ArUco marker ID, and 3D position (x, y, z) in meters (3D translation vector)
    if aruco_markers:
        for idx, marker in enumerate(aruco_markers):
            # Extract position from tvec
            tvec = marker.tvec
            text = f"ID {marker.marker_id}: ({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}) m"
            cv2.putText(output, text, (10, 120 + idx * 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        config.DETECTION_TEXT_SIZE, (0, 255, 255), 
                        config.DETECTION_TEXT_THICKNESS, cv2.LINE_AA)
    
    return output


def show_inference_visualisation(frame, detections, aruco_markers, aruco_corners, aruco_ids, gauge_pressure,
                                 camera_matrix=None, dist_coeffs=None, aruco_inset_bgr: Optional[np.ndarray] = None,
                                 is_video_mode: bool = True):
    """
    Shows a visualisation window. Draws YOLO on RGB. If an ArUco view is provided, render side-by-side.
    
    Args:
        is_video_mode: If True (video/live camera), auto-advances. If False (image directory), waits for keypress.
    """
    display_left = draw_detections_on_frame(frame, detections, aruco_markers)
    
    if config.SHOW_GAUGE_OVERLAY:
        display_left = draw_gauge_debug(display_left, detections)
    
    lh, lw = display_left.shape[:2]
    
    if gauge_pressure is not None:
        # Position gauge reading in top right corner of the left image
        text = f"Pressure: {gauge_pressure:.2f} bar"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, config.DETECTION_TEXT_THICKNESS)[0]
        text_x = lw - text_size[0] - 10
        cv2.putText(display_left, text, (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, 
                    (255, 255, 255), config.DETECTION_TEXT_THICKNESS)
    
    # Keep detection count in top left corner (left image)
    cv2.putText(display_left, f"Detections: {len(detections)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, config.DETECTION_TEXT_SIZE, 
                (255, 255, 255), config.DETECTION_TEXT_THICKNESS)

    # If we have the ArUco view, put it side-by-side with the RGB/YOLO view
    if aruco_inset_bgr is not None:
        right_img = aruco_inset_bgr
        if right_img.ndim == 2:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

        # Match heights while preserving aspect ratio
        target_h = max(display_left.shape[0], right_img.shape[0])

        def resize_to_h(img, h):
            ih, iw = img.shape[:2]
            scale = float(h) / float(ih)
            return cv2.resize(img, (int(iw * scale), h))

        left_resized = resize_to_h(display_left, target_h)
        right_resized = resize_to_h(right_img, target_h)

        # Add labels for each camera view (top-left corner)
        cv2.putText(left_resized, "RGB: YOLO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    config.DETECTION_TEXT_SIZE * 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(right_resized, "LEFT: ArUco", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    config.DETECTION_TEXT_SIZE * 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        # Compose canvas with separator
        sep = 8
        canvas_w = left_resized.shape[1] + sep + right_resized.shape[1]
        canvas = np.zeros((target_h, canvas_w, 3), dtype=np.uint8)

        canvas[:, :left_resized.shape[1]] = left_resized
        canvas[:, left_resized.shape[1] + sep:] = right_resized

        final_img = canvas
    else:
        # Fallback to single left image if ArUco view not available
        final_img = display_left
    
    # Create named window and set to fullscreen ONLY on first call
    window_name = config.TEST_MODE_WINDOW_NAME
    if not hasattr(show_inference_visualisation, '_window_created'):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        show_inference_visualisation._window_created = True
    
    # Display the image
    cv2.imshow(window_name, final_img)

    # Determine wait time based on mode
    if is_video_mode:
        # Video or live camera: brief wait to allow window updates
        # wait_ms = config.TEST_MODE_DISPLAY_TIME
        wait_ms = 1
    else:
        # Image directory: wait indefinitely for keypress
        wait_ms = 0
    
    key = cv2.waitKey(wait_ms) & 0xFF
    
    # Allow 'f' key to toggle fullscreen on/off
    if key == ord('f'):
        current_state = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        if current_state == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
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