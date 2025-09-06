"""
Vision processing module for the TAIP subsystem.
Contains functions for gauge reading calculation and ArUco marker detection.
"""

import cv2
import numpy as np
import math
from typing import List, Optional, Tuple, Dict, Any
from data_models import YoloDetection, ArucoDetection
import config


def calculate_gauge_reading(detections: List[YoloDetection]) -> Optional[float]:
    """
    Calculate pressure gauge reading from YOLO detections.
    
    Finds Needle_Tip and Gauge_Centre detections, calculates the angle,
    and maps it to a pressure value using the calibration parameters.
    
    Args:
        detections: List of YOLO detections
        
    Returns:
        Pressure value in bar, or None if calculation fails
    """
    needle_tip = None
    gauge_centre = None
    
    # Find the required detections
    for detection in detections:
        if detection.class_name == 'Needle_Tip':
            needle_tip = detection
        elif detection.class_name == 'Gauge_Centre':
            gauge_centre = detection
    
    # Check if both detections are found
    if needle_tip is None or gauge_centre is None:
        return None
    
    try:
        # Get center points of the bounding boxes
        needle_center = _get_bbox_center(needle_tip.bounding_box)
        gauge_center = _get_bbox_center(gauge_centre.bounding_box)
        
        # Calculate angle from gauge center to needle tip
        angle_rad = math.atan2(
            needle_center[1] - gauge_center[1],  # dy
            needle_center[0] - gauge_center[0]   # dx
        )
        
        # Convert to degrees and normalize to 0-360 range
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        # Map angle to pressure using linear interpolation
        pressure = _angle_to_pressure(angle_deg)
        
        # Validate pressure range
        if pressure < 0:
            pressure = 0.0
        elif pressure > config.GAUGE_MAX_PRESSURE:
            pressure = config.GAUGE_MAX_PRESSURE
        
        return pressure
        
    except (ValueError, ZeroDivisionError, AttributeError) as e:
        print(f"Warning: Gauge reading calculation failed: {e}")
        return None


def _get_bbox_center(bbox) -> Tuple[float, float]:
    """Get the center point of a bounding box."""
    center_x = (bbox.x_min + bbox.x_max) / 2
    center_y = (bbox.y_min + bbox.y_max) / 2
    return (center_x, center_y)


def _angle_to_pressure(angle_deg: float) -> float:
    """
    Convert angle to pressure using linear mapping.
    
    Gauge calibration:
    - -45° to 225° total range (270°)
    - 10 bar at -45° (or 315°), decreasing to 0 bar at 225°
    
    Args:
        angle_deg: Angle in degrees (0-360)
        
    Returns:
        Pressure in bar
    """
    # Normalize angle to the gauge's reference frame
    # Convert 0-360° to gauge angle range
    if angle_deg > 270:  # Handle wrap-around for negative angles
        normalized_angle = angle_deg - 360
    else:
        normalized_angle = angle_deg
    
    # Clamp to gauge range
    if normalized_angle < config.GAUGE_MIN_ANGLE:
        normalized_angle = config.GAUGE_MIN_ANGLE
    elif normalized_angle > config.GAUGE_MAX_ANGLE:
        normalized_angle = config.GAUGE_MAX_ANGLE
    
    # Linear interpolation: pressure decreases as angle increases
    angle_range = config.GAUGE_MAX_ANGLE - config.GAUGE_MIN_ANGLE
    pressure_range = config.GAUGE_MAX_PRESSURE - config.GAUGE_MIN_PRESSURE
    
    # Calculate pressure (inverse relationship with angle)
    angle_ratio = (normalized_angle - config.GAUGE_MIN_ANGLE) / angle_range
    pressure = config.GAUGE_MAX_PRESSURE - (angle_ratio * pressure_range)
    
    return pressure


def detect_aruco_markers(frame: np.ndarray) -> List[ArucoDetection]:
    """
    Detect ArUco markers in the input frame and estimate their poses.
    
    Args:
        frame: Input image as numpy array (BGR format)
        
    Returns:
        List of ArucoDetection objects
    """
    if frame is None or frame.size == 0:
        return []
    
    try:
        # Convert to grayscale for ArUco detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, config.ARUCO_DICT)
        )
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return []
        
        # Camera calibration parameters
        camera_matrix = np.array(config.CAMERA_MATRIX, dtype=np.float32)
        dist_coeffs = np.array(config.DISTORTION_COEFFICIENTS, dtype=np.float32)
        
        # Estimate pose for each detected marker
        aruco_detections = []
        for i, marker_id in enumerate(ids.flatten()):
            try:
                # Estimate pose
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i:i+1], 
                    config.ARUCO_MARKER_SIZE,
                    camera_matrix,
                    dist_coeffs
                )
                
                # Extract pose vectors
                rvec = rvecs[0][0]  # Rotation vector
                tvec = tvecs[0][0]  # Translation vector
                
                # Convert corner points to normalized coordinates
                corner_points = _normalize_corners(corners[i][0], frame.shape)
                
                # Create ArucoDetection object
                detection = ArucoDetection(
                    marker_id=int(marker_id),
                    position=(float(tvec[0]), float(tvec[1]), float(tvec[2])),
                    orientation=(float(rvec[0]), float(rvec[1]), float(rvec[2])),
                    corners=corner_points
                )
                
                aruco_detections.append(detection)
                
            except (cv2.error, ValueError, IndexError) as e:
                print(f"Warning: Failed to process ArUco marker {marker_id}: {e}")
                continue
        
        return aruco_detections
        
    except Exception as e:
        print(f"Error in ArUco detection: {e}")
        return []


def _normalize_corners(corners: np.ndarray, frame_shape: Tuple[int, ...]) -> List[Tuple[float, float]]:
    """
    Normalize corner coordinates to 0-1 range.
    
    Args:
        corners: Array of corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        frame_shape: Shape of the frame (height, width, channels)
        
    Returns:
        List of normalized corner coordinates
    """
    height, width = frame_shape[:2]
    normalized_corners = []
    
    for corner in corners:
        x_norm = corner[0] / width
        y_norm = corner[1] / height
        # Clamp to valid range
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        normalized_corners.append((x_norm, y_norm))
    
    return normalized_corners


def filter_detections_by_confidence(detections: List[YoloDetection], 
                                   min_confidence: float = None) -> List[YoloDetection]:
    """
    Filter YOLO detections by confidence threshold.
    
    Args:
        detections: List of YOLO detections
        min_confidence: Minimum confidence threshold (uses config if None)
        
    Returns:
        Filtered list of detections
    """
    if min_confidence is None:
        min_confidence = config.CONFIDENCE_THRESHOLD
    
    return [det for det in detections if det.confidence >= min_confidence]


def find_best_detection_by_class(detections: List[YoloDetection], 
                                class_name: str) -> Optional[YoloDetection]:
    """
    Find the detection with highest confidence for a specific class.
    
    Args:
        detections: List of YOLO detections
        class_name: Target class name
        
    Returns:
        Best detection for the class, or None if not found
    """
    class_detections = [det for det in detections if det.class_name == class_name]
    
    if not class_detections:
        return None
    
    return max(class_detections, key=lambda x: x.confidence)


def draw_detections_on_frame(frame: np.ndarray, 
                           detections: List[YoloDetection],
                           aruco_markers: List[ArucoDetection] = None) -> np.ndarray:
    """
    Draw bounding boxes and labels on the frame for visualization.
    
    Args:
        frame: Input frame
        detections: YOLO detections to draw
        aruco_markers: ArUco markers to draw (optional)
        
    Returns:
        Frame with drawn annotations
    """
    if frame is None or frame.size == 0:
        return frame
    
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Draw YOLO detections
    for detection in detections:
        # Convert normalized coordinates to absolute
        x1, y1, x2, y2 = detection.bounding_box.to_absolute(width, height)
        
        # Draw bounding box
        cv2.rectangle(
            annotated_frame, 
            (x1, y1), (x2, y2), 
            config.BBOX_COLOR, 
            config.BBOX_THICKNESS
        )
        
        # Draw label with confidence
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, 1
        )[0]
        
        # Background for text
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            config.BBOX_COLOR,
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.TEXT_SCALE,
            config.TEXT_COLOR,
            1
        )
    
    # Draw ArUco markers if provided
    if aruco_markers and config.DRAW_ARUCO_MARKERS:
        for marker in aruco_markers:
            if marker.corners:
                # Convert normalized corners to absolute coordinates
                corners_abs = []
                for corner in marker.corners:
                    x = int(corner[0] * width)
                    y = int(corner[1] * height)
                    corners_abs.append((x, y))
                
                # Draw marker outline
                for i in range(4):
                    start_point = corners_abs[i]
                    end_point = corners_abs[(i + 1) % 4]
                    cv2.line(annotated_frame, start_point, end_point, (0, 255, 255), 2)
                
                # Draw marker ID
                center_x = int(sum(corner[0] for corner in marker.corners) / 4 * width)
                center_y = int(sum(corner[1] for corner in marker.corners) / 4 * height)
                
                cv2.putText(
                    annotated_frame,
                    f"ID:{marker.marker_id}",
                    (center_x - 20, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
    
    return annotated_frame


def calculate_detection_statistics(detections: List[YoloDetection]) -> Dict[str, Any]:
    """
    Calculate statistics about the detections for monitoring.
    
    Args:
        detections: List of YOLO detections
        
    Returns:
        Dictionary with detection statistics
    """
    if not detections:
        return {
            "total_detections": 0,
            "class_counts": {},
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0
        }
    
    class_counts = {}
    confidences = []
    
    for detection in detections:
        # Count detections by class
        class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1
        confidences.append(detection.confidence)
    
    return {
        "total_detections": len(detections),
        "class_counts": class_counts,
        "avg_confidence": sum(confidences) / len(confidences),
        "max_confidence": max(confidences),
        "min_confidence": min(confidences)
    }


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_gauge_calibration() -> bool:
    """Validate gauge calibration by testing edge cases."""
    try:
        # Test minimum angle
        pressure_min = _angle_to_pressure(config.GAUGE_MIN_ANGLE)
        expected_max = config.GAUGE_MAX_PRESSURE
        
        # Test maximum angle  
        pressure_max = _angle_to_pressure(config.GAUGE_MAX_ANGLE)
        expected_min = config.GAUGE_MIN_PRESSURE
        
        print(f"Gauge calibration test:")
        print(f"  Angle {config.GAUGE_MIN_ANGLE}° -> {pressure_min:.2f} bar (expected: {expected_max:.2f})")
        print(f"  Angle {config.GAUGE_MAX_ANGLE}° -> {pressure_max:.2f} bar (expected: {expected_min:.2f})")
        
        # Allow small tolerance for floating point errors
        tolerance = 0.1
        min_valid = abs(pressure_min - expected_max) < tolerance
        max_valid = abs(pressure_max - expected_min) < tolerance
        
        if min_valid and max_valid:
            print("✓ Gauge calibration validation passed")
            return True
        else:
            print("✗ Gauge calibration validation failed")
            return False
            
    except Exception as e:
        print(f"✗ Gauge calibration validation error: {e}")
        return False


if __name__ == "__main__":
    # Run validation tests
    print("Testing vision processing module...")
    
    # Test gauge calibration
    validate_gauge_calibration()
    
    # Test with dummy data
    from data_models import BoundingBox, YoloDetection
    
    # Create test detections
    needle_bbox = BoundingBox(0.6, 0.4, 0.65, 0.45)  # Needle tip
    gauge_bbox = BoundingBox(0.5, 0.5, 0.55, 0.55)   # Gauge center
    
    test_detections = [
        YoloDetection("Needle_Tip", 0.9, needle_bbox),
        YoloDetection("Gauge_Centre", 0.85, gauge_bbox)
    ]
    
    # Test gauge reading calculation
    pressure = calculate_gauge_reading(test_detections)
    print(f"Test gauge reading: {pressure:.2f} bar" if pressure else "Failed to calculate gauge reading")
    
    # Test detection statistics
    stats = calculate_detection_statistics(test_detections)
    print(f"Detection statistics: {stats}")
    
    print("Vision processing module tests completed.")
