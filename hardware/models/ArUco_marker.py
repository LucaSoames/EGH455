#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os

# Set OpenCV backend before importing
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Create pipeline
pipeline = dai.Pipeline()

# Use mono camera (global shutter) - better for moving applications
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)

# Output
monoOut = pipeline.create(dai.node.XLinkOut)
monoOut.setStreamName("mono")
monoLeft.out.link(monoOut.input)

# ArUco detector setup with maximum precision parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters()

# Maximum precision parameters
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 2    # Smaller steps for more precision
aruco_params.minMarkerPerimeterRate = 0.02   # Slightly more lenient for detection
aruco_params.maxMarkerPerimeterRate = 4.0    
aruco_params.polygonalApproxAccuracyRate = 0.01  # Very strict polygon approximation
aruco_params.minCornerDistanceRate = 0.03    # Strict corner distance
aruco_params.minDistanceToBorder = 2         # Require distance from border
aruco_params.minMarkerDistanceRate = 0.03    # Minimum distance between markers

# Maximum precision corner refinement
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 7     # Larger window for better precision
aruco_params.cornerRefinementMaxIterations = 50  # More iterations
aruco_params.cornerRefinementMinAccuracy = 0.01  # Very high accuracy requirement

# Additional validation parameters for precision
aruco_params.errorCorrectionRate = 0.8       # Higher error correction
aruco_params.minOtsuStdDev = 5.0             # Minimum standard deviation for Otsu thresholding

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

frame_count = 0
detection_history = {}  # Track detection consistency
confidence_threshold = 3  # Require 3 consecutive detections

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print('Connected cameras:', device.getConnectedCameraFeatures())
    print('Looking for ArUco markers with maximum precision...')
    
    qMono = device.getOutputQueue(name="mono", maxSize=4, blocking=False)

    try:
        while True:
            inMono = qMono.get()
            frame = inMono.getCvFrame()
            frame_count += 1
            
            # Enhanced preprocessing for better detection
            # Apply histogram equalization
            frame_eq = cv2.equalizeHist(frame)
            
            # Apply bilateral filter to reduce noise while preserving edges
            frame_filtered = cv2.bilateralFilter(frame_eq, 9, 75, 75)
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((2,2), np.uint8)
            frame_morph = cv2.morphologyEx(frame_filtered, cv2.MORPH_CLOSE, kernel)
            
            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(frame_morph)
            
            # Convert to color for visualization
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Enhanced validation for detected markers
            if ids is not None:
                valid_markers = []
                valid_corners = []
                valid_ids = []
                
                for i, marker_id in enumerate(ids):
                    marker_corners = corners[i][0]
                    marker_id_val = int(marker_id[0])
                    
                    # Enhanced geometric validation
                    marker_area = cv2.contourArea(marker_corners)
                    
                    # Calculate side lengths and angles
                    side1 = np.linalg.norm(marker_corners[0] - marker_corners[1])
                    side2 = np.linalg.norm(marker_corners[1] - marker_corners[2])
                    side3 = np.linalg.norm(marker_corners[2] - marker_corners[3])
                    side4 = np.linalg.norm(marker_corners[3] - marker_corners[0])
                    
                    # Calculate angles
                    def angle_between_vectors(v1, v2):
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                    
                    v1 = marker_corners[1] - marker_corners[0]
                    v2 = marker_corners[3] - marker_corners[0]
                    angle1 = angle_between_vectors(v1, v2)
                    
                    v1 = marker_corners[0] - marker_corners[1]
                    v2 = marker_corners[2] - marker_corners[1]
                    angle2 = angle_between_vectors(v1, v2)
                    
                    # Check if sides are roughly equal (square-like)
                    sides = [side1, side2, side3, side4]
                    avg_side = np.mean(sides)
                    side_variance = np.var(sides) / (avg_side ** 2)
                    
                    # Check aspect ratio
                    rect = cv2.minAreaRect(marker_corners)
                    width, height = rect[1]
                    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float('inf')
                    
                    # Enhanced validation criteria
                    min_area = 150   # Minimum marker area
                    max_area = 40000 # Maximum marker area
                    max_side_variance = 0.08  # Very strict side variance
                    max_aspect_ratio = 1.3    # Nearly square
                    min_angle = 70    # Angles should be close to 90 degrees
                    max_angle = 110
                    
                    geometric_valid = (min_area < marker_area < max_area and 
                                     side_variance < max_side_variance and
                                     aspect_ratio < max_aspect_ratio and
                                     min_angle < angle1 < max_angle and
                                     min_angle < angle2 < max_angle)
                    
                    if geometric_valid:
                        # Track detection history for consistency
                        if marker_id_val not in detection_history:
                            detection_history[marker_id_val] = 0
                        detection_history[marker_id_val] += 1
                        
                        # Only accept markers detected consistently
                        if detection_history[marker_id_val] >= confidence_threshold:
                            valid_markers.append(i)
                            valid_corners.append(corners[i])
                            valid_ids.append(marker_id)
                    else:
                        # Reset detection history for failed validation
                        if marker_id_val in detection_history:
                            detection_history[marker_id_val] = max(0, detection_history[marker_id_val] - 1)
                
                # Decay detection history for markers not seen
                for marker_id_val in list(detection_history.keys()):
                    if ids is None or marker_id_val not in [int(id[0]) for id in ids]:
                        detection_history[marker_id_val] = max(0, detection_history[marker_id_val] - 1)
                        if detection_history[marker_id_val] == 0:
                            del detection_history[marker_id_val]
                
                # Draw only validated markers
                if valid_ids:
                    cv2.aruco.drawDetectedMarkers(frame_color, valid_corners, np.array(valid_ids))
                    
                    # Print valid marker IDs with confidence
                    marker_ids = [int(marker_id[0]) for marker_id in valid_ids]
                    confidences = [detection_history.get(mid, 0) for mid in marker_ids]
                    print(f"Detected ArUco marker(s) ID: {marker_ids} (confidence: {confidences})")
                    
                    for i, marker_id in enumerate(valid_ids):
                        marker_corners = valid_corners[i][0]
                        center_x = int(np.mean(marker_corners[:, 0]))
                        center_y = int(np.mean(marker_corners[:, 1]))
                        marker_size = cv2.contourArea(marker_corners)
                        confidence = detection_history.get(int(marker_id[0]), 0)
                        
                        # Color based on confidence
                        color = (0, 255, 0) if confidence >= confidence_threshold else (0, 255, 255)
                        
                        cv2.circle(frame_color, (center_x, center_y), 5, color, -1)
                        cv2.putText(frame_color, f"ID: {int(marker_id[0])} ({confidence})", 
                                   (center_x-30, center_y-25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Draw corner points for precision visualization
                        for corner in marker_corners:
                            cv2.circle(frame_color, (int(corner[0]), int(corner[1])), 2, (255, 0, 0), -1)

            # Add enhanced status overlay
            valid_count = len(valid_ids) if 'valid_ids' in locals() and valid_ids else 0
            total_tracked = len(detection_history)
            status_text = f"Valid: {valid_count}, Tracking: {total_tracked}"
            cv2.putText(frame_color, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("ArUco Detection - High Precision", frame_color)

            if cv2.waitKey(1) == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
        
    cv2.destroyAllWindows()