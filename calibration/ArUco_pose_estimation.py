'''
In order to perform an accurate pose estimation using the 
ArUCO library and markers, you need to know the intrinsic 
parameters of the camera you are using. These parameters 
include the radial and tangential distortion, focal length, 
and optical centers according to your camera lens.

However, the OAK-D lite camera comes calibrated from the 
factory and you just need to read the intrinsic parameters 
from its EEPROM memory. Execute the example from the Depth 
AI library (camera_calibration.py). 

Once, you have read the camera calibration parameters, 
you can use them to estimate the pose of an ArUCO marker 
using the ArUCO library functions.
'''
import cv2
import depthai as dai

# TAIP/config.py contains global variables. It is in a sibling folder in the parent directory EGH455.
import sys
from pathlib import Path
TAIP_PATH = (Path(__file__).resolve().parent.parent / 'TAIP').resolve()
# Check if path exists before adding
if TAIP_PATH.exists():
    # Add the parent directory of TAIP to sys.path, not TAIP itself
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in map(str, map(Path, sys.path)):
        sys.path.append(str(parent_dir))
else:
    raise FileNotFoundError(f"TAIP directory not found at {TAIP_PATH}")
    
from TAIP.config import (
    ARUCO_DICT,
    ARUCO_MARKER_SIZE_M,
    CAMERA_ARUCO_SOURCE,
    CAMERA_MATRIX,
    CAMERA_PREVIEW_SIZE,
    DISTORTION_COEFFS
)


def pose_estimation(frame, aruco_dict_type, matrix_coeffs, dist_coeffs):
    """
    Estimate and draw ArUco marker poses on frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # switch to new‐style detector
    params = cv2.aruco.DetectorParameters()

    # new vs old API detection
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict_type, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict_type, parameters=params)

    if ids is not None and len(corners) > 0:
        # draw the marker outlines + IDs
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # estimate each marker’s pose (module-level API)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            ARUCO_MARKER_SIZE_M,
            matrix_coeffs,
            dist_coeffs
        )

        # draw axes using the generic drawFrameAxes call
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(
                frame,
                matrix_coeffs,
                dist_coeffs,
                rvec,
                tvec,
                0.1,       # length in metres
                thickness=3
            )
    return frame

def create_oak_pipeline(source: str) -> dai.Pipeline:
    p = dai.Pipeline()
    if source.upper() == 'RGB':
        cam = p.createColorCamera()
        xout = p.createXLinkOut()

        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setPreviewSize(*CAMERA_PREVIEW_SIZE)
        cam.setInterleaved(True)  # now outputs 3-channel BGR
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(15)

        # link preview directly → guaranteed BGR888 interleaved frames
        cam.preview.link(xout.input)
        xout.setStreamName('frame')
    else:  # LEFT mono
        mono = p.createMonoCamera()
        xout = p.createXLinkOut()
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono.setFps(15)
        mono.out.link(xout.input)
        xout.setStreamName('frame')
    return p

if __name__ == '__main__':
    # Spin up OAK-D pipeline
    pipeline = create_oak_pipeline(CAMERA_ARUCO_SOURCE)
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name='frame', maxSize=4, blocking=False)
        while True:
            in_msg = q.get()             # depthai.ImgFrame
            data   = in_msg.getFrame()   # numpy array
            if CAMERA_ARUCO_SOURCE.upper() == 'LEFT':
                frame = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
            else:
                frame = data
            output = pose_estimation(frame, ARUCO_DICT, CAMERA_MATRIX, DISTORTION_COEFFS)
            cv2.imshow('ArUco Pose', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()