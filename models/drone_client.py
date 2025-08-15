import os, time, math, threading
import requests
import cv2
import numpy as np
import depthai as dai

LAPTOP_URL = os.environ.get("LAPTOP_URL", "http://10.88.30.16:5000")
POST_FRAME_FPS = 10
POST_TELEM_HZ = 5

BLOB_PATH = "/home/pi/EGH455/models/YOLOv11n.blob"
MODEL_IMG_SIZE = 640
MODEL_CONF = 0.35
MODEL_IOU = 0.45
NUM_CLASSES = 4
GAUGE_MIN_ANGLE = 225.0
GAUGE_MAX_ANGLE = -45.0
GAUGE_MIN_P = 0.0
GAUGE_MAX_P = 100.0

CLASS_NAMES = {
    0: "gauge_centre",
    1: "needle_tip", 
    2: "valve_closed",
    3: "valve_open"
}

try:
    import servo_motor
except Exception:
    servo_motor = None

def draw_detections(frame, detections):
    """Draws bounding boxes and labels on the frame."""
    for det in detections:
        x1 = int(det.xmin * frame.shape[1])
        y1 = int(det.ymin * frame.shape[0])
        x2 = int(det.xmax * frame.shape[1])
        y2 = int(det.ymax * frame.shape[0])
        
        label = f"{CLASS_NAMES.get(det.label, 'Unknown')} {det.confidence:.2f}"
        
        color = (0, 255, 0)
        if "valve" in CLASS_NAMES.get(det.label, ''):
            color = (255, 0, 0)
        elif "gauge" in CLASS_NAMES.get(det.label, '') or "needle" in CLASS_NAMES.get(det.label, ''):
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

class OakClient:
    def __init__(self):
        # Create two separate pipelines to avoid conflicts
        
        # Pipeline 1: Camera only for video
        self.camera_pipeline = dai.Pipeline()
        self.cam_rgb = self.camera_pipeline.create(dai.node.ColorCamera)
        self.cam_rgb.setPreviewSize(MODEL_IMG_SIZE, MODEL_IMG_SIZE)
        self.cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.cam_rgb.setFps(30)

        self.xout_rgb = self.camera_pipeline.create(dai.node.XLinkOut)
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)

        # Connect to device for camera
        try:
            self.camera_device = dai.Device(self.camera_pipeline)
            print(f"Camera connected to device: {self.camera_device.getDeviceInfo().getMxId()}")
            self.q_rgb = self.camera_device.getOutputQueue("rgb", maxSize=4, blocking=False)
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            raise

        # Pipeline 2: Neural Network for object detection
        self.nn_pipeline = dai.Pipeline()
        
        # Create camera for NN
        self.cam_nn = self.nn_pipeline.create(dai.node.ColorCamera)
        self.cam_nn.setPreviewSize(MODEL_IMG_SIZE, MODEL_IMG_SIZE)
        self.cam_nn.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam_nn.setInterleaved(False)
        self.cam_nn.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.cam_nn.setFps(30)
        
        # Create YOLO detection network
        self.detection_nn = self.nn_pipeline.create(dai.node.YoloDetectionNetwork)
        self.detection_nn.setConfidenceThreshold(MODEL_CONF)
        self.detection_nn.setNumClasses(NUM_CLASSES)
        self.detection_nn.setCoordinateSize(4)
        self.detection_nn.setIouThreshold(MODEL_IOU)
        self.detection_nn.setBlobPath(BLOB_PATH)
        
        # YOLO anchor configuration
        anchors = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
        anchorMasks = {"side8400": [0,1,2,3,4,5,6,7,8]}
        self.detection_nn.setAnchors(anchors)
        self.detection_nn.setAnchorMasks(anchorMasks)
        
        # Connect camera to NN
        self.cam_nn.preview.link(self.detection_nn.input)
        
        # NN output
        self.xout_nn = self.nn_pipeline.create(dai.node.XLinkOut)
        self.xout_nn.setStreamName("detections")
        self.detection_nn.out.link(self.xout_nn.input)
        
        # Connect to device for NN (using a separate device instance)
        try:
            self.nn_device = dai.Device(self.nn_pipeline)
            print(f"NN connected to device: {self.nn_device.getDeviceInfo().getMxId()}")
            self.q_nn = self.nn_device.getOutputQueue("detections", maxSize=4, blocking=False)
        except Exception as e:
            print(f"Failed to initialize NN pipeline: {e}")
            # Continue without NN if it fails
            self.nn_device = None
            self.q_nn = None
        
        # Thread-safe state management
        self.lock = threading.Lock()
        self.raw_frame = None
        self.annotated_frame = None
        self.detections = []  # Empty for now
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        frame_count = 0
        last_debug = time.time()
        
        while self.running:
            try:
                new_frame = None
                new_detections = None
                
                # Get RGB frame from camera
                in_rgb = self.q_rgb.tryGet()
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()
                    if frame is not None:
                        frame_count += 1
                        new_frame = frame.copy()
                        
                        # Debug output every 2 seconds
                        current_time = time.time()
                        if current_time - last_debug >= 2.0:
                            print(f"Frames processed: {frame_count}, mean: {frame.mean():.2f}")
                            last_debug = current_time

                # Get detections from NN pipeline
                if self.q_nn is not None:
                    in_nn = self.q_nn.tryGet()
                    if in_nn is not None:
                        new_detections = in_nn.detections
                        if len(new_detections) > 0:
                            print(f"Got {len(new_detections)} detections")

                # Update state and create annotated frame
                with self.lock:
                    if new_frame is not None:
                        self.raw_frame = new_frame
                    if new_detections is not None:
                        self.detections = new_detections
                    
                    # Create annotated frame with detections
                    if self.raw_frame is not None:
                        display_frame = self.raw_frame.copy()
                        if len(self.detections) > 0:
                            display_frame = draw_detections(display_frame, self.detections)
                        self.annotated_frame = display_frame

            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)
                continue
            
            time.sleep(0.001)

    def get_frame(self):
        """Returns the latest annotated frame."""
        with self.lock:
            return self.annotated_frame
    
    def get_detections(self):
        """Returns a copy of the latest detections."""
        with self.lock:
            return list(self.detections)

    def close(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            if hasattr(self, 'camera_device'):
                self.camera_device.close()
            if hasattr(self, 'nn_device') and self.nn_device is not None:
                self.nn_device.close()
        except Exception as e:
            print(f"Error closing device: {e}")

class Vision:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def aruco_ids(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return [] if ids is None else [int(i[0]) for i in ids]

def compute_pressure(angle_deg):
    a0 = (GAUGE_MIN_ANGLE + 360) % 360
    a1 = (GAUGE_MAX_ANGLE + 360) % 360
    def ad(a,b): return (a-b+540) % 360 - 180
    span = ad(a1, a0)
    pos = ad(angle_deg, a0)
    t = 0.0 if span == 0 else max(0.0, min(1.0, pos / span))
    return GAUGE_MIN_P + t * (GAUGE_MAX_P - GAUGE_MIN_P)

def main():
    print(f"Starting drone client (camera only for now)...")
    print(f"Model image size: {MODEL_IMG_SIZE}")
    
    try:
        oak = OakClient()
    except Exception as e:
        print(f"Failed to initialize OAK client: {e}")
        return
        
    vis = Vision()

    sess = requests.Session()
    last_frame_post = 0.0
    last_telem_post = 0.0
    target_pressure = 50.0

    try:
        while True:
            f = oak.get_frame()
            if f is None:
                time.sleep(0.01)
                continue

            now = time.time()
            ids = vis.aruco_ids(f)
            valve_state = None
            pressure = None
            
            # Get YOLO detections for object detection
            dets = oak.get_detections()

            # Extract valve open/closed (pick highest confidence if both)
            open_candidates = [d for d in dets if CLASS_NAMES.get(d.label) == "valve_open"]
            closed_candidates = [d for d in dets if CLASS_NAMES.get(d.label) == "valve_closed"]
            if open_candidates or closed_candidates:
                best_open = max(open_candidates, key=lambda d: d.confidence) if open_candidates else None
                best_closed = max(closed_candidates, key=lambda d: d.confidence) if closed_candidates else None
                if best_open and best_closed:
                    valve_state = "open" if best_open.confidence >= best_closed.confidence else "closed"
                elif best_open:
                    valve_state = "open"
                elif best_closed:
                    valve_state = "closed"

            # Needle tip / gauge centre for pressure reading
            needle_tip = next((d for d in dets if CLASS_NAMES.get(d.label) == "needle_tip"), None)
            gauge_centre = next((d for d in dets if CLASS_NAMES.get(d.label) == "gauge_centre"), None)
            if needle_tip and gauge_centre:
                # Use centers of bounding boxes (coordinates are 0.0-1.0)
                tx = (needle_tip.xmin + needle_tip.xmax) / 2.0
                ty = (needle_tip.ymin + needle_tip.ymax) / 2.0
                cx = (gauge_centre.xmin + gauge_centre.xmax) / 2.0
                cy = (gauge_centre.ymin + gauge_centre.ymax) / 2.0
                # Calculate angle between needle and center
                angle = (math.degrees(math.atan2(cy - ty, tx - cx)) + 360) % 360
                pressure = compute_pressure(angle)

            # Telemetry push
            if now - last_telem_post >= 1.0 / POST_TELEM_HZ:
                telem = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "aruco_ids": ids,
                    "ball_gauge": valve_state,  # Valve state from YOLO
                    "gauge_pressure": pressure,  # Pressure from gauge reading
                    "target_pressure": target_pressure,
                    "ready_to_drill": bool(pressure is not None and pressure >= target_pressure),
                    "env": {}
                }
                try:
                    sess.post(f"{LAPTOP_URL}/telemetry", json=telem, timeout=2)
                except Exception:
                    pass
                last_telem_post = now

            # Frame push
            if now - last_frame_post >= 1.0 / POST_FRAME_FPS:
                try:
                    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        sess.post(f"{LAPTOP_URL}/frame",
                                  data=buf.tobytes(),
                                  timeout=2,
                                  headers={"Content-Type": "application/octet-stream"})
                except Exception:
                    pass
                last_frame_post = now

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        oak.close()

if __name__ == "__main__":
    main()
