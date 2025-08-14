import os, time, math, threading
import requests
import cv2
import numpy as np
import depthai as dai

LAPTOP_URL = os.environ.get("LAPTOP_URL", "http://192.168.86.246:5000")
POST_FRAME_FPS = 10
POST_TELEM_HZ = 5

BLOB_PATH = "/home/pi/EGH455/models/YOLOv11n.blob"  # Updated to use .blob file
MODEL_IMG_SIZE = 640
MODEL_CONF = 0.35
MODEL_IOU = 0.45
NUM_CLASSES = 4  # Update this based on your model (gauge_centre, needle_tip, valve_closed, valve_open)
GAUGE_MIN_ANGLE = 225.0   # adjust after calibration
GAUGE_MAX_ANGLE = -45.0
GAUGE_MIN_P = 0.0
GAUGE_MAX_P = 100.0

# Class names mapping - updated based on your trained model
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
        # The YoloDetectionNetwork node provides coordinates normalized to the frame size (0.0 to 1.0)
        x1 = int(det.xmin * frame.shape[1])
        y1 = int(det.ymin * frame.shape[0])
        x2 = int(det.xmax * frame.shape[1])
        y2 = int(det.ymax * frame.shape[0])
        
        label = f"{CLASS_NAMES.get(det.label, 'Unknown')} {det.confidence:.2f}"
        
        color = (0, 255, 0) # Default green
        if "valve" in CLASS_NAMES.get(det.label, ''):
            color = (255, 0, 0) # Blue for valves
        elif "gauge" in CLASS_NAMES.get(det.label, '') or "needle" in CLASS_NAMES.get(det.label, ''):
            color = (0, 0, 255) # Red for gauge parts

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

class OakClient:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        
        # Create color camera node
        self.cam = self.pipeline.create(dai.node.ColorCamera)
        self.cam.setPreviewSize(MODEL_IMG_SIZE, MODEL_IMG_SIZE)
        self.cam.setFps(30)
        self.cam.setInterleaved(False)
        self.cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        # Create YoloDetectionNetwork node instead of NeuralNetwork
        self.nn = self.pipeline.create(dai.node.YoloDetectionNetwork)
        self.nn.setBlobPath(BLOB_PATH)
        self.nn.setConfidenceThreshold(MODEL_CONF)
        self.nn.setNumClasses(NUM_CLASSES)
        
        # Configure for YOLOv8-style output which is similar to v11
        # The model seems to have 1 anchor per cell, so we set mask to [0,1,2] for a single scale.
        self.nn.setCoordinateSize(4)
        self.nn.setAnchors([]) # YOLOv8/v11 are anchor-free
        self.nn.setAnchorMasks({})
        self.nn.setIouThreshold(MODEL_IOU)
        
        # Link camera to neural network
        self.cam.preview.link(self.nn.input)
        
        # Create outputs
        self.rgb_out = self.pipeline.create(dai.node.XLinkOut)
        self.rgb_out.setStreamName("rgb")
        self.cam.preview.link(self.rgb_out.input)
        
        self.nn_out = self.pipeline.create(dai.node.XLinkOut)
        self.nn_out.setStreamName("nn")
        self.nn.out.link(self.nn_out.input)
        
        # Initialize device and queues
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self.q_nn = self.device.getOutputQueue("nn", maxSize=4, blocking=False)
        
        self.frame = None
        self.detections = []
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            # Use get() to ensure packets are synced. tryGet() can lead to mismatches.
            rgb_pkt = self.q_rgb.get()
            nn_pkt = self.q_nn.get()

            if rgb_pkt is not None:
                self.frame = rgb_pkt.getCvFrame()

            if nn_pkt is not None:
                self.detections = nn_pkt.detections
                # Draw detections on the frame
                if self.frame is not None:
                    self.frame = draw_detections(self.frame, self.detections)
            
            time.sleep(0.001)

    def get_frame(self):
        return self.frame
    
    def get_detections(self):
        return self.detections.copy()

    def close(self):
        self.running = False
        try:
            self.device.close()
        except: 
            pass

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
    if not os.path.isfile(BLOB_PATH):
        print(f"Missing blob file: {BLOB_PATH}")
        return
        
    oak = OakClient()
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
            # ArUco detection is now performed on the frame that may have detections drawn on it.
            # This is fine as ArUco uses the grayscale image.
            ids = vis.aruco_ids(f)
            valve_state = None
            pressure = None

            # Get YOLO detections (already processed in the background thread)
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
                # We don't need to scale by image size for angle calculation
                angle = (math.degrees(math.atan2(cy - ty, tx - cx)) + 360) % 360
                pressure = compute_pressure(angle)

            # Telemetry push
            if now - last_telem_post >= 1.0 / POST_TELEM_HZ:
                telem = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "aruco_ids": ids,
                    "ball_gauge": valve_state,  # reuse field name for now
                    "gauge_pressure": pressure,
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

            # Drill command poll
            try:
                r = sess.get(f"{LAPTOP_URL}/control", timeout=2)
                if r.ok and r.json().get("drill") and servo_motor:
                    threading.Thread(target=servo_motor.drill_sequence, daemon=True).start()
            except Exception:
                pass

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        oak.close()

if __name__ == "__main__":
    main()