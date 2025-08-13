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

def parse_yolo_output(output, conf_threshold=0.35, iou_threshold=0.45, img_size=640):
    """Parse YOLOv11 output and apply NMS"""
    # YOLOv11 output shape: [1, 84, 8400] where 84 = 4 (bbox) + 80 (classes for COCO) 
    # For our model: [1, 8, 8400] where 8 = 4 (bbox) + 4 (our classes)
    output = output[0]  # Remove batch dimension: [8, 8400]
    
    # Transpose to [8400, 8]
    output = output.transpose()
    
    # Extract bounding boxes and class scores
    boxes = output[:, :4]  # [8400, 4] - x_center, y_center, width, height (normalized)
    scores = output[:, 4:]  # [8400, 4] - class scores
    
    detections = []
    
    for i in range(len(boxes)):
        # Get class with highest score
        class_scores = scores[i]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence >= conf_threshold:
            # Convert from center format to corner format
            x_center, y_center, width, height = boxes[i]
            
            # Convert from normalized coordinates to pixel coordinates
            x_center *= img_size
            y_center *= img_size
            width *= img_size
            height *= img_size
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            detections.append({
                'class_id': int(class_id),
                'confidence': float(confidence),
                'bbox': [x1, y1, x2, y2]
            })
    
    # Apply Non-Maximum Suppression
    if len(detections) > 0:
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            detections = [detections[i] for i in indices]
    
    return detections

class OakClient:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        
        # Create color camera node
        self.cam = self.pipeline.create(dai.node.ColorCamera)
        self.cam.setPreviewSize(MODEL_IMG_SIZE, MODEL_IMG_SIZE)  # Square input for YOLO
        self.cam.setFps(30)
        self.cam.setInterleaved(False)
        self.cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        # Create generic neural network node instead of YoloDetectionNetwork
        self.nn = self.pipeline.create(dai.node.NeuralNetwork)
        self.nn.setBlobPath(BLOB_PATH)
        self.nn.setNumInferenceThreads(2)
        self.nn.input.setBlocking(False)
        
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
            # Get RGB frame
            rgb_pkt = self.q_rgb.tryGet()
            if rgb_pkt is not None:
                self.frame = rgb_pkt.getCvFrame()
            
            # Get neural network output
            nn_pkt = self.q_nn.tryGet()
            if nn_pkt is not None:
                # Get raw output tensor
                output = nn_pkt.getFirstLayerFp16()
                output = np.array(output).reshape(1, NUM_CLASSES + 4, -1)  # Reshape to [1, 8, 8400]
                
                # Parse YOLO output
                raw_detections = parse_yolo_output(output, MODEL_CONF, MODEL_IOU, MODEL_IMG_SIZE)
                
                # Convert to our detection format
                detections = []
                for det in raw_detections:
                    detection = {
                        "cls_id": det['class_id'],
                        "name": CLASS_NAMES.get(det['class_id'], f"class_{det['class_id']}"),
                        "conf": det['confidence'],
                        "box": det['bbox']  # [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                self.detections = detections
            
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
            run_det = (int(now * 10) % 3 == 0)  # ~3-4 Hz for ArUco only
            ids = []
            valve_state = None
            pressure = None

            if run_det:
                ids = vis.aruco_ids(f)

            # Get YOLO detections (running continuously on OAK-D)
            dets = oak.get_detections()

            # Extract valve open/closed (pick highest confidence if both)
            open_candidates = [d for d in dets if d["name"] == "valve_open"]
            closed_candidates = [d for d in dets if d["name"] == "valve_closed"]
            if open_candidates or closed_candidates:
                best_open = max(open_candidates, key=lambda d: d["conf"]) if open_candidates else None
                best_closed = max(closed_candidates, key=lambda d: d["conf"]) if closed_candidates else None
                if best_open and best_closed:
                    valve_state = "open" if best_open["conf"] >= best_closed["conf"] else "closed"
                elif best_open:
                    valve_state = "open"
                elif best_closed:
                    valve_state = "closed"

            # Needle tip / gauge centre for pressure reading
            needle_tip = next((d for d in dets if d["name"] == "needle_tip"), None)
            gauge_centre = next((d for d in dets if d["name"] == "gauge_centre"), None)
            if needle_tip and gauge_centre:
                # Use centers of bounding boxes
                tx = (needle_tip["box"][0] + needle_tip["box"][2]) / 2.0
                ty = (needle_tip["box"][1] + needle_tip["box"][3]) / 2.0
                cx = (gauge_centre["box"][0] + gauge_centre["box"][2]) / 2.0
                cy = (gauge_centre["box"][1] + gauge_centre["box"][3]) / 2.0
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