import os, time, math, threading
import requests
import cv2
import numpy as np
import depthai as dai
from ultralytics import YOLO  # NEW

LAPTOP_URL = os.environ.get("LAPTOP_URL", "http://192.168.86.249:5000")
POST_FRAME_FPS = 10
POST_TELEM_HZ = 5

WEIGHTS_PATH = "/home/pi/EGH455/models/valve_rfdetr.pt"  # ensure file exists
MODEL_IMG_SIZE = 640
MODEL_CONF = 0.35
GAUGE_MIN_ANGLE = 225.0   # adjust after calibration
GAUGE_MAX_ANGLE = -45.0
GAUGE_MIN_P = 0.0
GAUGE_MAX_P = 100.0

try:
    import servo_motor
except Exception:
    servo_motor = None

class OakClient:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.ColorCamera)
        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam.setPreviewSize(640, 480)
        cam.setFps(30)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.preview.link(xout.input)
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            pkt = self.q_rgb.tryGet()
            if pkt is not None:
                self.frame = pkt.getCvFrame()
            time.sleep(0.001)

    def get_frame(self):
        return self.frame

    def close(self):
        self.running = False
        try:
            self.device.close()
        except: pass

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

# NEW: YOLO wrapper for RF-DETR weights
class ValveDetector:
    def __init__(self, weights):
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"Missing weights: {weights}")
        self.model = YOLO(weights)
        # Map class indices to names (ensure matches training order)
        # If model.names already matches, we use it directly.
        self.names = self.model.names  # dict: idx->name

    def infer(self, frame):
        # Returns list of dict {cls, name, conf, box:[x1,y1,x2,y2]}
        results = self.model.predict(frame, imgsz=MODEL_IMG_SIZE, conf=MODEL_CONF, verbose=False)[0]
        out = []
        if results.boxes is None:
            return out
        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        for (x1,y1,x2,y2), cid, cf in zip(boxes_xyxy, cls_ids, confs):
            out.append({
                "cls_id": int(cid),
                "name": self.names.get(int(cid), str(cid)),
                "conf": float(cf),
                "box": [float(x1), float(y1), float(x2), float(y2)]
            })
        return out

def compute_pressure(angle_deg):
    a0 = (GAUGE_MIN_ANGLE + 360) % 360
    a1 = (GAUGE_MAX_ANGLE + 360) % 360
    def ad(a,b): return (a-b+540) % 360 - 180
    span = ad(a1, a0)
    pos = ad(angle_deg, a0)
    t = 0.0 if span == 0 else max(0.0, min(1.0, pos / span))
    return GAUGE_MIN_P + t * (GAUGE_MAX_P - GAUGE_MIN_P)

def main():
    oak = OakClient()
    vis = Vision()
    try:
        valve_model = ValveDetector(WEIGHTS_PATH)
    except Exception as e:
        print(f"Model load failed: {e}")
        return

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
            run_det = (int(now * 10) % 3 == 0)  # ~3-4 Hz
            ids = []
            valve_state = None
            pressure = None

            if run_det:
                ids = vis.aruco_ids(f)
                dets = valve_model.infer(f)

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

                # Tip / centre for pressure
                tip = next((d for d in dets if d["name"] == "tip"), None)
                centre = next((d for d in dets if d["name"] == "centre"), None)
                if tip and centre:
                    # Use centers
                    tx = (tip["box"][0] + tip["box"][2]) / 2.0
                    ty = (tip["box"][1] + tip["box"][3]) / 2.0
                    cx = (centre["box"][0] + centre["box"][2]) / 2.0
                    cy = (centre["box"][1] + centre["box"][3]) / 2.0
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