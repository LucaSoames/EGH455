#!/usr/bin/env python3

from flask import Flask, Response, render_template_string, jsonify
import cv2
import depthai as dai
import threading
import time

import db
db.init_db()

import pimoroni_data  # ensure this file is in the same directory or on PYTHONPATH

# Launch the Pimoroni sensor loop in a daemon thread
sensor_thread = threading.Thread(target=pimoroni_data.main, daemon=True)
sensor_thread.start()

app = Flask(__name__)

class CameraStream:
    def __init__(self):
        self.frame = None
        self.pipeline = dai.Pipeline()
        
        # Define source and output
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        
        # Properties
        camRgb.setPreviewSize(640, 480)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        
        # Linking
        camRgb.preview.link(xout.input)
        
        self.device = dai.Device(self.pipeline)
        self.q = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        # Start capture thread
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()
    
    def update_frame(self):
        while True:
            inRgb = self.q.get()
            self.frame = inRgb.getCvFrame()
    
    def get_frame(self):
        if self.frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.frame)
            return buffer.tobytes()
        return None

camera = CameraStream()

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi Camera & Sensors</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }
            img { border: 2px solid #333; border-radius: 10px; }
            #sensor-data { margin-top: 20px; font-size: 1.1em; }
        </style>
    </head>
    <body>
        <h1>Live Camera & Enviro+ Readings</h1>
        <img src="/video_feed" style="width:100%;max-width:640px;">
        <div id="sensor-data">Loading sensor data…</div>

        <script>
        async function fetchSensor() {
            try {
                const res = await fetch('/sensor_data');
                const d = await res.json();
                if (Object.keys(d).length) {
                    document.getElementById('sensor-data').innerHTML = `
                        <strong>${d.timestamp}</strong><br>
                        Temp: ${d.temperature.toFixed(2)}°C &nbsp;
                        Humidity: ${d.humidity.toFixed(2)}%<br>
                        Gas R⁺=${d.gas_reducing}Ω, Oₓ=${d.gas_oxidising}Ω, NH₃=${d.gas_nh3}Ω<br>
                        Light: ${d.light.toFixed(1)} lux &nbsp;
                        Pr: ${d.proximity}
                    `;
                } else {
                    document.getElementById('sensor-data').innerText = 'No data yet.';
                }
            } catch(e) {
                console.error(e);
            }
        }
        setInterval(fetchSensor, 2000);
        window.onload = fetchSensor;
        </script>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sensor_data')
def sensor_data():
    latest = db.get_latest_reading() or {}
    return jsonify(latest)

if __name__ == '__main__':
    print("Starting Flask camera server (with Enviro+ logger)…")
    print("Access the camera feed at: http://<your-pi-ip>:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
