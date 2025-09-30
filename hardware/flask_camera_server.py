#!/usr/bin/env python3

from flask import Flask, Response, render_template_string, request
import cv2
import depthai as dai
import threading
import time
import numpy as np

app = Flask(__name__)

class CameraStream:
    def __init__(self):
        self.rgb_frame = None
        self.depth_frame = None
        self.colormap = cv2.COLORMAP_HOT  # Default colormap
        self.pipeline = dai.Pipeline()
        
        # Define sources and outputs
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        depth = self.pipeline.create(dai.node.StereoDepth)
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        
        # Create output streams
        depthOut = self.pipeline.create(dai.node.XLinkOut)
        rgbOut = self.pipeline.create(dai.node.XLinkOut)
        
        depthOut.setStreamName("depth")
        rgbOut.setStreamName("rgb")
        
        # Properties - Updated for OV7251 sensors (OAK-D Lite)
        # OV7251 only supports 480p and 400p resolutions
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Updated from LEFT
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Updated from RIGHT
        
        # Create depth output - Updated preset
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)  # Updated from HIGH_ACCURACY
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        depth.setSubpixel(False)
        
        # RGB camera properties
        camRgb.setPreviewSize(640, 480)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        
        # Linking
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.depth.link(depthOut.input)
        camRgb.preview.link(rgbOut.input)
        
        # Connect to device and start pipeline
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        # Start capture thread
        self.thread = threading.Thread(target=self.update_frames)
        self.thread.daemon = True
        self.thread.start()
    
    def set_colormap(self, colormap):
        self.colormap = colormap
    
    def update_frames(self):
        while True:
            try:
                # Get RGB frame
                inRgb = self.q_rgb.tryGet()
                if inRgb is not None:
                    self.rgb_frame = inRgb.getCvFrame()
                
                # Get depth frame
                inDepth = self.q_depth.tryGet()
                if inDepth is not None:
                    depth_frame = inDepth.getFrame()
                    # Convert depth to colormap for visualization
                    depth_frame_color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depth_frame_color = cv2.equalizeHist(depth_frame_color)
                    self.depth_frame = cv2.applyColorMap(depth_frame_color, self.colormap)
                
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in update_frames: {e}")
                break
    
    def get_rgb_frame(self):
        if self.rgb_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.rgb_frame)
            return buffer.tobytes()
        return None
    
    def get_depth_frame(self):
        if self.depth_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.depth_frame)
            return buffer.tobytes()
        return None
    
    def cleanup(self):
        """Clean up device resources"""
        try:
            if hasattr(self, 'device'):
                self.device.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

camera = CameraStream()

# Colormap mapping
COLORMAPS = {
    'hot': cv2.COLORMAP_HOT,
    'jet': cv2.COLORMAP_JET,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'plasma': cv2.COLORMAP_PLASMA,
    'inferno': cv2.COLORMAP_INFERNO,
    'magma': cv2.COLORMAP_MAGMA,
    'rainbow': cv2.COLORMAP_RAINBOW,
    'turbo': cv2.COLORMAP_TURBO,
    'parula': cv2.COLORMAP_PARULA,
    'bone': cv2.COLORMAP_BONE,
    'cool': cv2.COLORMAP_COOL,
    'autumn': cv2.COLORMAP_AUTUMN
}

def generate_rgb_frames():
    while True:
        frame = camera.get_rgb_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

def generate_depth_frames():
    while True:
        frame = camera.get_depth_frame()
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
        <title>Raspberry Pi OAK-D Lite Camera Stream</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                margin: 20px; 
                background-color: #f0f0f0;
            }
            .camera-container {
                display: flex;
                justify-content: center;
                gap: 20px;
                flex-wrap: wrap;
                margin: 20px 0;
            }
            .camera-feed {
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .depth-controls {
                margin-top: 15px;
                padding: 10px;
                background: #f8f8f8;
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            img { 
                border: 2px solid #333; 
                border-radius: 10px; 
                width: 640px;
                height: 480px;
                object-fit: cover;
            }
            h1 { color: #333; margin-bottom: 10px; }
            h3 { color: #666; margin: 10px 0; }
            .info { color: #888; font-size: 14px; margin: 5px 0; }
            select {
                padding: 8px 12px;
                font-size: 14px;
                border: 2px solid #333;
                border-radius: 5px;
                background: white;
                margin-left: 8px;
                min-width: 200px;
            }
            label {
                font-weight: bold;
                color: #333;
                font-size: 14px;
            }
            .footer-info {
                margin-top: 20px;
                color: #666;
                font-size: 14px;
            }
        </style>
        <script>
            function changeColormap() {
                const colormap = document.getElementById('colormap').value;
                fetch('/set_colormap', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({colormap: colormap})
                });
            }
        </script>
    </head>
    <body>
        <h1>Raspberry Pi 5 - OAK-D Lite Camera</h1>
        
        <div class="camera-container">
            <div class="camera-feed">
                <h3>RGB Camera</h3>
                <img src="/rgb_feed" alt="RGB Camera Feed">
                <p class="info">Color feed | 640x480</p>
            </div>
            
            <div class="camera-feed">
                <h3>Depth Camera</h3>
                <img src="/depth_feed" alt="Depth Camera Feed">
                <p class="info">Depth visualization | 480p mono cameras</p>
                
                <div class="depth-controls">
                    <label for="colormap">Depth Colormap:</label>
                    <select id="colormap" onchange="changeColormap()">
                        <option value="hot" selected>Hot (Red-Yellow)</option>
                        <option value="jet">Jet (Rainbow)</option>
                        <option value="viridis">Viridis (Purple-Yellow)</option>
                        <option value="plasma">Plasma (Purple-Pink-Yellow)</option>
                        <option value="inferno">Inferno (Black-Red-Yellow)</option>
                        <option value="magma">Magma (Black-Purple-Pink)</option>
                        <option value="rainbow">Rainbow</option>
                        <option value="turbo">Turbo (Blue-Green-Red)</option>
                        <option value="parula">Parula (Blue-Green-Yellow)</option>
                        <option value="bone">Bone (Black-White)</option>
                        <option value="cool">Cool (Cyan-Magenta)</option>
                        <option value="autumn">Autumn (Red-Yellow)</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="footer-info">
            <p>Refresh rate: ~30 FPS | OAK-D Lite with OV7251 sensors</p>
        </div>
    </body>
    </html>
    ''')

@app.route('/set_colormap', methods=['POST'])
def set_colormap():
    data = request.get_json()
    colormap_name = data.get('colormap', 'hot')
    
    if colormap_name in COLORMAPS:
        camera.set_colormap(COLORMAPS[colormap_name])
        return {'status': 'success', 'colormap': colormap_name}
    else:
        return {'status': 'error', 'message': 'Invalid colormap'}, 400

@app.route('/rgb_feed')
def rgb_feed():
    return Response(generate_rgb_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depth_feed')
def depth_feed():
    return Response(generate_depth_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        print("Starting Flask camera server with RGB and Depth...")
        print("OAK-D Lite detected - using 480p mono cameras")
        print("Access the camera feeds at: http://10.88.12.62:5000")
        print("RGB feed only: http://10.88.12.62:5000/rgb_feed")
        print("Depth feed only: http://10.88.12.62:5000/depth_feed")
        print("Press Ctrl+C to stop the server")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        camera.cleanup()
    except Exception as e:
        print(f"Error starting server: {e}")
        camera.cleanup()