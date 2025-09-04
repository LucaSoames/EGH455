import time
import depthai as dai
import os

def benchmark_model(blob_path, num_frames=200):
    # Build pipeline
    pipeline = dai.Pipeline()
    
    # Create color camera
    cam = pipeline.createColorCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(15)
    
    # Create ImageManip to ensure correct format for the neural network
    manip = pipeline.createImageManip()
    manip.initialConfig.setResize(640, 640)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    # Set maximum output frame size to accommodate 640x640x3 bytes
    manip.setMaxOutputFrameSize(1228800)  # 640 * 640 * 3 = 1,228,800
    cam.preview.link(manip.inputImage)
    
    # Neural network
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath(blob_path)
    manip.out.link(nn.input)
    
    # Output
    xout = pipeline.createXLinkOut()
    xout.setStreamName("out")
    nn.out.link(xout.input)

    # Run device
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("out", maxSize=4, blocking=False)
        
        for _ in range(30):
            if q.has():
                q.get()
            time.sleep(0.01)
        
        # Benchmark
        start = time.time()
        frame_count = 0
        
        for i in range(num_frames):
            if q.has():
                q.get()
                frame_count += 1
            time.sleep(0.01)
            
        elapsed = time.time() - start
        
    fps = frame_count / elapsed
    return fps, elapsed / frame_count if frame_count > 0 else 0

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))  # models/testing
models_dir = os.path.dirname(script_dir)  # models folder
blobs_dir = os.path.join(models_dir, "blobs")  # Path to models/blobs

# Dynamically collect all .blob files in models/blobs
blobs = {}
if os.path.exists(blobs_dir):
    for file in os.listdir(blobs_dir):
        if file.endswith(".blob"):
            name = os.path.splitext(file)[0]  # Use filename without extension as key
            path = os.path.join(blobs_dir, file)
            blobs[name] = path
else:
    print(f"Error: Blobs directory not found at {blobs_dir}")
    blobs = {}

if not blobs:
    print("No .blob files found in the blobs directory. Exiting.")
else:
    for name, path in blobs.items():
        if not os.path.exists(path):
            print(f"Error: Blob file not found at {path}")
            continue
            
        try:
            fps, latency = benchmark_model(path)
            print(f"{name}: {fps:.1f} FPS, {latency*1000:.1f} ms/frame")
        except Exception as e:
            print(f"Error testing {name}: {e}")