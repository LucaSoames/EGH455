import cv2
import depthai as dai
import numpy as np
import time
import os
import json
from pathlib import Path

# Suppress warnings
os.environ["QT_QPA_PLATFORM"] = "xcb"
# dai.Device.setLogLevel(dai.LogLevel.ERR)

# CONFIG:
# Path to YOLOv8 blob file and config
BLOB_PATH = "/home/pi/EGH455/models/blobs/YOLOv8s.blob"
CONFIG_PATH = "/home/pi/EGH455/models/blobs/YOLOv8s.json"

# Set detection thresholds
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.01

# IMAGES:
INPUT_PATH = "/home/pi/EGH455/models/testing/images/" # Iterate through images in folder

# VIDEOS:
# INPUT_PATH = "/home/pi/EGH455/models/testing/videos/far_blue.mp4"
# INPUT_PATH = "/home/pi/EGH455/models/testing/videos/far_silver_A.mp4"
# INPUT_PATH = "/home/pi/EGH455/models/testing/videos/far_silver_B.mp4"
# INPUT_PATH = "/home/pi/EGH455/models/testing/videos/near_blue_A.mp4"
# INPUT_PATH = "/home/pi/EGH455/models/testing/videos/near_blue_B.mp4"
# INPUT_PATH = "/home/pi/EGH455/models/testing/videos/near_silver_A.mp4"
# INPUT_PATH = "/home/pi/EGH455/models/testing/videos/near_silver_B.mp4"
# INPUT_PATH = "/home/pi/EGH455/models/testing/videos/near_silver_C.mp4"

# Set to None or "" to use live camera
# INPUT_PATH = None


# HELPER FUNCTIONS
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """Resize and convert to planar format (CHW)"""
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

def load_config(config_path):
    """Load configuration and class names from the YOLOv8 JSON config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    nn_config = config.get("nn_config", {})
    metadata = nn_config.get("NN_specific_metadata", {})
    
    class_names = config.get("mappings", {}).get("labels", [])
    num_classes = metadata.get("classes")
    coordinates = metadata.get("coordinates")
    anchors = metadata.get("anchors", [])
    anchor_masks = metadata.get("anchor_masks", {})
    
    if not all([class_names, num_classes is not None, coordinates is not None]):
        raise ValueError("Configuration file is missing required fields for YOLO detection.")
        
    input_size_str = nn_config.get("input_size")
    if not input_size_str:
        raise ValueError("Input size not found in config.")
    input_size = tuple(map(int, input_size_str.split('x')))

    # print(f"Classes: {class_names}")
    
    return {
        "class_names": class_names,
        "num_classes": num_classes,
        "coordinates": coordinates,
        "anchors": anchors,
        "anchor_masks": anchor_masks,
        "iou_threshold": IOU_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "input_size": input_size
    }

def draw_detections(frame, detections, class_names):
    """Draw detection boxes on the frame with proper text positioning."""
    color_map = {
        "Gauge_Centre": (255, 0, 0),    # Blue
        "Needle_Tip": (0, 255, 255),    # Yellow (corrected from cyan)
        "Valve_Open": (0, 255, 0),      # Green
        "Valve_Closed": (0, 0, 255),    # Red
    }
    default_color = (255, 255, 255)
    
    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    for detection in detections:
        bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        class_id = detection.label
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        color = color_map.get(class_name, default_color)
        
        # Draw bounding box with thicker lines (doubled from 2 to 4)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4)
        
        # Prepare label text
        label = f"{class_name}: {detection.confidence:.2f}"
        
        # Get text size to calculate proper positioning
        font_scale = 1.0
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Position text above the bounding box with padding
        text_x = bbox[0]
        text_y = bbox[1] - 20  # 20 pixels above the box
        
        # If text would go above image boundary, put it inside the box at the top
        if text_y - text_height < 0:
            text_y = bbox[1] + text_height + 10
        
        # Draw background rectangle for text (optional, for better readability)
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + baseline + 5), 
                     (0, 0, 0), 
                     -1)  # Filled black rectangle
        
        # Draw the text
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Draw detection count with larger font
    cv2.putText(frame, f"Detections: {len(detections)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)

# PIPELINE CREATION
def create_camera_pipeline(config):
    """Creates a pipeline for live camera feed."""
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detection_network = pipeline.create(dai.node.YoloDetectionNetwork)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)

    xout_rgb.setStreamName("rgb")
    xout_nn.setStreamName("nn")

    # Camera configuration
    cam_rgb.setPreviewSize(config["input_size"])
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    # YOLO detection network configuration
    detection_network.setBlobPath(BLOB_PATH)
    detection_network.setConfidenceThreshold(config["confidence_threshold"])
    detection_network.setNumClasses(config["num_classes"])
    detection_network.setCoordinateSize(config["coordinates"])
    detection_network.setAnchors(config["anchors"])
    detection_network.setAnchorMasks(config["anchor_masks"])
    detection_network.setIouThreshold(config["iou_threshold"])
    detection_network.setNumInferenceThreads(2)
    detection_network.input.setBlocking(False)
    
    # Link nodes
    cam_rgb.preview.link(detection_network.input)
    detection_network.passthrough.link(xout_rgb.input)
    detection_network.out.link(xout_nn.input)
    
    return pipeline

def create_file_pipeline(config):
    """Creates a pipeline for processing files from the host."""
    pipeline = dai.Pipeline()
    xinFrame = pipeline.create(dai.node.XLinkIn)
    detection_network = pipeline.create(dai.node.YoloDetectionNetwork)
    xout_nn = pipeline.create(dai.node.XLinkOut)
    
    xinFrame.setStreamName("inFrame")
    xout_nn.setStreamName("nn")

    # YOLO detection network configuration
    detection_network.setBlobPath(BLOB_PATH)
    detection_network.setConfidenceThreshold(config["confidence_threshold"])
    detection_network.setNumClasses(config["num_classes"])
    detection_network.setCoordinateSize(config["coordinates"])
    detection_network.setAnchors(config["anchors"])
    detection_network.setAnchorMasks(config["anchor_masks"])
    detection_network.setIouThreshold(config["iou_threshold"])
    detection_network.setNumInferenceThreads(2)
    detection_network.input.setBlocking(False)
    
    # Link nodes
    xinFrame.out.link(detection_network.input)
    detection_network.out.link(xout_nn.input)
    
    return pipeline

# MAIN
def main():
    
    # Load JSON and set up window
    config = load_config(CONFIG_PATH)
    model_name = Path(BLOB_PATH).stem
    window_name = f"{model_name} Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Determine input mode
    use_live_camera = INPUT_PATH is None or INPUT_PATH == "" or not Path(INPUT_PATH).exists()
    
    if not use_live_camera:
        path = Path(INPUT_PATH)
        print(f"Processing input: {INPUT_PATH}")
        
        # --- HOST-SIDE (FILE) PROCESSING ---
        pipeline = create_file_pipeline(config)
        
        try:
            with dai.Device(pipeline) as device:
                qIn = device.getInputQueue(name="inFrame")
                qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

                def process_frame(frame):
                    if frame is None:
                        return False
                    
                    # Prepare frame for inference
                    img = dai.ImgFrame()
                    img.setData(to_planar(frame, config["input_size"]))
                    img.setType(dai.ImgFrame.Type.BGR888p)
                    img.setWidth(config["input_size"][0])
                    img.setHeight(config["input_size"][1])
                    qIn.send(img)

                    # Get detections
                    in_det = qDet.get()
                    if in_det is not None:
                        detections = in_det.detections
                        draw_detections(frame, detections, config["class_names"])
                        cv2.imshow(window_name, frame)
                    
                    return True

                if path.is_dir():
                    # Process images in directory
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                    image_files = sorted([p for p in path.glob('*') if p.suffix.lower() in image_extensions])
                    
                    if not image_files:
                        print(f"No images found in {path}")
                        return
                    
                    print(f"Found {len(image_files)} images")
                    for i, image_file in enumerate(image_files):
                        print(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
                        frame = cv2.imread(str(image_file))
                        if not process_frame(frame):
                            continue
                        
                        key = cv2.waitKey(0)
                        if key == ord('q'):
                            break
                        elif key == ord('n'):  # Next image
                            continue
                            
                elif path.is_file():
                    # Process video file
                    print(f"Processing video: {path}")
                    cap = cv2.VideoCapture(str(path))
                    
                    if not cap.isOpened():
                        print(f"Error: Could not open video file {path}")
                        return
                    
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            print("End of video or failed to read frame")
                            break
                        
                        frame_count += 1
                        if not process_frame(frame):
                            continue
                            
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord(' '):  # Pause on spacebar
                            cv2.waitKey(0)
                    
                    cap.release()
                    print(f"Processed {frame_count} frames")
                    
        except Exception as e:
            print(f"Error during file processing: {e}")

    else:
        # --- LIVE CAMERA PROCESSING ---
        print("Starting live camera feed...")
        pipeline = create_camera_pipeline(config)
        
        try:
            with dai.Device(pipeline) as device:
                q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
                
                print("Camera started. Press 'q' to quit.")
                
                while True:
                    in_rgb = q_rgb.get()
                    in_nn = q_nn.get()
                    
                    if in_rgb is not None:
                        frame = in_rgb.getCvFrame()
                        
                        if in_nn is not None:
                            detections = in_nn.detections
                            draw_detections(frame, detections, config["class_names"])
                        
                        cv2.imshow(window_name, frame)
                    
                    if cv2.waitKey(1) == ord('q'):
                        break
                        
        except Exception as e:
            print(f"Error during live camera processing: {e}")

    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()