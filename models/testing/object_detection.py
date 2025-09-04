import cv2
import depthai as dai
import numpy as np
import time
import os
import json

# TEST WITH A VIDEO OR IMAGE - uncomment the one you want to use
# INPUT_PATH = "/home/pi/EGH455/models/testing/images/Gauge.mp4"
# INPUT_PATH = "/home/pi/EGH455/models/testing/images/Gauge_far.mp4"
INPUT_PATH = "/home/pi/EGH455/models/testing/images/Ball Valve Open Floor.jpg"
# INPUT_PATH = "/home/pi/EGH455/models/testing/images/Ball Valve Open.png"

# Path to YOLOv11n blob file and config
# BLOB_PATH = "/home/pi/EGH455/models/blobs/HubAI/YOLOv11n.blob"
# CONFIG_PATH = "/home/pi/EGH455/models/blobs/HubAI/YOLOv11n.json"

# Path to YOLOv8 blob file and config
BLOB_PATH = "/home/pi/EGH455/models/blobs/Roboflow/YOLOv8.blob"
CONFIG_PATH = "/home/pi/EGH455/models/blobs/Roboflow/YOLOv8.json"

# Detection parameters - adjusted for better detection
CONFIDENCE_THRESHOLD = 0.35  # Lowered to catch more potential detections
IOU_THRESHOLD = 0.01         # Standard NMS threshold
MIN_BOX_AREA = 0.001         # Minimum box area (as fraction of image)
MAX_BOX_AREA = 0.1           # Maximum box area (as fraction of image)


def load_config():
    """Load configuration and class names from the config file."""
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    if 'mappings' in cfg and 'nn_config' in cfg:
        # Roboflow/YOLOv8 JSON
        class_names = cfg['mappings']['labels']
        num_classes = cfg['nn_config']['NN_specific_metadata']['classes']
        input_size = tuple(map(int, cfg['nn_config']['input_size'].split('x')))  # (h, w)
        input_name = 'images'      # XLinkIn stream name
    else:
        # HubAI format
        model = cfg['model']
        class_names = model['heads'][0]['metadata']['classes']
        num_classes = model['heads'][0]['metadata']['n_classes']
        input_name = model['inputs'][0]['name']
        input_size = model['inputs'][0]['shape'][2:4]
    print(f"Model expects input '{input_name}' with size {input_size[0]}x{input_size[1]}")
    print(f"Classes: {class_names}")
    
    return class_names, num_classes, input_name, input_size

def create_pipeline(blob_path, input_name):
    """Create a pipeline for the neural network."""
    pipeline = dai.Pipeline()
    
    # Create neural network node
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    nn.setNumInferenceThreads(2)
    
    # Create input
    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName(input_name)
    
    # Create output
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("nn")
    
    # Link nodes
    xin.out.link(nn.input)
    nn.out.link(xout.input)
    
    return pipeline

def preprocess_image(frame, input_size):
    """Preprocess image: resize & convert to planar BGR uint8 for the NN."""
    # Resize
    resized = cv2.resize(frame, (input_size[1], input_size[0]))
    # HWC -> CHW and uint8 (BGR)
    img = resized.transpose(2, 0, 1).astype(np.uint8)
    return img

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def process_yolo_output(layer_data, grid_size, num_classes):
    """Process a YOLO output layer."""
    detections = []
    
    # Calculate values per grid cell
    values_per_cell = 5 + num_classes  # 4 box coords + 1 obj conf + num_classes
    
    try:
        # The NN output is in NCHW format. Reshape to (channels, height, width)
        # then transpose to (height, width, channels) for easier processing.
        expected_elements = values_per_cell * grid_size * grid_size
        if layer_data.size != expected_elements:
            print(f"Error: Layer data size ({layer_data.size}) does not match expected size ({expected_elements}) for grid {grid_size}x{grid_size}.")
            return []
            
        output = layer_data.reshape((values_per_cell, grid_size, grid_size)).transpose(1, 2, 0)
        
        # Process each cell in the grid
        for cy in range(grid_size):
            for cx in range(grid_size):
                # Get data for this grid cell
                cell_data = output[cy, cx]
                
                # Extract objectness score
                obj_conf = sigmoid(cell_data[4])
                
                if obj_conf > CONFIDENCE_THRESHOLD:
                    # Get class probabilities and find highest scoring class
                    class_scores = sigmoid(cell_data[5:])
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    
                    # Calculate final confidence
                    confidence = obj_conf * class_conf
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        # Get box coordinates (normalized)
                        # YOLOv8/v11n format: x, y are the center of the box
                        cx_rel = (cx + sigmoid(cell_data[0])) / grid_size
                        cy_rel = (cy + sigmoid(cell_data[1])) / grid_size
                        
                        # Correct width/height decoding for anchor-free YOLOv8/v11
                        w_rel = (2 * sigmoid(cell_data[2]))**2 / grid_size
                        h_rel = (2 * sigmoid(cell_data[3]))**2 / grid_size
                        
                        # Box area filtering - reject very small or large boxes
                        box_area = w_rel * h_rel
                        if box_area < MIN_BOX_AREA or box_area > MAX_BOX_AREA:
                            continue
                        
                        # Convert to corner format (top-left, bottom-right)
                        x1 = max(0, cx_rel - w_rel/2)
                        y1 = max(0, cy_rel - h_rel/2)
                        x2 = min(1, cx_rel + w_rel/2)
                        y2 = min(1, cy_rel + h_rel/2)
                        
                        detections.append({
                            'x1': float(x1), 
                            'y1': float(y1), 
                            'x2': float(x2), 
                            'y2': float(y2),
                            'confidence': float(confidence),
                            'class_id': int(class_id)
                        })
    except ValueError as e:
        print(f"Error reshaping layer data for grid size {grid_size}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in process_yolo_output for grid size {grid_size}: {e}")
    
    return detections

def non_max_suppression(detections, iou_threshold):
    """Apply Non-Maximum Suppression to remove overlapping detections."""
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    
    # Keep track of boxes to keep
    keep = []
    
    while sorted_dets:
        # Keep the highest confidence detection
        best_det = sorted_dets[0]
        keep.append(best_det)
        
        if len(sorted_dets) == 1:
            break
            
        # Remove the best detection from the list
        sorted_dets.pop(0)
        
        # Filter out detections that overlap too much with the best detection
        filtered_dets = []
        best_box = [best_det['x1'], best_det['y1'], best_det['x2'], best_det['y2']]
        best_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
        
        for det in sorted_dets:
            curr_box = [det['x1'], det['y1'], det['x2'], det['y2']]
            curr_area = (curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])
            
            # Calculate intersection
            x1 = max(best_box[0], curr_box[0])
            y1 = max(best_box[1], curr_box[1])
            x2 = min(best_box[2], curr_box[2])
            y2 = min(best_box[3], curr_box[3])
            
            # Skip if boxes don't overlap
            if x2 <= x1 or y2 <= y1:
                filtered_dets.append(det)
                continue
                
            inter_area = (x2 - x1) * (y2 - y1)
            iou = inter_area / (best_area + curr_area - inter_area)
            
            if iou <= iou_threshold:
                filtered_dets.append(det)
        
        # Update the list
        sorted_dets = filtered_dets
    
    return keep

def draw_detections(frame, detections, class_names):
    """Draw detection boxes on the frame."""
    h, w = frame.shape[:2]
    
    # Create a color map based on the class names provided
    # This ensures colors are consistent regardless of the model's label order
    color_map = {
        "needle_tip": (0, 0, 255),    # Red
        "gauge_centre": (0, 255, 0),    # Green
        "valve_open": (255, 0, 0),    # Blue
        "valve_closed": (255, 255, 0),  # Cyan
    }
    default_color = (255, 255, 255) # White for unknown classes
    
    for detection in detections:
        # Get box coordinates in pixel space
        x1 = int(detection['x1'] * w)
        y1 = int(detection['y1'] * h)
        x2 = int(detection['x2'] * w)
        y2 = int(detection['y2'] * h)
        
        # Skip tiny boxes (e.g., less than 10x10 pixels)
        if (x2-x1) < 10 or (y2-y1) < 10:
            continue
            
        # Get class info
        class_id = detection['class_id']
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        confidence = detection['confidence']
        
        # Choose color based on class name from the map
        color = color_map.get(class_name, default_color)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - baseline // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1) # Black text for better contrast
    
    # Add detection count to top of image
    cv2.putText(frame, f"Detections: {len(detections)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def process_output(nn_data, class_names, num_classes, original_frame):
    """Process network output and draw detections."""
    result_frame = original_frame.copy()
    
    # Get all layer names and tensors
    layers = nn_data.getAllLayers()
    
    # Process detections from each layer
    all_detections = []
    
    for layer in layers:
        # The layer shape is (1, channels, height, width)
        # channels = num_classes + 5
        # height and width are the grid size
        grid_size = layer.dims[2]
        
        # Get layer data as a numpy array of FP16 values, and convert to FP32 for processing.
        layer_data = np.array(nn_data.getLayerFp16(layer.name), dtype=np.float32)
        
        # Process this output layer
        detections = process_yolo_output(layer_data, grid_size, num_classes)
        all_detections.extend(detections)
    
    # Apply NMS to filter overlapping boxes
    filtered_detections = non_max_suppression(all_detections, IOU_THRESHOLD)
    
    # Optional: Filter out detections with very high confidence if needed for debugging
    # high_conf_detections = [d for d in filtered_detections if d['confidence'] > 0.9]
    # print(f"Found {len(high_conf_detections)} high-confidence detections (>0.9)")
    
    # Draw detections
    draw_detections(result_frame, filtered_detections, class_names)
    
    return result_frame, filtered_detections

def process_frame(frame, input_name, input_size, q_in, q_nn, class_names, num_classes):
    """Process a single frame and return the detection results."""
    # Preprocess image
    img = preprocess_image(frame, input_size)
    
    # Create tensor
    tensor = dai.NNData()
    tensor.setLayer(input_name, img)
    
    # Send the tensor for inference
    q_in.send(tensor)
    
    # Wait for inference results with timeout
    start_time = time.time()
    nn_data = None
    
    # Try to get results for up to 1 second
    while time.time() - start_time < 1.0:
        nn_data = q_nn.tryGet()
        if nn_data is not None:
            break
        time.sleep(0.01)  # Small delay
    
    if nn_data is not None:
        # Process output
        result_frame, detections = process_output(nn_data, class_names, num_classes, frame)
        return result_frame, detections
    else:
        print("Warning: Inference timeout")
        # Draw timeout message
        cv2.putText(frame, "INFERENCE TIMEOUT", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, []

def process_video(video_path, input_name, input_size, device, class_names, num_classes):
    """Process video file frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    print(f"Processing video: {os.path.basename(video_path)}")
    
    # Create input/output queues
    q_in = device.getInputQueue(input_name)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)
    
    frame_count = 0
    start_time = time.time()
    
    model_name = os.path.splitext(os.path.basename(BLOB_PATH))[0]
    
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result_frame, detections = process_frame(frame, input_name, input_size, q_in, q_nn, class_names, num_classes)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps_text = f"FPS: {frame_count / elapsed_time:.1f}"
            cv2.putText(result_frame, fps_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow(f"{model_name} Detection", result_frame)
        
        # Check for exit key (q)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    print(f"Processed {frame_count} frames in {elapsed_time:.1f} seconds")

def process_image(image_path, input_name, input_size, device, class_names, num_classes):
    """Process a single image file."""
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    print(f"Processing image: {os.path.basename(image_path)}")
    
    # Create input/output queues
    q_in = device.getInputQueue(input_name)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)
    
    # Process frame
    result_frame, detections = process_frame(frame, input_name, input_size, q_in, q_nn, class_names, num_classes)
    
    # Display result
    model_name = os.path.splitext(os.path.basename(BLOB_PATH))[0]
    cv2.imshow(f"{model_name} Detection", result_frame)
    cv2.waitKey(0)

def main():
    # Check if files exist
    if not os.path.exists(BLOB_PATH) or not os.path.exists(CONFIG_PATH):
        print(f"Error: Required files not found")
        return
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found: {INPUT_PATH}")
        return
    
    # Load configuration
    print("Loading configuration...")
    class_names, num_classes, input_name, input_size = load_config()
    
    # Create pipeline
    model_name = os.path.splitext(os.path.basename(BLOB_PATH))[0]
    print(f"Creating pipeline for {model_name}...")
    pipeline = create_pipeline(BLOB_PATH, input_name)
    
    # Initialize device
    with dai.Device(pipeline) as device:
        print("Pipeline created, starting inference...")
        
        # Process input based on type
        if INPUT_PATH.lower().endswith(('.mp4', '.avi', '.mov')):
            process_video(INPUT_PATH, input_name, input_size, device, class_names, num_classes)
        else:
            process_image(INPUT_PATH, input_name, input_size, device, class_names, num_classes)
    
    cv2.destroyAllWindows()
    print("Processing complete")

if __name__ == "__main__":
    main()