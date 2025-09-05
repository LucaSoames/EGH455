import os
import sys
import json
from ultralytics import YOLO
import blobconverter

# --- PATHS & MODEL NAME ---
MODEL_NAME = "YOLOv8s" # Change this to convert a different model
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PT_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt")
ONNX_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")
OUTPUT_BLOB_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.blob")

# --- MODEL CONFIG ---
IMG_SIZE = 640
OPSET = 11
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.1

# --- CLASS NAMES (in the order they were trained) ---
CLASS_NAMES = [
    "Gauge_Centre",
    "Needle_Tip",
    "Valve_Closed",
    "Valve_Open"
]
NUM_CLASSES = len(CLASS_NAMES)


def export_to_onnx():
    """Exports the PyTorch model to ONNX format."""
    print(f"[INFO] Loading model from {PT_PATH}...")
    
    if not os.path.exists(PT_PATH):
        print(f"[ERROR] PyTorch model not found at {PT_PATH}")
        sys.exit(1)
        
    try:
        model = YOLO(PT_PATH)

        print(f"[INFO] Exporting to ONNX with opset {OPSET}...")
        # Export with simplification ENABLED for better compatibility
        model.export(format="onnx", opset=OPSET, imgsz=IMG_SIZE, simplify=True)
        
        print("[INFO] ONNX export complete.")
        return True
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False

def convert_to_blob():
    """
    Converts the ONNX model to a .blob file with on-device YOLO decoding.
    """
    if not os.path.exists(ONNX_PATH):
        print(f"[ERROR] ONNX file not found at {ONNX_PATH}")
        return False

    output_dir = os.path.dirname(OUTPUT_BLOB_PATH)
    print("[INFO] Starting blob conversion with YOLO-specific parameters...")

    try:
        # For on-device decoding, YOLO parameters are passed via optimizer_params
        yolo_params = [
            f"--classes={NUM_CLASSES}",
            f"--iou_threshold={IOU_THRESHOLD}",
            f"--conf_threshold={CONF_THRESHOLD}",
            "--reverse_input_channels" # Common requirement for computer vision models
        ]

        blob_path = blobconverter.from_onnx(
            model=ONNX_PATH,
            output_dir=output_dir,
            data_type="FP16",
            shaves=6,           # Performance degrades beyond this
            version="2022.1",   # Faster performance with older version
            zoo_type="yolo",
            optimizer_params=yolo_params
        )
        
        # Rename the generated files to match MODEL_NAME
        base_path = os.path.join(output_dir, MODEL_NAME)
        json_path = f"{base_path}.json"
        
        os.rename(blob_path, f"{base_path}.blob")
        os.rename(blob_path.replace('.blob', '.json'), json_path)

        print(f"[SUCCESS] Blob created as: {base_path}.blob")
        
        # --- Add class names to the JSON file ---
        print(f"[INFO] Adding class names to {json_path}...")
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        config["mappings"] = {"labels": CLASS_NAMES}
        
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        print("[SUCCESS] JSON config updated with class names.")
        return True
    except Exception as e:
        print(f"[ERROR] Blob conversion failed: {e}")
        return False

def main():
    """Performs the complete conversion from PyTorch (.pt) to BLOB format."""
    print(f"[START] Beginning conversion for {MODEL_NAME}")
    
    if not export_to_onnx():
        print("[ERROR] Failed to convert PyTorch model to ONNX. Stopping.")
        sys.exit(1)
    
    if not convert_to_blob():
        print("[ERROR] Failed to convert ONNX to BLOB. Stopping.")
        sys.exit(1)
    
    print(f"[COMPLETE] Successfully converted {MODEL_NAME}.pt to BLOB format.")

if __name__ == "__main__":
    main()