import os
import sys
from ultralytics import YOLO
import blobconverter

# --- PATHS ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PT_PATH = os.path.join(MODEL_DIR, "YOLOv11n.pt")
ONNX_PATH = os.path.join(MODEL_DIR, "YOLOv11n.onnx")
OUTPUT_BLOB_NAME = "YOLOv11n"
OUTPUT_BLOB_PATH = os.path.join(MODEL_DIR, f"{OUTPUT_BLOB_NAME}.blob")

# --- MODEL CONFIG ---
IMG_SIZE = 640
OPSET = 11

def export_to_onnx():
    """Exports the PyTorch model to ONNX format."""
    print(f"[INFO] Loading model from {PT_PATH}...")
    
    if not os.path.exists(PT_PATH):
        print(f"[ERROR] PyTorch model not found at {PT_PATH}")
        sys.exit(1)
        
    try:
        model = YOLO(PT_PATH)

        print(f"[INFO] Exporting to ONNX with opset {OPSET}...")
        # Export with simplification disabled for maximum compatibility
        model.export(format="onnx", opset=OPSET, imgsz=IMG_SIZE, simplify=False)
        
        print("[INFO] ONNX export complete.")
        return True
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False

def convert_to_blob():
    """
    Converts the ONNX model to a .blob file using the modern blobconverter tool.
    This method is simpler and more reliable than using local OpenVINO tools.
    """
    if not os.path.exists(ONNX_PATH):
        print(f"[ERROR] ONNX file not found at {ONNX_PATH}")
        return False

    # Check if output blob already exists and remove it
    if os.path.exists(OUTPUT_BLOB_PATH):
        print(f"[INFO] Existing blob file found at {OUTPUT_BLOB_PATH}. Will be overwritten.")
        try:
            os.remove(OUTPUT_BLOB_PATH)
        except Exception as e:
            print(f"[WARNING] Could not remove existing blob file: {e}")

    print("[INFO] Starting blob conversion with blobconverter...")

    try:
        # Convert ONNX to blob
        blob_path = blobconverter.from_onnx(
            model=ONNX_PATH,
            output_dir=MODEL_DIR,
            data_type="FP16",
            shaves=6,
            optimizer_params=[
                "--scale_values", "[255,255,255]",
                "--input_shape", "[1,3,640,640]",
                "--layout", "BCHW",
                "--reverse_input_channels"
            ]
        )
        print(f"[INFO] Blob created at: {blob_path}")
        
        # Rename blob file
        os.rename(blob_path, OUTPUT_BLOB_PATH)
        print(f"[SUCCESS] Blob created as: {OUTPUT_BLOB_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Blob conversion failed: {e}")
        return False

def main():
    """Performs the complete conversion from PyTorch (.pt) to BLOB format."""
    print("[START] Beginning PyTorch to BLOB conversion process")
    
    # Step 1: Convert PT to ONNX
    if not export_to_onnx():
        print("[ERROR] Failed to convert PyTorch model to ONNX. Stopping.")
        sys.exit(1)
    
    # Step 2: Convert ONNX to BLOB
    if not convert_to_blob():
        print("[ERROR] Failed to convert ONNX to BLOB. Stopping.")
        sys.exit(1)
    
    print("[COMPLETE] Successfully converted PyTorch model to BLOB format")

if __name__ == "__main__":
    main()