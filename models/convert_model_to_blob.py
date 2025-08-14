import os
from ultralytics import YOLO
from modelconverter.hub import convert
import sys
from dotenv import load_dotenv
import shutil # Import the shutil module

# Load environment variables from .env file
load_dotenv()

# Paths
PT_PATH = "models/YOLOv11n.pt"
ONNX_PATH = "models/YOLOv11n.onnx"
BLOB_PATH = "models/YOLOv11n.blob"
IMG_SIZE = 640
OPSET = 17

HUBAI_API_KEY = os.getenv("HUBAI_API_KEY")

def export_to_onnx():
    """Exports the PyTorch model to ONNX format."""
    print(f"[INFO] Loading model from {PT_PATH}...")
    model = YOLO(PT_PATH)

    # Ensure the target ONNX gets overwritten
    if os.path.exists(ONNX_PATH):
        print(f"[INFO] Removing existing ONNX: {ONNX_PATH}")
        os.remove(ONNX_PATH)

    print("[INFO] Exporting to ONNX...")
    
    model.export(
        format="onnx",
        opset=OPSET,
        imgsz=IMG_SIZE,
    )

    # Ensure the exported file ends up at ONNX_PATH
    produced = PT_PATH.rsplit(".", 1)[0] + ".onnx"
    if os.path.exists(produced) and produced != ONNX_PATH and not os.path.exists(ONNX_PATH):
        os.replace(produced, ONNX_PATH)

    if os.path.exists(ONNX_PATH):
        print(f"[INFO] ONNX model saved to {ONNX_PATH}")
    else:
        print(f"[ERROR] ONNX export failed. Could not find {ONNX_PATH}")
        sys.exit(1)

def convert_to_blob():
    """Converts the ONNX model to a .blob file using the HubAI online service."""
    if not HUBAI_API_KEY:
        print("[ERROR] HUBAI_API_KEY not found. Please set it as an environment variable.")
        sys.exit(1)

    print("[INFO] Converting ONNX to .blob via HubAI (RVC2)...")
    temp_dir = None
    try:
        blob = convert.RVC2(
            path=ONNX_PATH,
            api_key=HUBAI_API_KEY,
            target_precision="FP16",
            tool_version="2022.3.0",
            number_of_shaves=4,  # Adjust based on device capability (drop to 4 if needed)
        )
        temp_dir = os.path.dirname(blob)

        # If the destination blob already exists, remove it first.
        if os.path.exists(BLOB_PATH):
            os.remove(BLOB_PATH)
        os.rename(blob, BLOB_PATH)
        print(f"[INFO] Blob ready: {BLOB_PATH}")
    except Exception as e:
        print(f"[ERROR] Blob conversion failed: {e}")
        sys.exit(1)
    finally:
        # Clean up the temporary directory after we are done
        if temp_dir and os.path.isdir(temp_dir):
            print(f"[INFO] Removing temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    export_to_onnx()
    convert_to_blob()