import torch
from ultralytics import YOLO
import blobconverter
import os

"""
Export YOLOv11n model weights (.pt) to ONNX (.onnx) 
then convert to OpenVINO blob for OAK-D deployment.
"""

# Hardcoded paths
PT_PATH = "models/YOLOv11n.pt"
ONNX_PATH = "models/YOLOv11n.onnx"
BLOB_PATH = "models/YOLOv11n.blob"
IMG_SIZE = 640
OPSET = 11

def export_to_onnx():
    print(f"Loading model from {PT_PATH}...")
    model = YOLO(PT_PATH)
    print("Model loaded successfully.")
    
    print("Exporting model to ONNX...")
    model.export(format="onnx", opset=OPSET, imgsz=IMG_SIZE)
    if not os.path.exists(ONNX_PATH):
        # YOLO saves with same basename as .pt
        default_path = PT_PATH.replace(".pt", ".onnx")
        os.rename(default_path, ONNX_PATH)
    print(f"ONNX export complete: {ONNX_PATH}")

def convert_to_blob():
    print("Converting ONNX to blob using blobconverter...")
    blob_path = blobconverter.from_onnx(
        model=ONNX_PATH,
        data_type="FP16",
        shaves=6,
        output_dir="models",
        compile_params={"MYRIAD_DETECT_NETWORK": "true"},
    )
    os.rename(blob_path, BLOB_PATH)
    print(f"Blob conversion complete: {BLOB_PATH}")

if __name__ == "__main__":
    export_to_onnx()
    convert_to_blob()
