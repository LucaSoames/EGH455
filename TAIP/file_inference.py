"""
Host file inference pipeline using OAK-D for YOLO device decoding.
Feeds prerecorded frames (images/videos) to the Myriad X while keeping
the same blob + decoding path as live mode.
"""

import depthai as dai
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List

import config
from data_models import YoloDetection

class FileInferenceProcessor:
    """Pushes host frames into a DepthAI pipeline for on-device YOLO inference (files/images)."""

    def __init__(self):
        self._load_model_config()
        self.pipeline = self._create_pipeline()
        try:
            self.device = dai.Device(self.pipeline)
        except Exception as e:
            raise RuntimeError(f"Failed to initialise DepthAI device for file inference: {e}")
        self.q_in = self.device.getInputQueue("host_in")
        self.q_nn = self.device.getOutputQueue("nn_out", maxSize=2, blocking=True)

    def _load_model_config(self):
        with open(config.CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        self.class_names = cfg["mappings"]["labels"]
        self.model_input = tuple(map(int, cfg["nn_config"]["input_size"].split('x')))  # (W, H)
        self.nn_meta = cfg["nn_config"]["NN_specific_metadata"]

    def _create_pipeline(self):
         p = dai.Pipeline()
         xin = p.createXLinkIn()
         nn  = p.createYoloDetectionNetwork()
         xout = p.createXLinkOut()

         xin.setStreamName("host_in")
         xout.setStreamName("nn_out")

         # Use on-device decoding via the Myriad X
         nn.setBlobPath(str(config.BLOB_PATH))
         nn.setConfidenceThreshold(config.CONFIDENCE_THRESHOLD)
         nn.setIouThreshold(config.IOU_THRESHOLD)
         nn.setNumClasses(self.nn_meta["classes"])
         nn.setCoordinateSize(self.nn_meta["coordinates"])
         nn.setNumInferenceThreads(2)
         nn.input.setBlocking(False)

         xin.out.link(nn.input)
         nn.out.link(xout.input)
         return p

    def run_inference(self, frame_bgr) -> List[YoloDetection]:
        # Resize on host to model input size (matches training/export)
        resized = cv2.resize(frame_bgr, self.model_input)
        planar = resized.transpose(2, 0, 1).flatten()

        img = dai.ImgFrame()
        img.setType(dai.RawImgFrame.Type.BGR888p)
        img.setData(planar)
        img.setWidth(self.model_input[0])
        img.setHeight(self.model_input[1])
        self.q_in.send(img)

        # Block until results ready (avoids empty detections)
        nn_packet = self.q_nn.get()
        dets: List[YoloDetection] = []
        for d in nn_packet.detections:
            if 0 <= d.label < len(self.class_names):
                dets.append(
                    YoloDetection(
                        class_name=self.class_names[d.label],
                        confidence=d.confidence,
                        box=(d.xmin, d.ymin, d.xmax, d.ymax)
                    )
                )
        return dets

    def close(self):
        try:
            if hasattr(self, "device"):
                self.device.close()
        except:
            pass