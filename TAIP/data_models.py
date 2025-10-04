# /home/pi/EGH455/TAIP/data_models.py

"""
Data Models for the EGH455 TAIP Subsystem

This file defines the data structures used throughout the application,
particularly for packaging data to be sent to the Ground Control Station (GCS).
Using dataclasses ensures a consistent, predictable, and self-documenting
data format.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class YoloDetection:
    """Represents a single object detected by the YOLOv8 model."""
    class_name: str
    confidence: float
    # Bounding box in [xmin, ymin, xmax, ymax] format with relative coordinates (0.0 to 1.0)
    box: Tuple[float, float, float, float]

@dataclass
class ArucoDetection:
    """Represents a single detected ArUco marker and its pose."""
    marker_id: int
    # Translation vector [x, y, z] in meters from the camera
    position: Tuple[float, float, float]
    # Rotation vector (Rodrigues notation)
    orientation: Tuple[float, float, float]

@dataclass
class GasReadings:
    """Stores gas sensor readings from the MICS6814 sensor on Enviro+ board."""
    reducing_ohms: float      # Sensitive to CO, H2S, NH3
    oxidising_ohms: float     # Sensitive to NO2, NO, O3
    nh3_ohms: float          # Sensitive to NH3, H2, ethanol

@dataclass
class EnvironmentalData:
    """Stores sensor readings from the Enviro+ board."""
    temperature_c: float
    pressure_hpa: float
    humidity_rh: float
    light_lux: float
    gas_readings: Optional[GasReadings] = None

@dataclass
class PayloadData:
    """
    The main data packet to be serialized into JSON and sent to the GCS.
    This structure aggregates all sensor and vision processing outputs.
    """
    timestamp: str
    yolo_detections: List[YoloDetection] = field(default_factory=list)
    aruco_markers: List[ArucoDetection] = field(default_factory=list)
    gauge_pressure_bar: Optional[float] = None
    environmental_data: Optional[EnvironmentalData] = None