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
    label: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

@dataclass
class ArucoDetection:
    """Represents a single detected ArUco marker and its pose."""
    marker_id: int
    tvec: List[float]
    rvec: List[float]
    distance_m: float
    corners: List[Tuple[float, float]]

@dataclass
class GasReadings:
    """Stores gas sensor readings from the MICS6814 sensor on Enviro+ board."""
    reducing_ohms: float      # Sensitive to CO, H2S, NH3
    oxidising_ohms: float     # Sensitive to NO2, NO, O3
    nh3_ohms: float          # Sensitive to NH3, H2, ethanol
    # PPM values (calibrated)
    reducing_ppm: Optional[float] = None  # CO estimate
    oxidising_ppm: Optional[float] = None  # NO2 estimate
    nh3_ppm: Optional[float] = None       # NH3 estimate

@dataclass
class EnvironmentalData:
    """Stores sensor readings from the Enviro+ board."""
    temperature_c: float
    pressure_hpa: float
    humidity_rh: float
    light_lux: float
    pi_temperature_c: Optional[float] = None  # Raspberry Pi CPU temperature
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