"""
Data models for the TAIP subsystem using Python dataclasses.
Defines structured data formats for detections, sensor readings, and GCS communication.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import json


@dataclass
class BoundingBox:
    """Represents a bounding box with normalized coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if not (0.0 <= self.x_min <= 1.0 and 0.0 <= self.y_min <= 1.0 and
                0.0 <= self.x_max <= 1.0 and 0.0 <= self.y_max <= 1.0):
            raise ValueError("Bounding box coordinates must be normalized (0.0 - 1.0)")
        
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Invalid bounding box: min coordinates must be less than max")
    
    def to_absolute(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute pixel coordinates."""
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height)
        )
    
    def to_list(self) -> List[float]:
        """Convert to list format for JSON serialization."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass
class YoloDetection:
    """Represents a single YOLO object detection."""
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    
    def __post_init__(self):
        """Validate detection parameters."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if not self.class_name:
            raise ValueError("Class name cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 3),
            "box": self.bounding_box.to_list()
        }


@dataclass
class ArucoDetection:
    """Represents a detected ArUco marker with pose information."""
    marker_id: int
    position: Tuple[float, float, float]  # Translation vector (x, y, z) in meters
    orientation: Tuple[float, float, float]  # Rotation vector (rvec_x, rvec_y, rvec_z)
    corners: Optional[List[Tuple[float, float]]] = None  # Corner points in image coordinates
    
    def __post_init__(self):
        """Validate ArUco detection parameters."""
        if self.marker_id < 0:
            raise ValueError("Marker ID must be non-negative")
        
        if len(self.position) != 3:
            raise ValueError("Position must be a 3D vector (x, y, z)")
        
        if len(self.orientation) != 3:
            raise ValueError("Orientation must be a 3D rotation vector")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "id": self.marker_id,
            "position": [round(pos, 4) for pos in self.position],
            "orientation": [round(rot, 4) for rot in self.orientation]
        }


@dataclass
class EnvironmentalData:
    """Environmental sensor readings from the Enviro+ HAT."""
    temperature_c: float
    pressure_hpa: float
    humidity_rh: float
    light_lux: float
    
    def __post_init__(self):
        """Validate environmental data ranges."""
        if not (-50.0 <= self.temperature_c <= 100.0):
            raise ValueError("Temperature out of reasonable range (-50°C to 100°C)")
        
        if not (800.0 <= self.pressure_hpa <= 1200.0):
            raise ValueError("Pressure out of reasonable range (800-1200 hPa)")
        
        if not (0.0 <= self.humidity_rh <= 100.0):
            raise ValueError("Humidity must be between 0% and 100%")
        
        if self.light_lux < 0.0:
            raise ValueError("Light level cannot be negative")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "temperature_c": round(self.temperature_c, 1),
            "pressure_hpa": round(self.pressure_hpa, 1),
            "humidity_rh": round(self.humidity_rh, 1),
            "light_lux": round(self.light_lux, 0)
        }


@dataclass
class SystemStatus:
    """System status information for monitoring and debugging."""
    camera_connected: bool
    model_loaded: bool
    gcs_connection: bool
    drill_active: bool
    processing_time_ms: float
    frames_processed: int
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "camera_connected": self.camera_connected,
            "model_loaded": self.model_loaded,
            "gcs_connection": self.gcs_connection,
            "drill_active": self.drill_active,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "frames_processed": self.frames_processed,
            "last_error": self.last_error
        }


@dataclass
class PayloadData:
    """Main data structure containing all information to be sent to GCS."""
    timestamp: str
    detections: List[YoloDetection]
    aruco_markers: List[ArucoDetection]
    gauge_pressure_bar: Optional[float]
    environmental_data: EnvironmentalData
    system_status: Optional[SystemStatus] = None
    
    @classmethod
    def create_timestamp(cls) -> str:
        """Create a properly formatted timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    @classmethod
    def create_empty(cls, environmental_data: EnvironmentalData) -> 'PayloadData':
        """Create an empty payload with only environmental data."""
        return cls(
            timestamp=cls.create_timestamp(),
            detections=[],
            aruco_markers=[],
            gauge_pressure_bar=None,
            environmental_data=environmental_data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        data = {
            "timestamp": self.timestamp,
            "detections": [detection.to_dict() for detection in self.detections],
            "aruco_markers": [marker.to_dict() for marker in self.aruco_markers],
            "gauge_pressure_bar": round(self.gauge_pressure_bar, 2) if self.gauge_pressure_bar is not None else None,
            "environmental_data": self.environmental_data.to_dict()
        }
        
        if self.system_status is not None:
            data["system_status"] = self.system_status.to_dict()
        
        return data
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def validate(self) -> bool:
        """Validate the payload data structure."""
        try:
            # Check timestamp format
            datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S.%f")
            
            # Validate all detections
            for detection in self.detections:
                if not isinstance(detection, YoloDetection):
                    raise ValueError("Invalid detection type")
            
            # Validate all ArUco markers
            for marker in self.aruco_markers:
                if not isinstance(marker, ArucoDetection):
                    raise ValueError("Invalid ArUco marker type")
            
            # Validate gauge pressure if present
            if self.gauge_pressure_bar is not None:
                if not (0.0 <= self.gauge_pressure_bar <= 20.0):
                    raise ValueError("Gauge pressure out of reasonable range")
            
            # Validate environmental data
            if not isinstance(self.environmental_data, EnvironmentalData):
                raise ValueError("Invalid environmental data type")
            
            return True
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Payload validation failed: {e}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_bounding_box_from_yolo(x_center: float, y_center: float, 
                                  width: float, height: float) -> BoundingBox:
    """
    Create a BoundingBox from YOLO format (center_x, center_y, width, height).
    All values should be normalized (0.0 - 1.0).
    """
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    
    # Clamp to valid range
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))
    
    return BoundingBox(x_min, y_min, x_max, y_max)


def parse_yolo_detection(detection_data: Dict[str, Any], 
                        class_names: Dict[int, str]) -> Optional[YoloDetection]:
    """
    Parse raw YOLO detection data into a YoloDetection object.
    
    Args:
        detection_data: Raw detection dictionary from DepthAI
        class_names: Mapping from class IDs to class names
    
    Returns:
        YoloDetection object or None if parsing fails
    """
    try:
        class_id = int(detection_data.get('label', -1))
        confidence = float(detection_data.get('confidence', 0.0))
        
        # Get bounding box coordinates (assuming they're normalized)
        x_min = float(detection_data.get('xmin', 0.0))
        y_min = float(detection_data.get('ymin', 0.0))
        x_max = float(detection_data.get('xmax', 1.0))
        y_max = float(detection_data.get('ymax', 1.0))
        
        # Get class name
        class_name = class_names.get(class_id, f"unknown_{class_id}")
        
        # Create bounding box
        bbox = BoundingBox(x_min, y_min, x_max, y_max)
        
        return YoloDetection(class_name, confidence, bbox)
        
    except (KeyError, ValueError, TypeError) as e:
        print(f"Warning: Failed to parse YOLO detection: {e}")
        return None


def create_test_payload() -> PayloadData:
    """Create a test payload for development and debugging."""
    env_data = EnvironmentalData(
        temperature_c=25.1,
        pressure_hpa=1012.5,
        humidity_rh=45.5,
        light_lux=350.0
    )
    
    test_bbox = BoundingBox(0.3, 0.4, 0.7, 0.8)
    test_detection = YoloDetection("Valve_Open", 0.89, test_bbox)
    
    test_marker = ArucoDetection(
        marker_id=23,
        position=(0.1, 0.05, 0.2),
        orientation=(0.1, 0.2, 0.3)
    )
    
    system_status = SystemStatus(
        camera_connected=True,
        model_loaded=True,
        gcs_connection=True,
        drill_active=False,
        processing_time_ms=45.2,
        frames_processed=1250
    )
    
    return PayloadData(
        timestamp=PayloadData.create_timestamp(),
        detections=[test_detection],
        aruco_markers=[test_marker],
        gauge_pressure_bar=4.5,
        environmental_data=env_data,
        system_status=system_status
    )


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the data models
    print("Testing data models...")
    
    try:
        test_payload = create_test_payload()
        test_payload.validate()
        
        print("✓ PayloadData validation passed")
        print("✓ JSON serialization test:")
        print(test_payload.to_json(indent=2))
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
