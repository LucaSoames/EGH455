# TAIP Subsystem - Drone Payload Control System

## Overview

The TAIP (Telemetry, Acquisition, Inference, Processing) subsystem is an autonomous drone payload system designed for environmental monitoring, visual inspection, and automated drilling operations. The system integrates computer vision, environmental sensing, and actuation capabilities with real-time telemetry to a Ground Control Station (GCS).

## Hardware Components

### Primary Components
- **Raspberry Pi 5**: Main processing unit and system orchestrator
- **Luxonis OAK-D Lite**: Stereo camera with onboard Intel Myriad X VPU for AI inference
  - RGB camera (640×640 @ 10 FPS for YOLO inference)
  - LEFT mono camera (for ArUco pose estimation)
- **Pimoroni Enviro+ HAT**: Environmental sensor suite with display
  - BME280: Temperature, humidity, pressure sensor
  - LTR559: Light and proximity sensor
  - ST7735 LCD: 160×80 pixel color display
- **Servo Motor**: PWM-controlled drill actuator (GPIO 18)

### Wiring & Connections
- Enviro+ HAT mounted on Pi GPIO header
- OAK-D Lite connected via USB 3.0
- Servo motor connected to GPIO 18 (PWM) and appropriate power supply
- Network connection via Wi-Fi or Ethernet (10.88.52.93 subnet)

## System Architecture

### Main Application (`main.py`)

The `MainApp` class orchestrates all subsystem components:

```
┌───────────────────────────────────────────────────────┐
│                      MainApp                          │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Input Source│  │Vision Process│  │Hardware I/O  │  │
│  │             │  │              │  │              │  │
│  │ • OakCamera │  │ • YOLO (OAK) │  │ • GCS Client │  │
│  │ • FileProc  │  │ • ArUco Pose │  │ • Env Sensors│  │
│  │             │  │ • Gauge Calc │  │ • LCD Display│  │
│  │             │  │              │  │ • Drill Ctrl │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────────────────────────────────────────┘
```

### Module Breakdown

#### 1. Input Pipeline (Dual-Mode Operation)

**Live Camera Mode** (`oak_camera.py`)
```python
# config.py
INPUT_PATH = None  # Activates live camera mode
```
- OAK-D Lite runs YOLOv8n inference on Myriad X VPU
- Device-side decoding reduces Pi CPU load
- LEFT mono camera stream for ArUco detection

**File Testing Mode** (`file_processing.py`)
```python
# config.py
INPUT_PATH = Path("models/testing/videos/test.mp4")  # Video mode
INPUT_PATH = Path("models/testing/images")           # Image mode
```
- Video: Auto-advance frames at specified FPS
- Images: Manual advance with keypress
- Uses same YOLO blob as live mode for consistency

#### 2. Vision Processing (`vision_processing.py`)

**YOLO Object Detection**
- YOLOv8n model trained for gauge needle and clock face detection
- Classes: `needle`, `clock_face`
- Runs onboard OAK-D Lite at 10 FPS
- Output: `YoloDetection` objects with bounding boxes and confidence

**ArUco Marker Detection & Pose Estimation**
- Uses `ArucoWorker` background thread for non-blocking operation
- Dictionary: `cv2.aruco.DICT_4X4_50`
- Marker size: 0.15 m (configurable in `config.py`)
- Outputs 6DOF pose (translation + rotation vectors)
- Runs up to 30 Hz (configurable: `max_hz` parameter)

**Gauge Pressure Calculation**
```python
gauge_reading = calculate_gauge_reading(yolo_detections)
```
- Calculates needle angle relative to clock face center
- Maps angle to pressure range (0-60 bar default)
- Handles edge cases (no detections, multiple detections)

#### 3. Hardware Integration

**Environmental Sensors** (`enviro_lcd.py`)
```python
env_sensors = EnvironmentalSensors()
env_data = env_sensors.get_readings()  # Returns EnvironmentalData
```
- Temperature (°C)
- Humidity (%)
- Atmospheric pressure (hPa)
- Light level (lux)
- Proximity (0-65535, for LCD mode switching)

**LCD Display** (`enviro_lcd.py`)

Four display modes cycled by proximity sensor:
1. **System Status**: IP address, timestamp, connection status
2. **Camera Preview**: Downscaled RGB frame with detection count
3. **Environmental Data**: Sensor readings in text format
4. **Detection Stats**: YOLO/ArUco counts, gauge pressure

**Drill Controller** (`drilling.py`)
```python
drill_controller = DrillController()
drill_controller.control_drill(gauge_reading)
```
- Activates when gauge pressure < `DRILL_PRESSURE_THRESHOLD` (default: 20 bar)
- Servo position: `DRILL_SERVO_ACTIVE_POSITION` (default: 45°)
- Auto-resets when pressure recovers above threshold + 2 bar margin

#### 4. Ground Control Station Communication (`gcs_client.py`)

**Telemetry Transmission** (5 Hz default)
```python
POST http://10.88.52.93:5000/data
Content-Type: application/json

{
  "timestamp": "2025-10-04T10:30:45.123456",
  "yolo_detections": [...],
  "aruco_markers": [...],
  "gauge_pressure_bar": 35.2,
  "environmental_data": {...}
}
```

**Video Frame Transmission** (10 FPS default)
```python
POST http://10.88.52.93:5000/frame
Content-Type: image/jpeg

<JPEG binary data>
```

Both endpoints use async requests with retry logic and error handling.

## Data Models (`data_models.py`)

```python
@dataclass
class YoloDetection:
    label: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

@dataclass
class ArucoDetection:
    marker_id: int
    tvec: List[float]      # Translation vector [x, y, z] in meters
    rvec: List[float]      # Rotation vector
    distance_m: float
    corners: List[Tuple[float, float]]

@dataclass
class EnvironmentalData:
    temperature: Optional[float]
    humidity: Optional[float]
    pressure: Optional[float]
    light: Optional[float]
    proximity: Optional[int]

@dataclass
class PayloadData:
    timestamp: str
    yolo_detections: List[YoloDetection]
    aruco_markers: List[ArucoDetection]
    gauge_pressure_bar: Optional[float]
    environmental_data: EnvironmentalData
```

## Configuration (`config.py`)

Key configuration parameters:

```python
# Input mode
INPUT_PATH = None  # None for live camera, Path for test files

# Camera intrinsics (from calibration)
CAMERA_MATRIX_RGB = np.array([...])
DISTORTION_COEFFS_RGB = np.array([...])
CAMERA_MATRIX_LEFT = np.array([...])
DISTORTION_COEFFS_LEFT = np.array([...])

# YOLO model
BLOB_PATH = PROJECT_ROOT / "models/yolov8n.blob"

# ArUco parameters
ARUCO_MARKER_SIZE_M = 0.15  # Marker size in meters
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Gauge parameters
GAUGE_ANGLE_MIN = -120  # Degrees
GAUGE_ANGLE_MAX = 120
GAUGE_PRESSURE_MIN = 0  # bar
GAUGE_PRESSURE_MAX = 60

# Drill control
DRILL_PRESSURE_THRESHOLD = 20.0  # bar
DRILL_SERVO_PIN = 18
DRILL_SERVO_ACTIVE_POSITION = 45  # degrees

# GCS communication
GCS_URL = "http://10.88.52.93:5000"
POST_TELEM_HZ = 5
POST_FRAME_FPS = 10

# Visualization
SHOW_LIVE_VISUALISATION = True
```

## Operation Modes

### Live Camera Mode

```bash
# Set in config.py
INPUT_PATH = None

# Run
python3 main.py
```

**Behavior:**
- OAK-D Lite streams live RGB and mono frames
- YOLO inference runs on device at 10 FPS
- ArUco detection uses LEFT mono camera (better for pose estimation)
- Continuous telemetry to GCS
- Real-time LCD updates
- Drill activates based on gauge readings

### Video Testing Mode

```bash
# Set in config.py
INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_blue_B.mp4"

# Run
python3 main.py
```

**Behavior:**
- Frames auto-advance at video's native FPS
- YOLO inference runs on each frame (device-side)
- ArUco detection uses RGB frames (RGB intrinsics)
- Press 'q' or ESC to quit
- No GCS transmission (configurable)

### Image Testing Mode

```bash
# Set in config.py
INPUT_PATH = PROJECT_ROOT / "models/testing/images"

# Run
python3 main.py
```

**Behavior:**
- Manual frame advance (press any key except 'q'/ESC)
- Allows detailed inspection of each detection
- Press 'q' or ESC to quit

## Visualization

When `SHOW_LIVE_VISUALISATION = True`, displays:

```
┌─────────────────────────────────────────┐
│         Main RGB Frame                  │
│  • YOLO bounding boxes (green/red)      │
│  • Gauge needle angle line              │
│  • Pressure reading text                │
│  • ArUco marker axes (RGB = XYZ)        │
│  • Detection counts                     │
│                                         │
│  ┌─────────────┐                        │
│  │  ArUco      │  ← Inset shows mono    │
│  │  Detection  │    frame with markers  │
│  └─────────────┘                        │
└─────────────────────────────────────────┘
```

## Main Processing Loop

```python
while running:
    # 1. Get frame from input source (camera or file)
    rgb_frame, mono_frame, yolo_detections = _get_frame_and_detections()
    
    # 2. Non-blocking ArUco detection (background thread)
    aruco_worker.update_frame(mono_frame or rgb_frame)
    aruco_detections, corners, ids, vis = aruco_worker.get_latest()
    
    # 3. Calculate gauge pressure from YOLO detections
    gauge_reading = calculate_gauge_reading(yolo_detections)
    
    # 4. Read environmental sensors
    env_data = env_sensors.get_readings()
    
    # 5. Control drill based on pressure
    drill_controller.control_drill(gauge_reading)
    
    # 6. Send telemetry and frames to GCS (rate-limited)
    _handle_gcs_communication(rgb_frame, yolo_detections, 
                              aruco_detections, gauge_reading, env_data)
    
    # 7. Update LCD display
    proximity = env_sensors.get_proximity()
    lcd_display.update_mode(proximity)
    lcd_display.update_display(ip_address, rgb_frame, yolo_detections,
                               env_data, gauge_reading)
    
    # 8. Show visualization (if enabled)
    if SHOW_LIVE_VISUALISATION:
        show_inference_visualisation(rgb_frame, yolo_detections, 
                                    aruco_detections, gauge_reading)
```

## Key Features

### Non-Blocking ArUco Detection
The `ArucoWorker` class runs in a background thread to prevent main loop blocking:
```python
aruco_worker = ArucoWorker(camera_matrix=K, dist_coeffs=D, 
                          marker_size_m=0.15, max_hz=30.0)
aruco_worker.start()

# In main loop (non-blocking)
aruco_worker.update_frame(frame)
detections, corners, ids, vis = aruco_worker.get_latest()
```

### Drill Auto-Reset
Drill resets when pressure recovers above threshold + 2 bar margin:
```python
if (drill_controller.drilling_complete and 
    gauge_reading >= DRILL_PRESSURE_THRESHOLD + 2.0):
    drill_controller.reset_drill_state()
```

### GCS Rate Limiting
Telemetry and frames sent at controlled rates to prevent network saturation:
```python
# Telemetry: 5 Hz (200ms interval)
if (now - last_telem_time) >= (1.0 / POST_TELEM_HZ):
    gcs_client.send_data(payload)

# Frames: 10 FPS (100ms interval)
if (now - last_frame_time) >= (1.0 / POST_FRAME_FPS):
    gcs_client.send_frame(rgb_frame)
```

### Camera Intrinsics Selection
Automatically selects correct intrinsics based on mode:
```python
# Live mode: ArUco uses LEFT mono → LEFT intrinsics
if not file_processor:
    K = config.CAMERA_MATRIX_LEFT
    D = config.DISTORTION_COEFFS_LEFT
else:
    # Test mode: ArUco uses RGB → RGB intrinsics
    K = config.CAMERA_MATRIX_RGB
    D = config.DISTORTION_COEFFS_RGB
```

## Error Handling

### Graceful Degradation
- Missing frames → continue with last valid data
- ArUco detection failure → continue without pose data
- GCS connection failure → local operation continues with retries
- Sensor read errors → use cached values or None

### Resource Cleanup
```python
try:
    app.setup()
    app.run_loop()
except KeyboardInterrupt:
    print("Application interrupted by user")
except Exception as e:
    print(f"FATAL ERROR: {e}")
    traceback.print_exc()
finally:
    app.shutdown()  # Guaranteed cleanup
```

The `shutdown()` method ensures:
1. OpenCV windows closed
2. ArUco worker thread stopped
3. Camera pipeline released
4. GCS connections closed
5. Drill servo disarmed
6. LCD display cleared

## Performance Characteristics

- **YOLO Inference**: ~10 FPS on OAK-D Lite Myriad X VPU
- **ArUco Detection**: Up to 30 Hz (configurable, typically 10-15 Hz actual)
- **Main Loop**: 100 Hz (10ms sleep), effectively limited by camera FPS
- **GCS Telemetry**: 5 Hz
- **GCS Frames**: 10 FPS
- **LCD Updates**: Every main loop iteration (~10-30 Hz)
- **Environmental Sensors**: Polled every main loop iteration

## Network Requirements

- Local network connection (Wi-Fi or Ethernet)
- Subnet: 10.88.52.x
- GCS server at: 10.88.52.93:5000
- Endpoints: `/data` (telemetry), `/frame` (video)
- Outbound HTTP POST requests only

## Troubleshooting

### Camera Not Detected
```bash
# Check USB connection
lsusb | grep 03e7  # Luxonis vendor ID

# Check depthai installation
python3 -c "import depthai; print(depthai.__version__)"
```

### GCS Connection Failure
- Verify GCS server is running: `curl http://10.88.52.93:5000/`
- Check network connectivity: `ping 10.88.52.93`
- Review `config.GCS_URL` setting

### LCD Not Displaying
- Check I2C connection: `sudo i2cdetect -y 1`
- Verify Enviro+ HAT is seated properly
- Check for conflicting I2C devices

### Drill Not Activating
- Verify GPIO 18 is not in use: `gpio readall`
- Check servo power supply (separate from Pi power)
- Review `config.DRILL_PRESSURE_THRESHOLD` setting
- Monitor gauge readings in visualization

### Low Frame Rate
- Reduce `POST_FRAME_FPS` in config
- Disable `SHOW_LIVE_VISUALISATION`
- Check CPU usage: `top` or `htop`
- Verify OAK-D Lite USB 3.0 connection (not USB 2.0)

---

**Last Updated**: October 4, 2025  
**System Version**: 1.0  
**Hardware**: Raspberry Pi 5 + OAK-D Lite + Enviro+ HAT