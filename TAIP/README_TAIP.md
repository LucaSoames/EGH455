# TAIP Subsystem - Target Acquisition and Image Processing

## Overview

The TAIP (Target Acquisition and Image Processing) subsystem is a core component of the EGH455 UAVPayloadTAQ project. It provides real-time computer vision capabilities for pressure gauge reading, ArUco marker detection, and autonomous drilling control.

## Architecture

The system consists of six main modules:

### Core Modules

1. **`config.py`** - Central configuration management
2. **`data_models.py`** - Data structures and serialization
3. **`vision_processing.py`** - Computer vision algorithms
4. **`oak_camera.py`** - OAK-D Lite camera interface
5. **`gcs_client.py`** - Ground Control Station communication
6. **`main.py`** - Main application orchestrator

### Hardware Components

- **OAK-D Lite Camera**: Primary vision sensor with onboard Myriad X VPU
- **Raspberry Pi 5**: Main processing unit
- **Pimoroni Enviro+ HAT**: Environmental sensors and LCD display
- **GPIO Interface**: Drill trigger control

## Key Features

### Computer Vision
- ✅ Real-time YOLO object detection (YOLOv8n model)
- ✅ Pressure gauge reading with angle-to-pressure mapping
- ✅ ArUco marker detection and pose estimation
- ✅ Configurable confidence thresholds and filtering

### Hardware Integration
- ✅ Thread-safe camera interface with DepthAI pipeline
- ✅ Environmental sensor monitoring (temperature, pressure, humidity, light)
- ✅ LCD display with multiple viewing modes
- ✅ GPIO-controlled drill activation

### Communication
- ✅ Asynchronous GCS communication via HTTP POST
- ✅ JSON telemetry data transmission
- ✅ Compressed video frame streaming
- ✅ Retry logic and error handling

### System Management
- ✅ Comprehensive logging and error tracking
- ✅ Performance monitoring and statistics
- ✅ Graceful shutdown and resource cleanup
- ✅ Configuration validation

## Installation

### Prerequisites

1. **Hardware Setup**:
   - Connect OAK-D Lite via USB 3.0
   - Install Pimoroni Enviro+ HAT on GPIO header
   - Connect drill trigger to GPIO pin 18

2. **Software Dependencies**:
   ```bash
   # Navigate to TAIP directory
   cd /home/pi/EGH455/TAIP
   
   # Run the automated setup script
   ./setup_taip.sh
   
   # Or install manually:
   pip install -r ../requirements.txt
   
   # For DepthAI (if not already installed)
   pip install depthai>=2.18,<3
   ```

3. **Model Files**:
   - Ensure YOLOv8n.blob is present in `../models/blobs/`
   - Verify model configuration JSON exists at `../models/blobs/YOLOv8s.json`

### Verification

Run the installation verification script from the TAIP directory:
```bash
cd /home/pi/EGH455/TAIP
python3 verify_taip_installation.py
```

## Configuration

### Key Settings in `config.py`

```python
# Camera settings
CAMERA_PREVIEW_SIZE = (640, 640)
CAMERA_FPS = 30
CONFIDENCE_THRESHOLD = 0.5

# Gauge calibration
GAUGE_MIN_ANGLE_DEG = 225.0      # 0 bar
GAUGE_MAX_ANGLE_DEG = -45.0      # 10 bar
DRILL_PRESSURE_THRESHOLD = 2.0

# Network settings
GCS_URL = "http://192.168.1.100:5000"
REQUEST_TIMEOUT = 2.0

# GPIO settings
DRILL_GPIO_PIN = 18
```

### Calibration

The pressure gauge calibration maps needle angles to pressure values:
- **-45° to 225°** (270° total range)
- **10 bar to 0 bar** (decreasing with clockwise rotation)
- **Configurable** via `config.py` parameters

## Usage

### Basic Operation

1. **Start the system**:
   ```bash
   cd /home/pi/EGH455/TAIP
   python3 main.py
   ```

2. **Monitor via LCD**:
   - Use proximity sensor to cycle through display modes
   - Modes: IP Address → Live Feed → Sensor Data → System Status

3. **View logs**:
   ```bash
   tail -f logs/taip_system.log
   ```

### Display Modes

- **Mode 0 - IP Address**: Shows system IP and GCS URL
- **Mode 1 - Live Feed**: Real-time detections and gauge reading
- **Mode 2 - Sensor Data**: Environmental sensor readings
- **Mode 3 - System Status**: Camera, GCS, and performance status

### Drilling Control

The system automatically activates drilling when:
- Pressure gauge reading < 2.0 bar (configurable)
- Valid needle tip and gauge center detections exist
- GPIO pin 18 goes HIGH to trigger DE subsystem

## Data Flow

### Processing Pipeline

1. **Frame Capture**: OAK-D captures 640x640 frames at 30 FPS
2. **YOLO Inference**: Myriad X VPU runs YOLOv8n model
3. **Gauge Reading**: Calculate pressure from needle angle
4. **ArUco Detection**: Find and estimate marker poses
5. **Environmental Sensing**: Read Enviro+ sensors
6. **Data Packaging**: Create JSON payload
7. **GCS Transmission**: Send telemetry and video frame
8. **Display Update**: Update LCD based on mode
9. **Drill Control**: Activate/deactivate based on pressure

### JSON Telemetry Format

```json
{
  "timestamp": "YYYY-MM-DD HH:MM:SS.sss",
  "detections": [
    {
      "class": "valve_open", 
      "confidence": 0.89,
      "box": [x_min, y_min, x_max, y_max]
    }
  ],
  "aruco_markers": [
    {
      "id": 23,
      "position": [x, y, z],
      "orientation": [rvec_x, rvec_y, rvec_z]
    }
  ],
  "gauge_pressure_bar": 4.5,
  "environmental_data": {
    "temperature_c": 25.1,
    "pressure_hpa": 1012.5,
    "humidity_rh": 45.5,
    "light_lux": 350.0
  },
  "system_status": {
    "camera_connected": true,
    "model_loaded": true,
    "gcs_connection": true,
    "drill_active": false,
    "processing_time_ms": 45.2,
    "frames_processed": 1250
  }
}
```

## Performance

### Target Performance
- **Processing Rate**: 10 Hz (10 FPS)
- **Latency**: < 400ms per frame
- **GCS Transmission**: Within 4 seconds of capture
- **Resource Usage**: Optimized for Raspberry Pi 5

### Monitoring
- Real-time performance statistics in logs
- LCD display shows current FPS
- Processing time tracking per frame
- Memory and network usage monitoring

## Error Handling

### Robustness Features
- **Camera Disconnection**: Automatic reconnection attempts
- **Network Issues**: Retry logic with exponential backoff
- **Sensor Failures**: Graceful degradation with dummy data
- **Model Loading**: Fallback to backup models
- **GPIO Errors**: Safe drill control with error logging

### Recovery Mechanisms
- Thread-safe component isolation
- Automatic resource cleanup
- Graceful shutdown on system signals
- Configuration validation on startup

## Development

### Testing Individual Components

```bash
# Test camera interface
python3 oak_camera.py

# Test vision processing
python3 vision_processing.py

# Test GCS client
python3 gcs_client.py

# Test data models
python3 data_models.py

# Validate configuration
python3 -c "import config; config.validate_config()"
```

### Debug Mode

Enable debug features in `config.py`:
```python
DEBUG_MODE = True
SAVE_DEBUG_IMAGES = True
LOG_LEVEL = "DEBUG"
```

### Adding New Features

1. **New Detection Classes**: Update `YOLO_CLASSES` in `config.py`
2. **Additional Sensors**: Extend `EnvironmentalData` in `data_models.py`
3. **Custom Processing**: Add functions to `vision_processing.py`
4. **New Endpoints**: Modify `gcs_client.py` for additional GCS APIs

## Integration

### With Web Visualization (WI) Subsystem
- JSON telemetry received via HTTP POST at `/telemetry`
- Video frames received via HTTP POST at `/frame`
- Real-time status updates for monitoring

### With Drilling & Enclosure (DE) Subsystem
- GPIO pin 18 provides HIGH signal when drilling required
- Signal duration controlled by pressure readings
- Safe shutdown coordination

## Troubleshooting

### Common Issues

1. **Camera Not Detected**:
   - Check USB 3.0 connection
   - Verify DepthAI installation
   - Run: `python3 -c "import depthai; print(depthai.Device.getAllAvailableDevices())"`

2. **Model Loading Failed**:
   - Verify blob file exists and is valid
   - Check file permissions
   - Try backup model path

3. **GCS Connection Failed**:
   - Verify network connectivity
   - Check GCS URL in config
   - Test endpoints manually

4. **Enviro+ Not Working**:
   - Check HAT connection to GPIO
   - Verify I2C is enabled
   - Install latest enviroplus library

5. **Performance Issues**:
   - Monitor CPU usage
   - Check camera frame rate
   - Reduce processing resolution if needed

### Log Analysis

Check logs for detailed error information:
```bash
# View recent logs
tail -n 100 logs/taip_system.log

# Monitor real-time
tail -f logs/taip_system.log

# Search for errors
grep "ERROR" logs/taip_system.log
```

## Maintenance

### Regular Tasks
- Monitor log file sizes
- Check model performance metrics
- Verify GCS connectivity
- Validate gauge calibration
- Update configuration as needed

### Updates
- Keep DepthAI library updated
- Monitor for new YOLO model versions
- Update camera calibration parameters
- Refresh network configurations

## Support

For technical support or issues:
1. Check this README and troubleshooting section
2. Review system logs for error details
3. Verify hardware connections
4. Test individual components
5. Contact the development team with specific error messages

---

**Project**: EGH455 UAVPayloadTAQ  
**Subsystem**: TAIP (Target Acquisition and Image Processing)  
**Version**: 1.0  
**Last Updated**: September 2025

## Camera usage and ArUco pose
- Object detection: RGB camera (OAK-D Lite preview) feeds YOLO on-device.
- ArUco detection: LEFT mono camera (CAM_B) only in live mode for pose stability.
- Test mode (files): ArUco runs on RGB frames using RGB intrinsics.
- Visualisation: The main window shows the RGB frame with YOLO boxes. An inset at bottom-left shows the LEFT mono view with detected markers and pose axes drawn using LEFT intrinsics. This avoids axis scrambling from intrinsics mismatch.

## Non-blocking ArUco
ArUco detection and pose run in a background thread (ArucoWorker). The main loop never blocks on marker detection, preventing freezes.

## Configuration
- Removed camera switching flags for ArUco. LEFT mono is always used in live mode.
- Keep both CAMERA_MATRIX_LEFT and CAMERA_MATRIX_RGB for correct intrinsics in live/test modes.

## GCS Server Setup

The Ground Control Station (GCS) server receives telemetry and video from the Pi and serves the web interface.

### Running on GCS Laptop

1. **Copy required files to laptop**:
   ```bash
   # On laptop
   scp -r pi@<pi-ip>:/home/pi/EGH455/TAIP/gcs_server.py .
   scp -r pi@<pi-ip>:/home/pi/EGH455/frontend ./
   ```

2. **Install dependencies**:
   ```bash
   pip install flask flask-socketio flask-cors
   ```

3. **Start GCS server**:
   ```bash
   python3 gcs_server.py
   ```

4. **Update Pi config**:
   Edit `/home/pi/EGH455/TAIP/config.py` and set:
   ```python
   GCS_LAPTOP_IP = "<your-laptop-ip>"
   ```

### Running GCS Server on Pi (Testing)

If you want to run everything on the Pi for testing:

1. **Start GCS server** (in one terminal):
   ```bash
   cd /home/pi/EGH455/TAIP
   python3 gcs_server.py
   ```

2. **Update config** to use localhost:
   ```python
   GCS_LAPTOP_IP = "127.0.0.1"
   ```

3. **Start TAIP system** (in another terminal):
   ```bash
   cd /home/pi/EGH455/TAIP
   python3 main.py
   ```

4. **Access web interface**:
   Open browser to `http://<pi-ip>:5000`
