# TAIP Test Mode Documentation

## Overview
The TAIP (Target Acquisition and Image Processing) system includes comprehensive test mode functionality that allows you to process videos and images instead of using the live camera feed. Test mode is controlled simply by editing the `TEST_INPUT_PATH` in `config.py` - no command line arguments needed!

## Simple Usage

### How to Switch Modes
Simply edit `/home/pi/EGH455/TAIP/config.py` and modify the `TEST_INPUT_PATH` setting:

#### 1. Live Camera Mode (Default)
```python
INPUT_PATH = None
```

#### 2. Test with Image Directory
```python
INPUT_PATH = PROJECT_ROOT / "models/testing/images"
```

#### 3. Test with Video File
```python
INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_blue_A.mp4"
```

### Running the System
After editing config.py, simply run:
```bash
cd /home/pi/EGH455/TAIP
/home/pi/venvs/depthai_env/bin/python main.py
```

The system automatically detects whether to run in live mode or test mode based on the config setting.

## Configuration

### Available Test Videos
The system includes several pre-recorded test videos in `/home/pi/EGH455/models/testing/videos/`:
- `far_blue.mp4`
- `far_silver_A.mp4` 
- `far_silver_B.mp4`
- `near_blue_A.mp4`
- `near_blue_B.mp4`
- `near_silver_A.mp4`
- `near_silver_B.mp4`
- `near_silver_C.mp4`

### Easy Mode Switching in config.py
The configuration file provides commented examples for easy switching:

```python
# Live camera mode (default)
INPUT_PATH = None

# Test with all images in folder
# INPUT_PATH = PROJECT_ROOT / "models/testing/images"

# Test with specific videos (uncomment one to use)
# INPUT_PATH = PROJECT_ROOT / "models/testing/videos/far_blue.mp4"
# INPUT_PATH = PROJECT_ROOT / "models/testing/videos/near_blue_A.mp4"
# etc...
```

Simply comment/uncomment the line you want to use!

### Key Controls in Test Mode
- **ESC**: Exit test mode
- **SPACE**: Pause/resume video playback
- **Arrow Keys**: Navigate frames (images mode)

## Detection Classes
The system uses 4 trained YOLO classes:
1. **Gauge_Centre**: Center point of pressure gauge
2. **Needle_Tip**: Tip of gauge needle
3. **Valve_Closed**: Closed valve position
4. **Valve_Open**: Open valve position

## File Structure
```
TAIP/
├── main.py              # Main application with test mode support
├── test_mode.py         # Test mode processor and display classes
├── oak_camera.py        # OAK camera with test mode methods
├── config.py            # Configuration with test mode settings
├── vision_processing.py # Computer vision algorithms
├── data_models.py       # Data structures
├── gcs_client.py        # Ground control station client
└── logs/               # System logs
```

## Development Workflow

### 1. Quick Test Setup
```bash
# Edit config.py to set test input
nano /home/pi/EGH455/TAIP/config.py

# Run the system
cd /home/pi/EGH455/TAIP
/home/pi/venvs/depthai_env/bin/python main.py
```

### 2. Validate Detection Performance
- Edit config.py to use a test video
- Run main.py to see live detection visualization
- Check gauge reading calculations with known reference values

### 3. Algorithm Development
- Modify `vision_processing.py` for gauge reading algorithms
- Test changes immediately by running main.py
- No need for physical hardware during development

### 4. Performance Testing
```bash
# Edit config.py to use high frame rate video
# Run performance test
/home/pi/venvs/depthai_env/bin/python main.py
```

## Integration with Original object_detection.py

The test mode functionality incorporates the best features from the original `object_detection.py`.

### Similar Features
- Video and image processing capabilities
- OpenCV display with detection overlays
- Frame-by-frame control for development
- Support for different input formats

### Enhanced Features
- **Full TAIP integration**: Works with complete system
- **Gauge reading calculations**: Processes detections for gauge values
- **GCS communication**: Optional telemetry in test mode
- **Environmental data**: Simulated sensor data for testing
- **Modular architecture**: Easy to extend and modify

## Troubleshooting

### Common Issues
1. **"No module named 'depthai'"**: Use the configured virtual environment
2. **File not found**: Check input paths and file permissions
3. **Display issues**: Ensure X11 forwarding for remote sessions

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=/home/pi/EGH455/TAIP
/home/pi/venvs/depthai_env/bin/python main.py --test --input test_video.mp4 --verbose
```

## Performance Notes
- Test mode processes frames at approximately 30 FPS
- GPU acceleration through DepthAI VPU (Myriad X)
- Memory usage optimized for Raspberry Pi constraints
- Supports HD video input (1920x1080)

## Future Enhancements
1. **Batch processing**: Process multiple files automatically
2. **Result export**: Save detection data to CSV/JSON
3. **Comparative analysis**: Compare results across different models
4. **Remote test mode**: Process files over network connection

This test mode functionality provides a complete development and testing environment that mirrors the capabilities of the original `object_detection.py` while being fully integrated with the TAIP system architecture.
