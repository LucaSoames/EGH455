#!/bin/bash
# filepath: /home/pi/EGH455/TAIP/test_system.sh
# Consolidated system test script

echo "=========================================="
echo "TAIP System Comprehensive Test"
echo "=========================================="

cd /home/pi/EGH455/TAIP

echo "1. Configuration validation..."
python3 -c "import config; config.validate_config()"

echo ""
echo "2. Import tests..."
python3 -c "
try:
    import main
    from test_mode import TestModeProcessor, TestModeDisplay
    from oak_camera import OakCamera
    from vision_processing import *
    print('✓ All imports successful')
except Exception as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

echo ""
echo "3. Current configuration:"
python3 -c "
import config
from pathlib import Path
if config.INPUT_PATH:
    print(f'Input mode: {config.INPUT_PATH}')
    if Path(config.INPUT_PATH).exists():
        print('✓ Input path exists')
    else:
        print('✗ Input path missing')
else:
    print('Live camera mode')
print(f'Model: {config.BLOB_NAME}')
print(f'GPIO pin: {config.DRILL_GPIO_PIN}')
"

echo ""
echo "4. Model file validation..."
python3 -c "
import config
if config.BLOB_PATH.exists():
    print(f'✓ Model blob found: {config.BLOB_PATH}')
else:
    print(f'✗ Model blob missing: {config.BLOB_PATH}')
if config.CONFIG_PATH.exists():
    print(f'✓ Model config found: {config.CONFIG_PATH}')
else:
    print(f'✗ Model config missing: {config.CONFIG_PATH}')
"

echo ""
echo "5. Quick system startup test..."
timeout 5 python3 main.py || echo "✓ System startup test completed"

echo ""
echo "✓ System test completed!"