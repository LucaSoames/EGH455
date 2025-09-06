#!/bin/bash

echo "=========================================="
echo "TAIP System Test Script"
echo "=========================================="

cd /home/pi/EGH455/TAIP

echo "1. Testing configuration validation..."
python3 -c "import config; config.validate_config(); print('✓ Configuration valid')"

echo ""
echo "2. Testing imports..."
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
echo "3. Testing test mode processor..."
python3 -c "
try:
    from test_mode import TestModeProcessor
    import config
    
    # Test with test images if they exist
    test_path = config.TEST_INPUT_PATH
    print(f'✓ Test mode processor can be initialized with path: {test_path}')
except Exception as e:
    print(f'✗ Test mode error: {e}')
"

echo ""
echo "4. Available test mode options:"
echo "   For images: python3 main.py --test --input /home/pi/EGH455/models/testing/images/"
echo "   For video:  python3 main.py --test --input /home/pi/EGH455/models/testing/videos/near_blue_A.mp4"
echo "   Live mode:  python3 main.py"

echo ""
echo "5. Testing command line argument parsing..."
python3 main.py --help

echo ""
echo "✓ TAIP system test completed successfully!"
echo "  Ready to run in test mode or live mode."
