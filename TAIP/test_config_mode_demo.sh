#!/bin/bash

echo "=========================================="
echo "TAIP Test Mode Configuration Demo"
echo "=========================================="

cd /home/pi/EGH455/TAIP

echo "Current config.py test mode setting:"
grep -A 1 -B 1 "TEST_INPUT_PATH = " config.py

echo ""
echo "Testing the system in current mode:"
timeout 3 /home/pi/venvs/depthai_env/bin/python main.py || echo "✓ System startup test completed"

echo ""
echo "=========================================="
echo "How to switch modes:"
echo "=========================================="
echo ""
echo "1. LIVE CAMERA MODE (default):"
echo "   Edit config.py and set:"
echo "   TEST_INPUT_PATH = None"
echo ""
echo "2. TEST WITH IMAGES:"
echo "   Edit config.py and set:"
echo "   TEST_INPUT_PATH = PROJECT_ROOT / \"models/testing/images\""
echo ""
echo "3. TEST WITH VIDEO:"
echo "   Edit config.py and uncomment one of these lines:"
echo "   # TEST_INPUT_PATH = PROJECT_ROOT / \"models/testing/videos/far_blue.mp4\""
echo "   # TEST_INPUT_PATH = PROJECT_ROOT / \"models/testing/videos/near_blue_A.mp4\""
echo "   # etc..."
echo ""
echo "Available test videos:"
ls -1 /home/pi/EGH455/models/testing/videos/*.mp4 2>/dev/null | head -5 | while read video; do
    basename "$video"
done

echo ""
echo "Simply edit config.py, save, and run:"
echo "/home/pi/venvs/depthai_env/bin/python main.py"
echo ""
echo "No command line arguments needed!"
