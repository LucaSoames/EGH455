#!/bin/bash

# TAIP Subsystem Setup Script
# EGH455 UAVPayloadTAQ Project

echo "=============================================="
echo "TAIP Subsystem Setup Script"
echo "EGH455 UAVPayloadTAQ Project"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    print_warning "This script is designed for Raspberry Pi. Continuing anyway..."
fi

# Update system packages
echo -e "\n1. Updating system packages..."
if sudo apt update && sudo apt upgrade -y; then
    print_status "System packages updated"
else
    print_error "Failed to update system packages"
    exit 1
fi

# Install required system packages
echo -e "\n2. Installing system dependencies..."
SYSTEM_PACKAGES="python3-pip python3-dev python3-venv git cmake build-essential pkg-config"
SYSTEM_PACKAGES="$SYSTEM_PACKAGES libhdf5-dev libhdf5-serial-dev libatlas-base-dev"
SYSTEM_PACKAGES="$SYSTEM_PACKAGES libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev"
SYSTEM_PACKAGES="$SYSTEM_PACKAGES libswscale-dev libv4l-dev libxvidcore-dev libx264-dev"
SYSTEM_PACKAGES="$SYSTEM_PACKAGES libgtk-3-dev libcanberra-gtk3-dev libqtgui4 libqt4-test"
SYSTEM_PACKAGES="$SYSTEM_PACKAGES i2c-tools libi2c-dev"

if sudo apt install -y $SYSTEM_PACKAGES; then
    print_status "System dependencies installed"
else
    print_error "Failed to install system dependencies"
    exit 1
fi

# Enable I2C and SPI for Enviro+
echo -e "\n3. Enabling I2C and SPI interfaces..."
if sudo raspi-config nonint do_i2c 0 && sudo raspi-config nonint do_spi 0; then
    print_status "I2C and SPI enabled"
else
    print_warning "Could not automatically enable I2C/SPI. Please enable manually using raspi-config"
fi

# Create virtual environment
echo -e "\n4. Creating Python virtual environment..."
if python3 -m venv venv; then
    print_status "Virtual environment created"
else
    print_error "Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo -e "\n5. Upgrading pip..."
if pip install --upgrade pip; then
    print_status "Pip upgraded"
else
    print_warning "Failed to upgrade pip"
fi

# Install Python dependencies
echo -e "\n6. Installing Python dependencies..."
if pip install -r requirements.txt; then
    print_status "Python dependencies installed"
else
    print_error "Failed to install Python dependencies"
    echo "Trying to install individual critical packages..."
    
    # Install critical packages one by one
    CRITICAL_PACKAGES="numpy opencv-contrib-python-headless depthai requests flask"
    for package in $CRITICAL_PACKAGES; do
        echo "Installing $package..."
        if pip install $package; then
            print_status "$package installed"
        else
            print_error "Failed to install $package"
        fi
    done
fi

# Install Pimoroni libraries (Raspberry Pi specific)
echo -e "\n7. Installing Pimoroni Enviro+ libraries..."
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    if pip install enviroplus RPi.GPIO; then
        print_status "Enviro+ libraries installed"
    else
        print_warning "Failed to install Enviro+ libraries"
    fi
else
    print_warning "Skipping Enviro+ installation (not on Raspberry Pi)"
fi

# Create logs directory
echo -e "\n8. Creating logs directory..."
if mkdir -p logs; then
    print_status "Logs directory created"
else
    print_warning "Could not create logs directory"
fi

# Create debug images directory
echo -e "\n9. Creating debug directory..."
if mkdir -p debug_images; then
    print_status "Debug directory created"
else
    print_warning "Could not create debug directory"
fi

# Check for model files
echo -e "\n10. Checking model files..."
if [ -f "../models/blobs/YOLOv8n.blob" ]; then
    print_status "YOLOv8n model found"
elif [ -f "../models/blobs/YOLOv8s.blob" ]; then
    print_status "YOLOv8s model found (will use as backup)"
else
    print_error "No model files found in ../models/blobs/"
    echo "Please ensure model .blob files are present before running the system"
fi

# Test OAK camera connection
echo -e "\n11. Testing OAK camera connection..."
if timeout 10 python3 -c "
import depthai as dai
try:
    devices = dai.Device.getAllAvailableDevices()
    if devices:
        print('OAK camera detected:', devices[0].name)
        exit(0)
    else:
        print('No OAK cameras found')
        exit(1)
except Exception as e:
    print('Error testing camera:', e)
    exit(1)
" 2>/dev/null; then
    print_status "OAK camera detected"
else
    print_warning "OAK camera not detected or not connected"
fi

# Run verification script
echo -e "\n12. Running installation verification..."
if python3 verify_taip_installation.py; then
    print_status "Installation verification passed"
else
    print_warning "Installation verification had issues (see above)"
fi

# Final instructions
echo -e "\n=============================================="
echo "Setup completed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Configure GCS URL in config.py if needed"
echo "3. Connect OAK-D Lite camera via USB 3.0"
echo "4. Connect Pimoroni Enviro+ HAT to GPIO"
echo "5. Run the system: python3 main.py"
echo ""
echo "For troubleshooting, see README_TAIP.md"
echo ""

# Deactivate virtual environment
deactivate

print_status "Setup script completed successfully!"

echo ""
echo "To start the TAIP system:"
echo "  source venv/bin/activate"
echo "  python3 main.py"
echo ""
