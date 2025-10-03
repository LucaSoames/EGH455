#!/bin/bash

# Integration setup script for TAIP + React Frontend
# This script sets up and builds the integrated system

set -e  # Exit on any error

echo "=========================================="
echo "TAIP Frontend Integration Setup"
echo "=========================================="

# Get project root
PROJECT_ROOT="/home/pi/EGH455"
FRONTEND_DIR="$PROJECT_ROOT/frontend/frontend"
TAIP_DIR="$PROJECT_ROOT/TAIP"

# Check if directories exist
if [ ! -d "$FRONTEND_DIR" ]; then
    echo "❌ Frontend directory not found: $FRONTEND_DIR"
    exit 1
fi

if [ ! -d "$TAIP_DIR" ]; then
    echo "❌ TAIP directory not found: $TAIP_DIR"
    exit 1
fi

echo "✅ Project directories found"

# Step 1: Install Python dependencies for TAIP
echo ""
echo "📦 Installing Python dependencies for TAIP..."
cd "$PROJECT_ROOT"
pip3 install Flask-SocketIO Flask-CORS

# Step 2: Install Node.js dependencies for React frontend
echo ""
echo "📦 Installing Node.js dependencies for React frontend..."
cd "$FRONTEND_DIR"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found in frontend directory"
    exit 1
fi

# Install dependencies
npm install

# Step 3: Build React frontend
echo ""
echo "🏗️  Building React frontend..."
npm run build

# Check if build was successful
if [ ! -d "build" ]; then
    echo "❌ Frontend build failed - build directory not created"
    exit 1
fi

echo "✅ Frontend built successfully"

# Step 4: Create start script for integrated system
echo ""
echo "📝 Creating integrated system start script..."

cat > "$PROJECT_ROOT/start_integrated_system.sh" << 'EOF'
#!/bin/bash

# Start script for TAIP + React Frontend integrated system

PROJECT_ROOT="/home/pi/EGH455"
TAIP_DIR="$PROJECT_ROOT/TAIP"

echo "=========================================="
echo "Starting TAIP Integrated System"
echo "=========================================="

cd "$TAIP_DIR"

# Check if required files exist
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in TAIP directory"
    exit 1
fi

if [ ! -f "web_server.py" ]; then
    echo "❌ web_server.py not found in TAIP directory"
    exit 1
fi

# Check if React build exists
if [ ! -d "$PROJECT_ROOT/frontend/frontend/build" ]; then
    echo "❌ React frontend build not found. Please run setup_integration.sh first."
    exit 1
fi

echo "🚀 Starting TAIP system with integrated web interface..."
echo "📱 Web interface will be available at: http://localhost:5000"
echo "🌐 Or from external devices at: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop the system"
echo ""

# Start the main TAIP application (which includes the web server)
python3 main.py
EOF

chmod +x "$PROJECT_ROOT/start_integrated_system.sh"

# Step 5: Create test script for web server only
echo ""
echo "📝 Creating web server test script..."

cat > "$PROJECT_ROOT/test_web_server.sh" << 'EOF'
#!/bin/bash

# Test script for web server component only

PROJECT_ROOT="/home/pi/EGH455"
TAIP_DIR="$PROJECT_ROOT/TAIP"

echo "=========================================="
echo "Testing TAIP Web Server"
echo "=========================================="

cd "$TAIP_DIR"

# Check if React build exists
if [ ! -d "$PROJECT_ROOT/frontend/frontend/build" ]; then
    echo "❌ React frontend build not found. Please run setup_integration.sh first."
    exit 1
fi

echo "🧪 Starting web server in test mode with mock data..."
echo "📱 Web interface will be available at: http://localhost:5000"
echo "🌐 Or from external devices at: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop the test server"
echo ""

# Start the web server with mock data
python3 web_server.py
EOF

chmod +x "$PROJECT_ROOT/test_web_server.sh"

echo ""
echo "=========================================="
echo "✅ Integration setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. To test the web server with mock data:"
echo "   ./test_web_server.sh"
echo ""
echo "2. To start the full integrated system:"
echo "   ./start_integrated_system.sh"
echo ""
echo "🌐 The web interface will be available at:"
echo "   - Local: http://localhost:5000"
echo "   - Network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "📊 Features available:"
echo "   - Real-time telemetry data"
echo "   - Live video streaming"
echo "   - System event logs"
echo "   - Environmental sensor readings"
echo "   - Drill system status"