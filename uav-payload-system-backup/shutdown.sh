#!/usr/bin/env bash
set -euo pipefail

# shutdown.sh
# Gracefully shuts down the UAV Payload system services
# - Stops backend and frontend processes
# - Kills processes on common development ports
# - Provides feedback on shutdown status

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"

echo "Shutting down UAV Payload System..."

# Function to kill processes on a specific port
kill_port() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null || echo "")
    if [ -n "$pids" ]; then
        echo "Stopping processes on port $port..."
        echo "$pids" | xargs kill -15 2>/dev/null || true
        sleep 1
        # Force kill if still running
        local remaining=$(lsof -ti :$port 2>/dev/null || echo "")
        if [ -n "$remaining" ]; then
            echo "Force stopping processes on port $port..."
            echo "$remaining" | xargs kill -9 2>/dev/null || true
        fi
        echo "Port $port cleared"
    else
        echo "No processes found on port $port"
    fi
}

# Kill processes on development ports
echo "Checking development ports..."
kill_port 5000  # Backend
kill_port 3000  # Frontend default
kill_port 3001  # Frontend alternate
kill_port 3002  # Frontend alternate
kill_port 3003  # Frontend alternate

# Kill any remaining Node.js/npm processes from this project
echo "Stopping any remaining Node.js processes from this project..."
pkill -f "uav-payload-system.*node" 2>/dev/null || true
pkill -f "uav-payload-system.*npm" 2>/dev/null || true
pkill -f "react-scripts.*start" 2>/dev/null || true

# Kill any Python processes from this project
echo "Stopping any remaining Python processes from this project..."
pkill -f "uav-payload-system.*python" 2>/dev/null || true
pkill -f "run\.py" 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 2

# Verify shutdown
echo ""
echo "Verifying shutdown..."
remaining_ports=""
for port in 3000 3001 3002 3003 5000; do
    if lsof -ti :$port >/dev/null 2>&1; then
        remaining_ports="$remaining_ports $port"
    fi
done

if [ -n "$remaining_ports" ]; then
    echo "⚠️  Warning: Some processes may still be running on ports:$remaining_ports"
    echo "You may need to manually kill them with: kill -9 \$(lsof -ti :PORT)"
else
    echo "✅ All UAV Payload System processes stopped successfully"
fi

# Display log locations for debugging if needed
if [ -d "$LOG_DIR" ]; then
    echo ""
    echo "Logs available at: $LOG_DIR"
    echo "- Backend log: $LOG_DIR/backend.log"
    echo "- Frontend log: $LOG_DIR/frontend.log"
fi

echo "Shutdown complete!"