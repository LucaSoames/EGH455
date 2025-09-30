#!/usr/bin/env bash
set -euo pipefail

# run_all.sh
# Starts backend and frontend for the UAV Payload system.
# - Uses workspace virtualenv if present: <workspace>/.venv
# - Writes logs to ./logs
# - Starts backend and frontend in background and prints PIDs and URLs

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
LOG_DIR="$ROOT_DIR/logs"
FRONTEND_PORT="${PORT:-3000}"

mkdir -p "$LOG_DIR"

echo "Root: $ROOT_DIR"
echo "Logs: $LOG_DIR"

# Ensure virtualenv exists and determine python executable (prefer workspace .venv)
if [ ! -d "$ROOT_DIR/.venv" ]; then
  echo "Creating virtualenv at $ROOT_DIR/.venv"
  python3 -m venv "$ROOT_DIR/.venv"
fi

PY_CMD="$ROOT_DIR/.venv/bin/python3"
PIP_CMD="$ROOT_DIR/.venv/bin/pip"

if [ ! -x "$PY_CMD" ]; then
  echo "Warning: expected python at $PY_CMD not found, falling back to system python3"
  PY_CMD="python3"
  PIP_CMD="pip3"
fi

echo "Using Python: $PY_CMD"

# Verify python version to avoid compiling heavy scientific packages on unsupported Python
PY_VER=$($PY_CMD -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))' 2>/dev/null || echo "unknown")
echo "Detected Python version: $PY_VER"
if [ "$PY_VER" = "unknown" ]; then
  echo "Warning: could not detect Python version for $PY_CMD"
fi

# If running on macOS with Python 3.13, many scientific wheels (numpy/pandas) may not be available.
# Recommend using Python 3.11 or a conda env. Fail fast with instructions if version is 3.13.
if [ "$PY_VER" = "3.13" ]; then
  cat <<EOF
ERROR: Detected Python 3.13 which may not have prebuilt wheels for numpy/pandas.
Please install Python 3.11 (recommended) and recreate the venv:

  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install --prefer-binary -r backend/requirements.txt

Or use conda to create a Python 3.11 environment and install requirements there.

The script will now exit to avoid long build failures.
EOF
  exit 1
fi

start_backend() {
  echo "Starting backend..."
  cd "$BACKEND_DIR"

  # Install backend dependencies into .venv (prefer prebuilt wheels)
  echo "Installing backend Python requirements (this may take a bit)..."
  $PIP_CMD install --upgrade pip setuptools wheel
  $PIP_CMD install --prefer-binary -r requirements.txt || {
    echo "Backend pip install failed. See logs: $LOG_DIR/backend.log"
  }

  # Run backend entrypoint run.py
  nohup $PY_CMD run.py > "$LOG_DIR/backend.log" 2>&1 &
  BACKEND_PID=$!
  echo "Backend started (pid=$BACKEND_PID) - logs: $LOG_DIR/backend.log"
}

start_frontend() {
  # Pick an available port if the desired one is already in use
  if lsof -iTCP -sTCP:LISTEN -P | grep -q ":$FRONTEND_PORT "; then
    echo "Port $FRONTEND_PORT is in use, finding an available port..."
    # find free port in 3001..3010
    for p in $(seq 3001 3010); do
      if ! lsof -iTCP -sTCP:LISTEN -P | grep -q ":$p "; then
        FRONTEND_PORT=$p
        break
      fi
    done
    echo "Using frontend port: $FRONTEND_PORT"
  fi

  echo "Starting frontend on port $FRONTEND_PORT..."
  cd "$FRONTEND_DIR"

  # Ensure node modules
  if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies (npm install)..."
    npm install
  fi

  # Start frontend (CRA) in background with chosen port
  PORT="$FRONTEND_PORT" nohup npm start > "$LOG_DIR/frontend.log" 2>&1 &
  FRONTEND_PID=$!
  echo "Frontend started (pid=$FRONTEND_PID) - logs: $LOG_DIR/frontend.log"
}

print_summary() {
  echo ""
  echo "--- Services Started ---"
  if [ -n "${BACKEND_PID:-}" ]; then
    echo "Backend PID: $BACKEND_PID  (http://localhost:5000)"
  fi
  if [ -n "${FRONTEND_PID:-}" ]; then
    echo "Frontend PID: $FRONTEND_PID  (http://localhost:$FRONTEND_PORT)"
  fi
  echo "To stop: kill $BACKEND_PID $FRONTEND_PID"
  echo "Logs: $LOG_DIR"
  echo "--- End ---"
}

# Main
start_backend
start_frontend
sleep 2
print_summary

exit 0
