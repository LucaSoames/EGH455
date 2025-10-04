# AI Coding Agent Guidelines for TAIP System

## Project Overview
The TAIP system is a **client-server architecture** designed for UAV payload tracking and control. It consists of:
- **GCS Server**: A Flask-based backend that receives telemetry and video frames from the Raspberry Pi and serves the React frontend.
- **GCS Client**: A Python application running on the Raspberry Pi, responsible for sending telemetry and video frames to the server.
- **React Frontend**: A web interface for monitoring system status and telemetry data.

## Key Components
- **Backend** (`TAIP/gcs_server.py`):
  - Flask server with SocketIO for real-time communication.
  - API endpoints:
    - `POST /telemetry`: Receives telemetry data.
    - `POST /frame`: Receives video frames.
    - `GET /api/health`: Health check endpoint.
- **Frontend** (`frontend/frontend`):
  - React-based UI served from the GCS server.
  - Built using `npm run build`.
- **Client** (`TAIP/main.py`):
  - Interfaces with the OAK-D Lite camera and Enviro+ sensors.
  - Sends data to the GCS server.

## Developer Workflows

### Local Testing
1. Build the frontend:
   ```cmd
   cd "frontend/frontend"
   npm install
   npm run build
   ```
2. Start the GCS server:
   ```cmd
   cd "TAIP"
   python gcs_server.py
   ```
3. Open the frontend at `http://localhost:3000`.
4. Simulate Pi data:
   ```cmd
   python gcs_client.py
   ```

### Production Setup (Raspberry Pi)
1. Update `GCS_LAPTOP_IP` in `config.py` to the laptop's IP.
2. Start the GCS server on the laptop:
   ```cmd
   python gcs_server.py --host 0.0.0.0 --port 3000
   ```
3. Run the main application on the Raspberry Pi:
   ```bash
   python3 main.py
   ```

## Project-Specific Conventions
- **Configuration**:
  - Key parameters are defined in `TAIP/config.py`.
  - Example: `GCS_URL`, `POST_FRAME_FPS`, `BLOB_PATH`.
- **Error Handling**:
  - Use `try-except` blocks for hardware initialization (e.g., OAK-D Lite, Enviro+ sensors).
- **Logging**:
  - Server logs appear in the terminal running `gcs_server.py`.
  - Client logs appear in the terminal running `main.py`.

## Integration Points
- **YOLO Models**:
  - Stored in `models/blobs/`.
  - Ensure `.blob` files are present before running the system.
- **Environmental Sensors**:
  - Managed via `enviro_lcd.py`.
  - Provides temperature, humidity, and pressure data.
- **Camera**:
  - OAK-D Lite used for video streaming and object detection.

## Troubleshooting
- **Frontend Not Loading**:
  - Rebuild the frontend: `npm run build`.
  - Ensure `frontend/frontend/build/` contains `index.html`.
- **Server Issues**:
  - Check if the server is running: `curl http://localhost:3000/api/health`.
  - Verify firewall settings.
- **Low Frame Rate**:
  - Reduce `POST_FRAME_FPS` in `config.py`.
  - Disable `SHOW_LIVE_VISUALISATION`.

## Examples
- **Health Check Endpoint**:
  ```cmd
  curl http://localhost:3000/api/health
  ```
- **Changing Ports**:
  ```cmd
  python gcs_server.py --port 5000
  ```

## Notes
- Ensure all dependencies are installed as per `requirements.txt`.
- Follow the `RUN_INSTRUCTIONS.md` for detailed setup steps.