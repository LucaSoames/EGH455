# 🚀 TAIP System - Running Instructions

## Overview
Your system has a **client-server architecture** with:
- **GCS Server** (Ground Control Station) - Receives data from Raspberry Pi and serves the frontend
- **GCS Client** (On Raspberry Pi) - Sends telemetry and video frames to the server
- **React Frontend** - Web interface for monitoring

## ⚙️ Port Configuration
- **GCS Server**: Port `3000` (Flask + SocketIO)
- **Frontend**: Served from port `3000` (integrated with server)
- **Pi Client**: Connects to GCS server at `10.88.52.93:3000` (configurable in `config.py`)

---

## 📋 Prerequisites

### On Your Laptop (Windows):
1. **Python 3.x** installed
2. **Node.js and npm** installed (for frontend)
3. Install Python dependencies:
   ```cmd
   cd "c:\Users\ncart\OneDrive\QUT\EGH455 Advanced Systems Design\EGH455"
   pip install flask flask-socketio flask-cors opencv-python numpy
   ```

### On Raspberry Pi:
1. All dependencies from `requirements.txt` installed
2. Network connection to laptop (same WiFi/network)

---

## 🖥️ Running on Your Laptop (Testing Locally)

### Step 1: Build the Frontend (First Time Only)
```cmd
cd "c:\Users\ncart\OneDrive\QUT\EGH455 Advanced Systems Design\EGH455\frontend\frontend"
npm install
npm run build
```

### Step 2: Start the GCS Server
```cmd
cd "c:\Users\ncart\OneDrive\QUT\EGH455 Advanced Systems Design\EGH455\TAIP"
python gcs_server.py
```

You should see:
```
============================================================
GCS Server initialized on 0.0.0.0:3000
============================================================
Starting GCS server...
Frontend: http://0.0.0.0:3000/
API endpoints:
  POST /telemetry  - Receive telemetry from Pi
  POST /frame      - Receive video frames from Pi
  GET  /api/health - Health check
============================================================
```

### Step 3: Open the Frontend
Open your browser and go to:
```
http://localhost:3000
```

### Step 4: Test the GCS Client (Simulate Pi)
In a **new terminal**, test sending data:
```cmd
cd "c:\Users\ncart\OneDrive\QUT\EGH455 Advanced Systems Design\EGH455\TAIP"
python gcs_client.py
```

This will send 5 test packets to verify the connection works.

---

## 🤖 Running on Raspberry Pi (Production)

### Step 1: Update Configuration
Edit `TAIP/config.py` on your **Raspberry Pi**:

```python
# Update this to your laptop's IP address
GCS_LAPTOP_IP = "YOUR_LAPTOP_IP_HERE"  # e.g., "192.168.1.100"
GCS_URL = f"http://{GCS_LAPTOP_IP}:3000"
```

To find your laptop's IP:
```cmd
# On Windows
ipconfig
# Look for "IPv4 Address" under your active network adapter
```

### Step 2: Start GCS Server on Laptop
```cmd
cd "c:\Users\ncart\OneDrive\QUT\EGH455 Advanced Systems Design\EGH455\TAIP"
python gcs_server.py --host 0.0.0.0 --port 3000
```

The `--host 0.0.0.0` allows connections from other devices on the network.

### Step 3: Run Main Application on Pi
On the Raspberry Pi:
```bash
cd ~/EGH455/TAIP
python3 main.py
```

This will:
- Start the camera and sensors
- Process video and detect objects
- Send telemetry and frames to your laptop's GCS server
- Display live feed on Pi (if enabled)

---

## 🔍 Troubleshooting

### Frontend Shows "Disconnected" or No Data

**Check 1: Server Running?**
```cmd
curl http://localhost:3000/api/health
```
Should return: `{"status":"healthy",...}`

**Check 2: Firewall**
- Ensure Windows Firewall allows Python on port 3000
- Or temporarily disable firewall for testing

**Check 4: Network**
- Laptop and Pi must be on same network
- Ping laptop from Pi: `ping YOUR_LAPTOP_IP`

### Blank Page / 404 Errors for JS/CSS Files

**Solution: Rebuild and Restart**
```cmd
cd "c:\Users\ncart\OneDrive\QUT\EGH455 Advanced Systems Design\EGH455\frontend\frontend"
npm run build

cd ..\..\TAIP
python gcs_server.py
```

The server must be **restarted** after rebuilding the frontend.

### Pi Can't Connect to Laptop

**Check 1: Verify IP**
```python
# On Pi, test connection
import requests
response = requests.get("http://YOUR_LAPTOP_IP:3000/api/health")
print(response.json())
```

**Check 2: Server Binding**
Ensure server started with `--host 0.0.0.0`:
```cmd
python gcs_server.py --host 0.0.0.0 --port 3000
```

### Frontend Not Loading

**Rebuild Frontend:**
```cmd
cd "c:\Users\ncart\OneDrive\QUT\EGH455 Advanced Systems Design\EGH455\frontend\frontend"
npm run build
```

**Check Build Directory:**
Ensure `frontend/frontend/build/` exists with `index.html`

### Port Already in Use

**Change Port:**
```cmd
# Use different port
python gcs_server.py --port 3001
```

Then update frontend to match (see below).

---

## 🔧 Advanced Configuration

### Using Different Port (e.g., 5000)

**Option 1: Change Server to Match Frontend** ✅ EASIER
```cmd
python gcs_server.py --port 5000
```

**Option 2: Rebuild Frontend for Different Port**
1. Edit all frontend `*.tsx` files: Change `localhost:3000` → `localhost:YOUR_PORT`
2. Rebuild: `npm run build`

### Testing Without Pi

Run the test client to simulate Pi data:
```cmd
cd TAIP
python gcs_client.py
```

### Viewing Logs

**Server Logs:**
Flask/SocketIO logs appear in the terminal running `gcs_server.py`

**Client Logs:**
Logs appear in terminal running `main.py` on Pi

---

## 📊 System Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Raspberry Pi   │         │   Laptop (GCS)   │
│                 │         │                  │
│  ┌──────────┐   │  HTTP   │  ┌────────────┐  │
│  │ Camera   │   │  POST   │  │ Flask      │  │
│  │ Sensors  │───┼────────►│  │ Server     │  │
│  │ YOLO     │   │ :3000   │  │ (Port 3000)│  │
│  └──────────┘   │         │  └─────┬──────┘  │
│  main.py        │         │        │         │
│  gcs_client.py  │         │        │         │
└─────────────────┘         │  ┌─────▼──────┐  │
                            │  │ SocketIO   │  │
                            │  │ WebSocket  │  │
                            │  └─────┬──────┘  │
                            │        │         │
                            │  ┌─────▼──────┐  │
                            │  │  React     │  │
                            │  │  Frontend  │  │
                            │  └────────────┘  │
                            │                  │
                            │  Browser         │
                            │  localhost:3000  │
                            └──────────────────┘
```

---

## 🎯 Quick Start Summary

### **Local Testing (Laptop Only)**
1. `npm run build` (in frontend/frontend)
2. `python gcs_server.py` (in TAIP)
3. Open `http://localhost:3000`
4. `python gcs_client.py` (test data)

### **Pi → Laptop (Production)**
1. Update `GCS_LAPTOP_IP` in Pi's `config.py`
2. Start server on laptop: `python gcs_server.py --host 0.0.0.0`
3. Open `http://localhost:3000` on laptop
4. Run `python3 main.py` on Pi

---

## 📞 Support

If issues persist:
1. Check all terminals for error messages
2. Verify network connectivity (`ping`, `curl`)
3. Ensure all dependencies installed
4. Check firewall settings
5. Try rebooting both devices

**Health Check Endpoint:**
```
http://localhost:3000/api/health
```

Returns telemetry/frame counts and server status.
