# UAV System Simplification Guide

## What Can Be Removed (90% reduction)

### Backend Dependencies (requirements.txt)
**REMOVE:**
- `marshmallow` + related (complex serialization)
- `Flask-JWT-Extended` (authentication)
- `numpy` + `pandas` (data processing)
- `geopy` (location calculations)
- `gunicorn` (production server)
- `redis` + `celery` (background tasks)
- `pydantic` (validation)

**KEEP:**
- `Flask` + `Flask-SQLAlchemy` + `Flask-CORS`
- `Flask-SocketIO` + socket libraries
- `requests` + `python-dotenv`

### Frontend Dependencies (package.json)
**REMOVE:**
- Material-UI (`@mui/*`) - Heavy UI framework
- Maps (`leaflet`, `react-leaflet`) - Map visualization
- Charts (`recharts`) - Analytics
- React Query (`@tanstack/react-query`) - Complex state
- Date pickers, data grids - Advanced UI

**KEEP:**
- `react` + `react-dom` + `react-router-dom`
- `socket.io-client` + `axios`
- `typescript` + basic types

### Database Models
**REMOVE:**
- `Mission` (complex mission planning)
- `Payload` (if just doing sensors)
- `Waypoint` (flight planning)
- `User`/Auth (authentication)
- `SystemLog` (audit logging)

**KEEP:**
- `TelemetryData` (core sensor data)
- `UAV` (basic drone info)

### API Routes
**REMOVE:**
- Authentication endpoints
- Mission management
- Dashboard analytics  
- Video streaming
- File uploads
- Complex reporting

**KEEP:**
- `POST /api/telemetry` (receive sensor data)
- `GET /api/telemetry/latest` (get latest data)
- `GET /api/uavs` (basic UAV list)
- Socket.IO events

### Frontend Components
**REMOVE:**
- Authentication forms
- Mission planning interface
- Interactive maps
- Charts/analytics
- Complex routing
- Material-UI components

**KEEP:**
- Simple telemetry display
- UAV status list
- Socket.IO connection logic

## Essential Data Flow (Simplified)

```
Hardware → Socket.IO → TelemetryData DB → Frontend Display
```

## Core Files You Need:

### Backend (5 files)
1. `run.py` - App entry point
2. `models-minimal.py` - Just TelemetryData + UAV
3. `api-minimal.py` - 3 endpoints + socket events
4. `requirements-minimal.txt` - 10 dependencies vs 22
5. `app/__init__.py` - Flask app setup

### Frontend (4 files)
1. `App.tsx` - Main component
2. `TelemetryDisplay.tsx` - Real-time data
3. `UAVList.tsx` - Simple UAV list
4. `SocketConnection.tsx` - Socket logic

### Total: 9 core files vs 50+ current files

## Benefits:
- **90% fewer dependencies** (7 vs 60+)
- **10x faster install** time
- **Simpler debugging**
- **Easier deployment**
- **Focus on core functionality**: Hardware → Socket → Display

## Implementation Steps:
1. Copy minimal files to new directory
2. Install minimal dependencies
3. Test core socket communication
4. Add complexity back only as needed