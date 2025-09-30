# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EGH455 Systems Engineering Assessment - A comprehensive UAV payload tracking and acquisition system with hardware integration components. The project consists of three main parts:

1. **UAV Payload System** (`uav-payload-system/`) - Full-stack web application for UAV fleet management
2. **Hardware Integration** (`hardware/`) - Python scripts for Raspberry Pi sensor integration and hardware control  
3. **Documentation** - Various design documents and system requirements

## Architecture

### UAV Payload System (Web Application)
- **Frontend**: React TypeScript application with Material-UI components
- **Backend**: Flask Python API with SQLAlchemy ORM
- **Database**: SQLite for development, designed for PostgreSQL in production
- **Real-time**: Socket.IO for live telemetry updates
- **Features**: Authentication, UAV management, mission planning, payload tracking, telemetry monitoring

### Hardware Integration
- **Platform**: Raspberry Pi with various sensors (environmental, camera, servo control)
- **Sensors**: Pimoroni environmental sensors, servo motors, camera systems
- **Communication**: Flask servers for hardware data bridging
- **Database**: SQLite for sensor data storage

## Quick Start Commands

### Full System Startup
```bash
# Start both frontend and backend services
./uav-payload-system/run_all.sh

# Stop all services
./uav-payload-system/shutdown.sh
```

### Individual Components

#### UAV Web Application
```bash
# Backend only
cd uav-payload-system/backend
source ../.venv/bin/activate
python run.py

# Frontend only  
cd uav-payload-system/frontend
npm install
npm start

# Initialize database with sample data
cd uav-payload-system/backend
python -c "from app import db; db.create_all()"
# or use Flask CLI: flask init_db
```

#### Hardware Components
```bash
# Install hardware dependencies
cd hardware
pip install -r requirements.txt

# Run hardware bridge server
python hardware_bridge.py

# Run individual sensor tests
python pimoroni_v5.py
python servo_motor.py
```

## Development Setup

### Prerequisites
- Python 3.11+ (avoid 3.13 due to NumPy/Pandas compatibility issues)
- Node.js 16+
- Virtual environment recommended

### Python Virtual Environment
The system uses a shared virtual environment at the root level (`.venv/`) to avoid duplication between hardware and web backend components.

### Database Management
```bash
# Reset database
cd uav-payload-system/backend  
python -c "from app import db; db.drop_all(); db.create_all()"

# Create admin user
python -c "from run import *; create_admin()"
```

### Hardware Development
- Raspberry Pi GPIO access required for full hardware functionality
- Pimoroni environmental sensors (temperature, humidity, air quality)
- Servo motor control for mechanical systems
- Camera integration for visual data

## Project Structure

```
egh455/
├── uav-payload-system/          # Web application
│   ├── backend/                 # Flask API server
│   │   ├── app/                 # Application modules
│   │   ├── run.py              # Main entry point
│   │   └── requirements.txt    # Python dependencies
│   ├── frontend/               # React TypeScript app
│   │   ├── src/               # Source code
│   │   ├── package.json       # Node.js dependencies  
│   │   └── build/            # Production build
│   └── run_all.sh            # Startup script
├── hardware/                   # Hardware integration
│   ├── hardware_bridge.py     # Main hardware bridge server
│   ├── pimoroni_v5.py         # Environmental sensors
│   ├── servo_motor.py         # Motor control
│   ├── requirements.txt       # Hardware dependencies
│   └── models/               # Hardware models and utilities
└── docs/                     # Design documentation (*.md, *.docx files)
```

## Default Credentials

**UAV Web Application:**
- Admin: `admin` / `admin123`  
- Operator: `operator` / `operator123`

## Important Notes

### Python Version Compatibility
- Use Python 3.11 for best compatibility with scientific packages (NumPy, Pandas)
- Python 3.13 may cause build issues with precompiled wheels

### Hardware Requirements  
- Hardware scripts require Linux/Raspberry Pi OS for GPIO access
- Windows/Mac users can run web application but not hardware components

### Port Configuration
- Backend API: http://localhost:5000
- Frontend: http://localhost:3000 (auto-detects available ports)
- Hardware Bridge: Various ports depending on component

### Testing
- Frontend: `npm test` (React testing library)
- Backend: No formal test suite currently implemented
- Hardware: Individual component test files available

## Common Development Tasks

### Adding New UAV Models
1. Update database schema in `uav-payload-system/backend/app/models.py`
2. Run database migration
3. Update frontend components in `uav-payload-system/frontend/src/`

### Hardware Integration
1. Add new sensor modules to `hardware/` directory
2. Update `hardware_bridge.py` to include new endpoints
3. Test with individual component scripts

### Mission Planning
- Waypoints stored in database with lat/lng coordinates
- Mission status tracking through backend API
- Real-time telemetry updates via Socket.IO