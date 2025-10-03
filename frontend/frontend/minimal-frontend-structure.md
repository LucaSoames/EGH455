# Minimal Frontend Structure

## Keep Only These Components:

```
src/
├── App.tsx                 # Main app component
├── index.tsx              # Entry point
└── components/
    ├── TelemetryDisplay.tsx   # Real-time telemetry data
    ├── UAVList.tsx           # Simple UAV list
    └── SocketConnection.tsx   # Socket.IO connection logic

## Remove These Heavy Components:
- Authentication system (unless needed)
- Complex routing (react-router-dom can be simplified)
- Material-UI components (@mui/*)
- Charts and analytics (recharts)
- Map components (react-leaflet, leaflet)
- Date pickers (@mui/x-date-pickers)
- Data grids (@mui/x-data-grid)
- React Query for complex state management

## Essential Frontend Code:

### App.tsx (Minimal)
```tsx
import React from 'react';
import TelemetryDisplay from './components/TelemetryDisplay';
import UAVList from './components/UAVList';
import SocketConnection from './components/SocketConnection';

function App() {
  return (
    <div className="App">
      <h1>UAV Telemetry System</h1>
      <SocketConnection />
      <UAVList />
      <TelemetryDisplay />
    </div>
  );
}

export default App;
```

### TelemetryDisplay.tsx (Minimal)
```tsx
import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

interface TelemetryData {
  latitude?: number;
  longitude?: number;
  altitude?: number;
  battery_level?: number;
  temperature?: number;
  humidity?: number;
  status: string;
  timestamp: string;
}

function TelemetryDisplay() {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);

  useEffect(() => {
    socket.on('telemetry_update', (data) => {
      setTelemetry(data);
    });

    return () => {
      socket.off('telemetry_update');
    };
  }, []);

  if (!telemetry) return <div>No telemetry data</div>;

  return (
    <div>
      <h2>Live Telemetry</h2>
      <p>Battery: {telemetry.battery_level}%</p>
      <p>Temperature: {telemetry.temperature}°C</p>
      <p>Humidity: {telemetry.humidity}%</p>
      <p>Status: {telemetry.status}</p>
      <p>Last Update: {telemetry.timestamp}</p>
    </div>
  );
}

export default TelemetryDisplay;
```

## What This Removes:
- ~90% of dependencies (from 20+ to 7 essential ones)
- Complex UI components
- Authentication system
- Mission planning interface
- Map visualization
- Charts and analytics
- File upload functionality
- Advanced routing
- State management complexity
```