import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

interface TelemetryData {
  id?: number;
  // No uav_id needed - single UAV system
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
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const socket = io('http://localhost:5000');

    socket.on('connect', () => {
      console.log('Connected to server');
      setConnected(true);
      // Request initial telemetry data
      socket.emit('request_telemetry', {});  // No UAV ID needed
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from server');
      setConnected(false);
    });

    socket.on('telemetry_update', (data: TelemetryData) => {
      setTelemetry(data);
    });

    socket.on('error', (error) => {
      console.error('Socket error:', error);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  if (!connected) {
    return (
      <div className="card">
        <p style={{ color: '#e74c3c' }}>⚠️ Disconnected from server</p>
      </div>
    );
  }

  if (!telemetry) {
    return (
      <div className="card">
        <p>📡 Waiting for telemetry data...</p>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return '#27ae60';
      case 'warning': return '#f39c12';
      case 'critical': return '#e74c3c';
      default: return '#95a5a6';
    }
  };

  return (
    <div className="card">
      <div style={{ marginBottom: '1rem' }}>
        <span style={{ color: getStatusColor(telemetry.status), fontWeight: 'bold' }}>
          ● {telemetry.status.toUpperCase()}
        </span>
        <span style={{ float: 'right', color: '#666', fontSize: '0.9rem' }}>
          THE UAV
        </span>
      </div>

      <div className="telemetry-grid">
        <div className="telemetry-item">
          <span>🔋 Battery</span>
          <strong>{telemetry.battery_level?.toFixed(1) || 'N/A'}%</strong>
        </div>
        
        <div className="telemetry-item">
          <span>🌡️ Temperature</span>
          <strong>{telemetry.temperature?.toFixed(1) || 'N/A'}°C</strong>
        </div>
        
        <div className="telemetry-item">
          <span>💧 Humidity</span>
          <strong>{telemetry.humidity?.toFixed(1) || 'N/A'}%</strong>
        </div>
        
        <div className="telemetry-item">
          <span>📍 Altitude</span>
          <strong>{telemetry.altitude?.toFixed(1) || 'N/A'}m</strong>
        </div>
        
        {telemetry.latitude && telemetry.longitude && (
          <>
            <div className="telemetry-item">
              <span>🧭 Latitude</span>
              <strong>{telemetry.latitude.toFixed(6)}</strong>
            </div>
            
            <div className="telemetry-item">
              <span>🧭 Longitude</span>
              <strong>{telemetry.longitude.toFixed(6)}</strong>
            </div>
          </>
        )}
      </div>
      
      <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#666' }}>
        Last update: {new Date(telemetry.timestamp).toLocaleString()}
      </div>
    </div>
  );
}

export default TelemetryDisplay;