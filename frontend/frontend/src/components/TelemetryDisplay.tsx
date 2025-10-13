import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

interface EnvironmentalData {
  temperature_c?: number;
  pressure_hpa?: number;
  humidity_rh?: number;
  light_lux?: number;
  pi_temperature_c?: number;
  gas_readings?: {
    reducing_ohms?: number;
    oxidising_ohms?: number;
    nh3_ohms?: number;
    reducing_ppm?: number;
    oxidising_ppm?: number;
    nh3_ppm?: number;
  };
}

interface TelemetryData {
  timestamp: string;
  gauge_pressure_bar?: number;
  environmental_data?: EnvironmentalData;
  yolo_detections?: Array<{
    label: string;
    confidence: number;
  }>;
  aruco_markers?: Array<{
    marker_id: number;
    distance_m: number;
  }>;
}

function TelemetryDisplay() {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const socket = io(window.location.origin, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    socket.on('connect', () => {
      console.log('TelemetryDisplay: Connected to server');
      setConnected(true);
      socket.emit('request_telemetry', {});
    });

    socket.on('disconnect', () => {
      console.log('TelemetryDisplay: Disconnected from server');
      setConnected(false);
    });

    socket.on('telemetry_update', (data: TelemetryData) => {
      console.log('Received telemetry:', data);
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
        <p style={{ color: '#e74c3c' }}>Disconnected from server</p>
      </div>
    );
  }

  if (!telemetry) {
    return (
      <div className="card">
        <p>Waiting for telemetry data...</p>
      </div>
    );
  }

  const env = telemetry.environmental_data;
  const getStatusColor = () => {
    if (!telemetry.gauge_pressure_bar) return '#95a5a6';
    if (telemetry.gauge_pressure_bar < 1.0) return '#e74c3c';
    if (telemetry.gauge_pressure_bar < 3.0) return '#f39c12';
    return '#27ae60';
  };

  return (
    <div className="card">
      <div style={{ marginBottom: '1rem' }}>
        <span style={{ color: getStatusColor(), fontWeight: 'bold' }}>{connected ? 'CONNECTED' : 'DISCONNECTED'}</span>
        <span style={{ float: 'right', color: '#666', fontSize: '0.9rem' }}>
          TAIP PAYLOAD
        </span>
      </div>

      {/* Gauge Pressure */}
      <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
        <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1rem', color: '#666' }}>Gauge Pressure</h3>
        <div style={{ fontSize: '2rem', fontWeight: 'bold', color: getStatusColor() }}>
          {telemetry.gauge_pressure_bar != null ? telemetry.gauge_pressure_bar.toFixed(2) : 'N/A'} bar
        </div>
      </div>

      {/* Environmental Sensors */}
      <h3 style={{ marginBottom: '1rem', fontSize: '1rem', color: '#666' }}>Environmental Data</h3>
      <div className="telemetry-grid">
        <div className="telemetry-item">
          <span>Temperature</span>
          <strong>{env?.temperature_c?.toFixed(1) || 'N/A'}°C</strong>
        </div>
        
        <div className="telemetry-item">
          <span>Humidity</span>
          <strong>{env?.humidity_rh?.toFixed(1) || 'N/A'}%</strong>
        </div>
        
        <div className="telemetry-item">
          <span>Pressure</span>
          <strong>{env?.pressure_hpa?.toFixed(1) || 'N/A'} hPa</strong>
        </div>
        
        <div className="telemetry-item">
          <span>Light</span>
          <strong>{env?.light_lux?.toFixed(0) || 'N/A'} lux</strong>
        </div>

        {env?.pi_temperature_c != null && (
          <div className="telemetry-item">
            <span>Pi CPU Temp</span>
            <strong>{env.pi_temperature_c.toFixed(1)}°C</strong>
          </div>
        )}
      </div>

      {/* Gas Sensors (if available) */}
      {env?.gas_readings && (
        <>
          <h3 style={{ marginTop: '1.5rem', marginBottom: '1rem', fontSize: '1rem', color: '#666' }}>Gas Sensors</h3>
          <div className="telemetry-grid">
            {env.gas_readings.reducing_ppm != null && (
              <div className="telemetry-item">
                <span>CO (Reducing)</span>
                <strong>{env.gas_readings.reducing_ppm.toFixed(1)} ppm</strong>
              </div>
            )}
            
            {env.gas_readings.oxidising_ppm != null && (
              <div className="telemetry-item">
                <span>NO2 (Oxidising)</span>
                <strong>{env.gas_readings.oxidising_ppm.toFixed(1)} ppm</strong>
              </div>
            )}
            
            {env.gas_readings.nh3_ppm != null && (
              <div className="telemetry-item">
                <span>NH3</span>
                <strong>{env.gas_readings.nh3_ppm.toFixed(1)} ppm</strong>
              </div>
            )}
          </div>
        </>
      )}

      {/* Detection Stats */}
      <h3 style={{ marginTop: '1.5rem', marginBottom: '1rem', fontSize: '1rem', color: '#666' }}>Vision System</h3>
      <div className="telemetry-grid">
        <div className="telemetry-item">
          <span>YOLO Detections</span>
          <strong>{telemetry.yolo_detections?.length || 0}</strong>
        </div>
        
        <div className="telemetry-item">
          <span>ArUco Markers</span>
          <strong>{telemetry.aruco_markers?.length || 0}</strong>
        </div>
        
        {telemetry.aruco_markers && telemetry.aruco_markers.length > 0 && (
          <div className="telemetry-item">
            <span>Marker Distance</span>
            <strong>{telemetry.aruco_markers[0].distance_m.toFixed(2)} m</strong>
          </div>
        )}
      </div>

      <div style={{ marginTop: '1.5rem', fontSize: '0.85rem', color: '#666', textAlign: 'center' }}>
        Last update: {new Date(telemetry.timestamp).toLocaleString()}
      </div>
    </div>
  );
}

export default TelemetryDisplay;