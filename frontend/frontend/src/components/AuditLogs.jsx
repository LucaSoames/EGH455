import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

const AuditLogs = () => {
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState({
    uptime_seconds: 0,
    total_events: 0,
    events_last_hour: 0,
    drill_active: false
  });
  const [filter, setFilter] = useState('all');
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    // Initialize Socket.IO connection
    const newSocket = io('http://localhost:5000', {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
      timeout: 20000,
    });

    setSocket(newSocket);

    newSocket.on('connect', () => {
      setConnected(true);
      addEvent({
        id: `connect_${Date.now()}`,
        type: 'system',
        message: 'Connected to TAIP system',
        timestamp: new Date().toISOString()
      });
    });

    newSocket.on('disconnect', () => {
      setConnected(false);
      addEvent({
        id: `disconnect_${Date.now()}`,
        type: 'system',
        message: 'Disconnected from TAIP system',
        timestamp: new Date().toISOString()
      });
    });

    // Listen for telemetry data
    newSocket.on('telemetry_data', (data) => {
      const newEvents = [];
      
      // Process telemetry data and create events
      if (data.gauge_pressure_bar !== undefined) {
        const action = data.gauge_pressure_bar > 2.0 ? '⚠️ High pressure detected' : '📊 Pressure reading';
        newEvents.push({
          id: `pressure_${Date.now()}`,
          type: 'telemetry',
          message: `Gauge pressure: ${data.gauge_pressure_bar.toFixed(2)} bar`,
          timestamp: new Date().toISOString(),
          data: { pressure: data.gauge_pressure_bar }
        });
      }

      if (data.temperature !== undefined) {
        newEvents.push({
          id: `temp_${Date.now() + 1}`,
          type: 'telemetry',
          message: `Temperature: ${data.temperature.toFixed(1)}°C`,
          timestamp: new Date().toISOString(),
          data: { temperature: data.temperature }
        });
      }

      if (data.humidity !== undefined) {
        newEvents.push({
          id: `humidity_${Date.now() + 2}`,
          type: 'telemetry',
          message: `Humidity: ${data.humidity.toFixed(1)}%`,
          timestamp: new Date().toISOString(),
          data: { humidity: data.humidity }
        });
      }

      if (newEvents.length > 0) {
        setEvents(prev => [...newEvents, ...prev.slice(0, 97)]); // Keep total under 100
      }
    });

    // Listen for system events
    newSocket.on('system_event', (eventData) => {
      addEvent({
        id: eventData.id || `system_${Date.now()}`,
        type: 'system',
        message: eventData.message || 'System event occurred',
        timestamp: eventData.timestamp || new Date().toISOString(),
        data: eventData.data
      });
    });

    // Listen for drill events
    newSocket.on('drill_event', (eventData) => {
      addEvent({
        id: eventData.id || `drill_${Date.now()}`,
        type: 'drill',
        message: eventData.message || 'Drill event occurred',
        timestamp: eventData.timestamp || new Date().toISOString(),
        data: eventData.data
      });
      
      // Update drill status
      setStats(prev => ({
        ...prev,
        drill_active: eventData.active || false
      }));
    });

    // Listen for camera/detection events
    newSocket.on('detection_event', (eventData) => {
      addEvent({
        id: eventData.id || `detection_${Date.now()}`,
        type: 'detection',
        message: eventData.message || 'Object detected',
        timestamp: eventData.timestamp || new Date().toISOString(),
        data: eventData.data
      });
    });

    // Listen for ArUco events
    newSocket.on('aruco_event', (eventData) => {
      addEvent({
        id: eventData.id || `aruco_${Date.now()}`,
        type: 'aruco',
        message: eventData.message || 'ArUco marker detected',
        timestamp: eventData.timestamp || new Date().toISOString(),
        data: eventData.data
      });
    });

    // Listen for system stats updates
    newSocket.on('system_stats', (statsData) => {
      setStats(prev => ({
        ...prev,
        ...statsData
      }));
    });

    return () => {
      newSocket.close();
    };
  }, []);

  const addEvent = (event) => {
    setEvents(prev => [event, ...prev.slice(0, 99)]); // Keep last 100 events
    setStats(prev => ({
      ...prev,
      total_events: prev.total_events + 1,
      events_last_hour: prev.events_last_hour + 1
    }));
  };

  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getEventIcon = (type) => {
    switch (type) {
      case 'telemetry': return '📊';
      case 'system': return '⚙️';
      case 'drill': return '🔧';
      case 'camera': return '📷';
      case 'detection': return '🎯';
      case 'aruco': return '🏷️';
      case 'error': return '❌';
      default: return '📝';
    }
  };

  const getEventColor = (type) => {
    switch (type) {
      case 'error': return '#e74c3c';
      case 'drill': return '#f39c12';
      case 'detection': return '#9b59b6';
      case 'aruco': return '#3498db';
      case 'system': return '#2ecc71';
      case 'telemetry': return '#34495e';
      default: return '#7f8c8d';
    }
  };

  const filteredEvents = events.filter(event =>
    filter === 'all' || event.type === filter
  );

  if (!connected) {
    return (
      <div className="card">
        <p style={{ color: '#393939ff' }}>⚠️ Not connected to TAIP system</p>
      </div>
    );
  }

  return (
    <div>
      {/* System Status Section */}
      {connected && (
        <div className="card" style={{ marginBottom: '2rem' }}>
          <h3>System Status</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div className="telemetry-item">
              <span>⏱️ Uptime</span>
              <strong>{formatUptime(stats.uptime_seconds)}</strong>
            </div>
            <div className="telemetry-item">
              <span>📊 Total Events</span>
              <strong>{stats.total_events}</strong>
            </div>
            <div className="telemetry-item">
              <span>🕐 Last Hour</span>
              <strong>{stats.events_last_hour}</strong>
            </div>
            <div className="telemetry-item">
              <span>🔧 Drill Status</span>
              <strong style={{ color: stats.drill_active ? '#e74c3c' : '#27ae60' }}>
                {stats.drill_active ? 'ACTIVE' : 'IDLE'}
              </strong>
            </div>
          </div>
        </div>
      )}

      {/* Events Filter and List */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <label style={{ marginRight: '0.5rem' }}>Filter by type:</label>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ddd' }}
          >
            <option value="all">All Events</option>
            <option value="telemetry">Telemetry</option>
            <option value="system">System</option>
            <option value="drill">Drill</option>
            <option value="camera">Camera</option>
            <option value="detection">Detections</option>
            <option value="aruco">ArUco</option>
            <option value="error">Errors</option>
          </select>
          <span style={{ marginLeft: 'auto', fontSize: '0.9em', color: '#666' }}>
            Showing {filteredEvents.length} of {events.length} events
          </span>
        </div>
      </div>

      {/* Events List */}
      <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
        {filteredEvents.length === 0 ? (
          <div className="card">
            <p style={{ textAlign: 'center', color: '#666', fontStyle: 'italic' }}>
              No events to display
            </p>
          </div>
        ) : (
          filteredEvents.map((event) => (
            <div
              key={event.id}
              className="card"
              style={{
                marginBottom: '0.5rem',
                borderLeft: `4px solid ${getEventColor(event.type)}`,
                padding: '0.75rem'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                    <span style={{ fontSize: '1.1em' }}>{getEventIcon(event.type)}</span>
                    <span style={{ 
                      backgroundColor: getEventColor(event.type),
                      color: 'white',
                      padding: '0.2rem 0.5rem',
                      borderRadius: '12px',
                      fontSize: '0.75em',
                      textTransform: 'uppercase',
                      fontWeight: 'bold'
                    }}>
                      {event.type}
                    </span>
                  </div>
                  <p style={{ margin: '0.25rem 0', fontSize: '0.95em' }}>
                    {event.message}
                  </p>
                  {event.data && (
                    <details style={{ marginTop: '0.5rem' }}>
                      <summary style={{ cursor: 'pointer', fontSize: '0.85em', color: '#666' }}>
                        View data
                      </summary>
                      <pre style={{ 
                        backgroundColor: '#f8f9fa',
                        padding: '0.5rem',
                        borderRadius: '4px',
                        fontSize: '0.8em',
                        margin: '0.5rem 0 0 0',
                        overflow: 'auto'
                      }}>
                        {JSON.stringify(event.data, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
                <div style={{ 
                  fontSize: '0.8em',
                  color: '#666',
                  textAlign: 'right',
                  minWidth: '80px'
                }}>
                  {formatTimestamp(event.timestamp)}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AuditLogs;