import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

interface SystemEvent {
  id: string;
  type: 'telemetry' | 'system' | 'drill' | 'camera' | 'sensor' | 'vision';
  action: string;
  details: string;
  timestamp: string;
  status: 'info' | 'warning' | 'error' | 'success';
}

interface SystemStats {
  uptime_seconds: number;
  total_events: number;
  events_last_hour: number;
  system_status: string;
  drill_active: boolean;
  pressure_status: string;
}

function AuditLogs() {
  const [events, setEvents] = useState<SystemEvent[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [connected, setConnected] = useState<boolean>(false);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    // Connect to the same host that served the page
    const socket = io(window.location.origin, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    socket.on('connect', () => {
      console.log('AuditLogs: Connected to server');
      setConnected(true);
      socket.emit('request_system_logs', {});
      socket.emit('request_system_stats', {});
    });

    socket.on('disconnect', () => {
      console.log('AuditLogs: Disconnected from server');
      setConnected(false);
    });

    socket.on('system_event', (event: SystemEvent) => {
      setEvents((prev: SystemEvent[]) => [event, ...prev.slice(0, 99)]); // Keep last 100 events
    });

    socket.on('system_stats', (statsData: SystemStats) => {
      setStats(statsData);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  // Simulate system events based on telemetry updates
  useEffect(() => {
    const socket = io('http://localhost:3000');

    socket.on('telemetry_update', (data: any) => {
      // Create system events based on telemetry data
      const timestamp = new Date().toISOString();
      const newEvents: SystemEvent[] = [];

      // Pressure event
      if (data.gauge_pressure_bar != null) {
        let status: 'info' | 'warning' | 'error' | 'success' = 'info';
        let action = 'Pressure Reading';
        
        if (data.gauge_pressure_bar < 1.0) {
          status = 'error';
          action = 'Critical Pressure';
        } else if (data.gauge_pressure_bar < 3.0) {
          status = 'warning';
          action = 'Low Pressure';
        } else {
          status = 'success';
          action = 'Normal Pressure';
        }

        newEvents.push({
          id: `pressure_${Date.now()}`,
          type: 'telemetry',
          action,
          details: `Gauge pressure: ${data.gauge_pressure_bar.toFixed(2)} bar`,
          timestamp,
          status
        });
      }

      // Environmental events
      if (data.temperature != null) {
        let status: 'info' | 'warning' | 'error' | 'success' = 'info';
        if (data.temperature > 40) {
          status = 'warning';
        } else if (data.temperature < 0) {
          status = 'error';
        } else {
          status = 'info';
        }

        newEvents.push({
          id: `temp_${Date.now()}`,
          type: 'sensor',
          action: 'Temperature Update',
          details: `Temperature: ${data.temperature.toFixed(1)}°C, Humidity: ${data.humidity?.toFixed(1) || 'N/A'}%`,
          timestamp,
          status
        });
      }

      // Add events
      if (newEvents.length > 0) {
        setEvents((prev: SystemEvent[]) => [...newEvents, ...prev.slice(0, 97)]); // Keep total under 100
      }
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return '#27ae60';
      case 'warning': return '#f39c12';
      case 'error': return '#e74c3c';
      default: return '#3498db';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'telemetry': return '📊';
      case 'system': return '⚙️';
      case 'drill': return '🔧';
      case 'camera': return '📷';
      case 'sensor': return '🌡️';
      case 'vision': return '👁️';
      default: return '📝';
    }
  };

  const filteredEvents = events.filter((event: SystemEvent) => 
    filter === 'all' || event.type === filter
  );

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  if (!connected) {
    return (
      <div className="card">
        <p style={{ color: '#e74c3c' }}>⚠️ Not connected to TAIP system</p>
      </div>
    );
  }

  return (
    <div>
      {/* System Statistics */}
      {stats && (
        <div className="card" style={{ marginBottom: '2rem' }}>
          <h3>System Status</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div className="telemetry-item">
              <span>⏱️ Uptime</span>
              <strong>{formatUptime(stats.uptime_seconds)}</strong>
            </div>
            <div className="telemetry-item">
              <span>� Total Events</span>
              <strong>{stats.total_events}</strong>
            </div>
            <div className="telemetry-item">
              <span>🕐 Last Hour</span>
              <strong>{stats.events_last_hour}</strong>
            </div>
            <div className="telemetry-item">
              <span>🔧 Drill Status</span>
              <strong style={{ color: stats.drill_active ? '#e74c3c' : '#27ae60' }}>
                {stats.drill_active ? 'ACTIVE' : 'INACTIVE'}
              </strong>
            </div>
          </div>
        </div>
      )}

      {/* Event Filter */}
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
            <option value="sensor">Sensors</option>
            <option value="vision">Vision</option>
          </select>
          
          <span style={{ marginLeft: 'auto', color: '#666', fontSize: '0.9rem' }}>
            Showing {filteredEvents.length} of {events.length} events
          </span>
        </div>
      </div>

      {/* System Events */}
      <div className="card">
        <h3>System Events</h3>
        
        {filteredEvents.length === 0 ? (
          <p style={{ color: '#666', textAlign: 'center', padding: '2rem' }}>
            {connected ? 'No events to display. System events will appear here as they occur.' : 'Connecting to system...'}
          </p>
        ) : (
          <div style={{ display: 'grid', gap: '0.5rem', maxHeight: '600px', overflowY: 'auto' }}>
            {filteredEvents.map((event) => (
              <div
                key={event.id}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '1rem',
                  padding: '1rem',
                  backgroundColor: '#f8f9fa',
                  borderRadius: '4px',
                  borderLeft: `4px solid ${getStatusColor(event.status)}`
                }}
              >
                <div style={{ fontSize: '1.2rem' }}>
                  {getTypeIcon(event.type)}
                </div>
                
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
                    <strong style={{ color: getStatusColor(event.status) }}>
                      {event.action}
                    </strong>
                    <span style={{ 
                      fontSize: '0.8rem', 
                      color: '#666',
                      backgroundColor: 'white',
                      padding: '0.2rem 0.5rem',
                      borderRadius: '12px',
                      textTransform: 'uppercase'
                    }}>
                      {event.type}
                    </span>
                  </div>
                  
                  <div style={{ fontSize: '0.9rem', color: '#555', marginBottom: '0.5rem' }}>
                    {event.details}
                  </div>
                  
                  <div style={{ fontSize: '0.8rem', color: '#999' }}>
                    {new Date(event.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default AuditLogs;