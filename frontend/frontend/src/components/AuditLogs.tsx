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

function AuditLogs() {
  const [events, setEvents] = useState<SystemEvent[]>([]);
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
    });

    socket.on('disconnect', () => {
      console.log('AuditLogs: Disconnected from server');
      setConnected(false);
    });

    socket.on('system_event', (event: SystemEvent) => {
      setEvents((prev: SystemEvent[]) => [event, ...prev.slice(0, 99)]); // Keep last 100 events
    });

    // Listen for LCD control events
    socket.on('lcd_tab_update', (data) => {
      const tabNames = ['IP', 'CAM', 'TEMP'];
      setEvents((prev: SystemEvent[]) => [
        {
          id: `lcd_${Date.now()}`,
          type: 'system',
          action: 'LCD Tab Changed',
          details: `LCD display switched to: ${tabNames[data.tab_index] || 'Unknown'} (Tab ${data.tab_index})`,
          timestamp: new Date().toISOString(),
          status: 'info'
        },
        ...prev.slice(0, 99)
      ]); // Keep last 100 events
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

  if (!connected) {
    return (
      <div className="card">
        <p style={{ color: '#e74c3c' }}>⚠️ Not connected to TAIP system</p>
      </div>
    );
  }

  return (
    <div>
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