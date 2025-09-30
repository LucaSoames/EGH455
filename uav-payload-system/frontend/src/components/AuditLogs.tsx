import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface AuditEvent {
  id: number;
  user_id?: number;
  action: string;
  resource?: string;
  resource_id?: number;
  details?: string;
  ip_address?: string;
  timestamp: string;
}

interface AuditStats {
  total_events: number;
  recent_events_24h: number;
  action_breakdown: Array<{ action: string; count: number }>;
}

function AuditLogs() {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [stats, setStats] = useState<AuditStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);

  // Filters
  const [actionFilter, setActionFilter] = useState<string>('');
  const [resourceFilter, setResourceFilter] = useState<string>('');

  useEffect(() => {
    fetchAuditLogs();
    fetchAuditStats();
  }, [currentPage, actionFilter, resourceFilter]);

  const fetchAuditLogs = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      params.append('page', currentPage.toString());
      params.append('per_page', '20');
      
      if (actionFilter) params.append('action', actionFilter);
      if (resourceFilter) params.append('resource', resourceFilter);
      
      const response = await axios.get(`/api/audit/events?${params}`);
      setEvents(response.data.events);
      setTotalPages(response.data.pagination.pages);
      setError('');
    } catch (err) {
      setError('Failed to load audit logs');
      console.error('Error fetching audit logs:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAuditStats = async () => {
    try {
      const response = await axios.get('/api/audit/stats');
      setStats(response.data);
    } catch (err) {
      console.error('Error fetching audit stats:', err);
    }
  };

  const getActionColor = (action: string) => {
    if (action.includes('login')) return '#3498db';
    if (action.includes('failed')) return '#e74c3c';
    if (action.includes('created')) return '#27ae60';
    if (action.includes('telemetry')) return '#f39c12';
    return '#95a5a6';
  };

  const clearFilters = () => {
    setActionFilter('');
    setResourceFilter('');
    setCurrentPage(1);
  };

  if (loading && events.length === 0) {
    return (
      <div className="card">
        <p>Loading audit logs...</p>
      </div>
    );
  }

  if (error && events.length === 0) {
    return (
      <div className="card">
        <p style={{ color: '#e74c3c' }}>⚠️ {error}</p>
        <button onClick={fetchAuditLogs} className="btn btn-primary">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div>
      {/* Statistics */}
      {stats && (
        <div className="card" style={{ marginBottom: '2rem' }}>
          <h3>Audit Statistics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div className="telemetry-item">
              <span>📊 Total Events</span>
              <strong>{stats.total_events}</strong>
            </div>
            <div className="telemetry-item">
              <span>🕐 Last 24h</span>
              <strong>{stats.recent_events_24h}</strong>
            </div>
          </div>
          
          {stats.action_breakdown.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <strong>Action Breakdown:</strong>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
                {stats.action_breakdown.map(item => (
                  <span
                    key={item.action}
                    style={{
                      padding: '0.25rem 0.5rem',
                      backgroundColor: getActionColor(item.action),
                      color: 'white',
                      borderRadius: '12px',
                      fontSize: '0.8rem'
                    }}
                  >
                    {item.action} ({item.count})
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Filters */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <div>
            <label style={{ marginRight: '0.5rem' }}>Action:</label>
            <select
              value={actionFilter}
              onChange={(e) => {
                setActionFilter(e.target.value);
                setCurrentPage(1);
              }}
              style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ddd' }}
            >
              <option value="">All Actions</option>
              <option value="user_login">User Login</option>
              <option value="user_login_failed">Login Failed</option>
              <option value="user_logout">User Logout</option>
              <option value="telemetry_received">Telemetry Received</option>
            </select>
          </div>

          <div>
            <label style={{ marginRight: '0.5rem' }}>Resource:</label>
            <select
              value={resourceFilter}
              onChange={(e) => {
                setResourceFilter(e.target.value);
                setCurrentPage(1);
              }}
              style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ddd' }}
            >
              <option value="">All Resources</option>
              <option value="user">User</option>
              <option value="telemetry">Telemetry</option>
            </select>
          </div>

          <button onClick={clearFilters} className="btn btn-primary">
            Clear Filters
          </button>

          <button onClick={fetchAuditLogs} className="btn btn-primary" style={{ marginLeft: 'auto' }}>
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Audit Events */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3>Audit Events</h3>
          <span style={{ color: '#666', fontSize: '0.9rem' }}>
            Page {currentPage} of {totalPages}
          </span>
        </div>

        {events.length === 0 ? (
          <p>No audit events found.</p>
        ) : (
          <div style={{ display: 'grid', gap: '0.5rem' }}>
            {events.map((event) => (
              <div
                key={event.id}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  padding: '1rem',
                  backgroundColor: '#f8f9fa',
                  borderRadius: '4px',
                  borderLeft: `4px solid ${getActionColor(event.action)}`
                }}
              >
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
                    <strong style={{ color: getActionColor(event.action) }}>
                      {event.action.replace(/_/g, ' ').toUpperCase()}
                    </strong>
                    {event.resource && (
                      <span style={{ fontSize: '0.9rem', color: '#666' }}>
                        {event.resource}
                        {event.resource_id && `#${event.resource_id}`}
                      </span>
                    )}
                    {event.user_id && (
                      <span style={{ fontSize: '0.9rem', color: '#666' }}>
                        User ID: {event.user_id}
                      </span>
                    )}
                  </div>
                  
                  {event.details && (
                    <div style={{ fontSize: '0.9rem', color: '#555', marginBottom: '0.5rem' }}>
                      {event.details}
                    </div>
                  )}
                  
                  <div style={{ fontSize: '0.8rem', color: '#999' }}>
                    {new Date(event.timestamp).toLocaleString()}
                    {event.ip_address && ` • IP: ${event.ip_address}`}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display: 'flex', justifyContent: 'center', gap: '0.5rem', marginTop: '1rem' }}>
            <button
              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
              className="btn btn-primary"
            >
              ← Previous
            </button>
            <span style={{ padding: '0.75rem 1rem', alignSelf: 'center' }}>
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
              disabled={currentPage === totalPages}
              className="btn btn-primary"
            >
              Next →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default AuditLogs;