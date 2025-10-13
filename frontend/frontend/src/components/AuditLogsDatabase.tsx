import React, { useState, useEffect } from 'react';
import './AuditLogsDatabase.css';

interface AuditLog {
  id: number;
  timestamp: string;
  event_type: string;
  action: string;
  details: string;
  status: string;
  metadata?: any;
  created_at: string;
}

interface AuditStats {
  total_events: number;
  events_by_type: { [key: string]: number };
  events_by_status: { [key: string]: number };
  events_last_hour: number;
  events_last_day: number;
}

function AuditLogsDatabase() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [stats, setStats] = useState<AuditStats | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Filters and pagination
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [eventTypeFilter, setEventTypeFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalCount, setTotalCount] = useState<number>(0);
  const logsPerPage = 50;

  // Time range filters
  const [startDate, setStartDate] = useState<string>('');
  const [startTime, setStartTime] = useState<string>('00:00');
  const [endDate, setEndDate] = useState<string>('');
  const [endTime, setEndTime] = useState<string>('23:59');

  // Auto-refresh toggle
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [refreshCountdown, setRefreshCountdown] = useState<number>(10);

  // Fetch logs from the API
  const fetchLogs = async () => {
    setLoading(true);
    setError(null);

    try {
      const offset = (currentPage - 1) * logsPerPage;
      const params = new URLSearchParams({
        limit: logsPerPage.toString(),
        offset: offset.toString(),
      });

      if (searchQuery) params.append('search', searchQuery);
      if (eventTypeFilter !== 'all') params.append('event_type', eventTypeFilter);
      if (statusFilter !== 'all') params.append('status', statusFilter);
      
      // Add time range filters
      if (startDate) {
        const startDateTime = `${startDate}T${startTime}:00`;
        params.append('start_date', startDateTime);
      }
      if (endDate) {
        const endDateTime = `${endDate}T${endTime}:59`;
        params.append('end_date', endDateTime);
      }

      const response = await fetch(`${window.location.origin}/api/audit/logs?${params}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.statusText}`);
      }

      const data = await response.json();
      setLogs(data.logs || []);
      setTotalCount(data.total_count || 0);
    } catch (err: any) {
      setError(err.message);
      console.error('Error fetching logs:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch statistics
  const fetchStats = async () => {
    try {
      const response = await fetch(`${window.location.origin}/api/audit/stats`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch stats: ${response.statusText}`);
      }

      const data = await response.json();
      setStats(data);
    } catch (err: any) {
      console.error('Error fetching stats:', err);
    }
  };

  // Initial load
  useEffect(() => {
    fetchLogs();
    fetchStats();
  }, [currentPage, searchQuery, eventTypeFilter, statusFilter, startDate, startTime, endDate, endTime]);

  // Auto-refresh every 10 seconds if enabled
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchLogs();
      fetchStats();
      setRefreshCountdown(10); // Reset countdown
    }, 10000);

    return () => clearInterval(interval);
  }, [autoRefresh, currentPage, searchQuery, eventTypeFilter, statusFilter, startDate, startTime, endDate, endTime]);

  // Countdown timer for auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const countdown = setInterval(() => {
      setRefreshCountdown(prev => (prev > 0 ? prev - 1 : 10));
    }, 1000);

    return () => clearInterval(countdown);
  }, [autoRefresh]);

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'success': return '#27ae60';
      case 'warning': return '#f39c12';
      case 'error': return '#e74c3c';
      default: return '#3498db';
    }
  };

  const getTypeIcon = (type: string): string => {
    switch (type) {
      case 'telemetry': return '';
      case 'system': return '';
      case 'drill': return '';
      case 'camera': return '';
      case 'sensor': return '';
      case 'vision': return '';
      case 'network': return '';
      case 'error': return '';
      default: return '';
    }
  };

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const totalPages = Math.ceil(totalCount / logsPerPage);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setCurrentPage(1); // Reset to first page on search
    fetchLogs();
  };

  const handleClearFilters = () => {
    setSearchQuery('');
    setEventTypeFilter('all');
    setStatusFilter('all');
    setStartDate('');
    setStartTime('00:00');
    setEndDate('');
    setEndTime('23:59');
    setCurrentPage(1);
  };

  const setQuickRange = (range: 'today' | 'yesterday' | '7days' | '30days') => {
    const now = new Date();
    const endD = now.toISOString().split('T')[0];
    const endT = now.toTimeString().slice(0, 5);
    
    let startD: string;
    const startT = '00:00';
    
    switch (range) {
      case 'today':
        startD = endD;
        break;
      case 'yesterday':
        const yesterday = new Date(now);
        yesterday.setDate(yesterday.getDate() - 1);
        startD = yesterday.toISOString().split('T')[0];
        setEndDate(startD);
        setEndTime('23:59');
        break;
      case '7days':
        const week = new Date(now);
        week.setDate(week.getDate() - 7);
        startD = week.toISOString().split('T')[0];
        break;
      case '30days':
        const month = new Date(now);
        month.setDate(month.getDate() - 30);
        startD = month.toISOString().split('T')[0];
        break;
    }
    
    setStartDate(startD);
    setStartTime(startT);
    if (range !== 'yesterday') {
      setEndDate(endD);
      setEndTime(endT);
    }
    setCurrentPage(1);
  };

  return (
    <div className="audit-logs-database">
      <h2>Audit Logs Database</h2>

      {/* Statistics Dashboard */}
      {stats && (
        <div className="stats-dashboard">
          <div className="stat-card">
            <div className="stat-icon">Total</div>
            <div className="stat-content">
              <div className="stat-label">Total Events</div>
              <div className="stat-value">{stats.total_events.toLocaleString()}</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">Hour</div>
            <div className="stat-content">
              <div className="stat-label">Last Hour</div>
              <div className="stat-value">{stats.events_last_hour}</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">Day</div>
            <div className="stat-content">
              <div className="stat-label">Last 24 Hours</div>
              <div className="stat-value">{stats.events_last_day}</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">OK</div>
            <div className="stat-content">
              <div className="stat-label">Success</div>
              <div className="stat-value">{stats.events_by_status.success || 0}</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">Warn</div>
            <div className="stat-content">
              <div className="stat-label">Warnings</div>
              <div className="stat-value">{stats.events_by_status.warning || 0}</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">Err</div>
            <div className="stat-content">
              <div className="stat-label">Errors</div>
              <div className="stat-value">{stats.events_by_status.error || 0}</div>
            </div>
          </div>
        </div>
      )}

      {/* Filters and Search */}
      <div className="filters-section">
        {/* Time Range Selection */}
        <div className="time-range-section">
          <h3>Time Range</h3>
          <div className="time-range-controls">
            <div className="time-input-group">
              <label>Start Date:</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => {
                  setStartDate(e.target.value);
                  setCurrentPage(1);
                }}
                className="date-input"
              />
              <input
                type="time"
                value={startTime}
                onChange={(e) => {
                  setStartTime(e.target.value);
                  setCurrentPage(1);
                }}
                className="time-input"
              />
            </div>
            <div className="time-input-group">
              <label>End Date:</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => {
                  setEndDate(e.target.value);
                  setCurrentPage(1);
                }}
                className="date-input"
              />
              <input
                type="time"
                value={endTime}
                onChange={(e) => {
                  setEndTime(e.target.value);
                  setCurrentPage(1);
                }}
                className="time-input"
              />
            </div>
          </div>
          <div className="quick-range-buttons">
            <button onClick={() => setQuickRange('today')} className="btn btn-quick-range">
              Today
            </button>
            <button onClick={() => setQuickRange('yesterday')} className="btn btn-quick-range">
              Yesterday
            </button>
            <button onClick={() => setQuickRange('7days')} className="btn btn-quick-range">
              Last 7 Days
            </button>
            <button onClick={() => setQuickRange('30days')} className="btn btn-quick-range">
              Last 30 Days
            </button>
          </div>
        </div>

        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            placeholder="Search logs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
          <button type="submit" className="btn btn-primary">Search</button>
        </form>

        <div className="filter-controls">
          <select
            value={eventTypeFilter}
            onChange={(e) => {
              setEventTypeFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="filter-select"
          >
            <option value="all">All Types</option>
            <option value="telemetry">Telemetry</option>
            <option value="system">System</option>
            <option value="drill">Drill</option>
            <option value="camera">Camera</option>
            <option value="sensor">Sensor</option>
            <option value="vision">Vision</option>
            <option value="network">Network</option>
            <option value="error">Error</option>
          </select>

          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="filter-select"
          >
            <option value="all">All Statuses</option>
            <option value="info">Info</option>
            <option value="success">Success</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
          </select>

          <button onClick={handleClearFilters} className="btn btn-secondary">
            Clear Filters
          </button>

          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh {autoRefresh && `(${refreshCountdown}s)`}
          </label>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}

      {/* Loading Indicator */}
      {loading && <div className="loading-indicator">Loading logs...</div>}

      {/* Logs Table */}
      <div className="table-container">
        <table className="logs-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Time</th>
              <th>Type</th>
              <th>Action</th>
              <th>Details</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {logs.length === 0 ? (
              <tr>
                <td colSpan={6} className="no-data">
                  {loading ? 'Loading...' : 'No logs found'}
                </td>
              </tr>
            ) : (
              logs.map((log) => (
                <tr key={log.id} className={`log-row status-${log.status}`}>
                  <td className="log-id">{log.id}</td>
                  <td className="log-time">{formatTimestamp(log.timestamp)}</td>
                  <td className="log-type">
                    <span className="type-badge">
                      {getTypeIcon(log.event_type)} {log.event_type}
                    </span>
                  </td>
                  <td className="log-action">{log.action}</td>
                  <td className="log-details">{log.details}</td>
                  <td className="log-status">
                    <span
                      className="status-badge"
                      style={{ backgroundColor: getStatusColor(log.status) }}
                    >
                      {log.status}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="pagination">
        <div className="pagination-info">
          Showing {logs.length > 0 ? (currentPage - 1) * logsPerPage + 1 : 0} -{' '}
          {Math.min(currentPage * logsPerPage, totalCount)} of {totalCount} logs
        </div>
        <div className="pagination-controls">
          <button
            onClick={() => setCurrentPage(1)}
            disabled={currentPage === 1}
            className="btn btn-pagination"
          >
            First
          </button>
          <button
            onClick={() => setCurrentPage(currentPage - 1)}
            disabled={currentPage === 1}
            className="btn btn-pagination"
          >
            Previous
          </button>
          <span className="page-indicator">
            Page {currentPage} of {totalPages || 1}
          </span>
          <button
            onClick={() => setCurrentPage(currentPage + 1)}
            disabled={currentPage >= totalPages}
            className="btn btn-pagination"
          >
            Next
          </button>
          <button
            onClick={() => setCurrentPage(totalPages)}
            disabled={currentPage === totalPages}
            className="btn btn-pagination"
          >
            Last
          </button>
        </div>
      </div>
    </div>
  );
}

export default AuditLogsDatabase;
