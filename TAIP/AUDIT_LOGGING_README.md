# Audit Logging System

## Overview

The TAIP system now includes a comprehensive audit logging system that stores all system events in a SQLite database with timestamps. This provides persistent logging that survives system restarts and enables advanced searching and filtering capabilities.

## Features

### Database Backend
- **SQLite Database**: All audit logs are stored in `/home/pi/EGH455/TAIP/audit_logs.db`
- **Indexed Queries**: Fast searching by timestamp, event type, and status
- **Persistent Storage**: Logs survive system restarts
- **Thread-Safe**: Safe for concurrent access from multiple threads

### Event Types
- `telemetry`: Telemetry data events (pressure readings, etc.)
- `system`: System events (startup, shutdown, configuration changes)
- `drill`: Drill controller events
- `camera`: Camera-related events
- `sensor`: Environmental sensor events
- `vision`: Vision processing events (YOLO, ArUco)
- `network`: Network communication events
- `error`: Error events

### Event Statuses
- `info`: Informational events
- `success`: Successful operations
- `warning`: Warning conditions
- `error`: Error conditions

## Backend API

### Audit Logger Module (`audit_logger.py`)

```python
from audit_logger import log_system, log_telemetry, log_vision, log_error

# Log a system event
log_system("System Started", "TAIP initialized successfully", "success")

# Log telemetry with metadata
log_telemetry("Pressure Reading", "Gauge pressure: 5.2 bar", pressure=5.2, status="info")

# Log a vision event
log_vision("Object Detection", "3 objects detected", count=3)

# Log an error
log_error("Camera Error", "Failed to initialize OAK-D")
```

### REST API Endpoints

#### Get Audit Logs
```
GET /api/audit/logs?limit=50&offset=0&event_type=telemetry&status=error&search=pressure
```

**Query Parameters:**
- `limit`: Number of logs to return (default: 100)
- `offset`: Number of logs to skip for pagination (default: 0)
- `event_type`: Filter by event type (optional)
- `status`: Filter by status (optional)
- `search`: Search in action and details fields (optional)
- `start_date`: Filter logs after this date in ISO format (optional)
- `end_date`: Filter logs before this date in ISO format (optional)

**Response:**
```json
{
  "logs": [
    {
      "id": 1,
      "timestamp": "2025-10-09T10:30:45.123456",
      "event_type": "telemetry",
      "action": "Pressure Reading",
      "details": "Gauge pressure: 5.2 bar",
      "status": "info",
      "metadata": {"pressure": 5.2},
      "created_at": "2025-10-09T10:30:45.123456"
    }
  ],
  "total_count": 1,
  "limit": 50,
  "offset": 0
}
```

#### Get Statistics
```
GET /api/audit/stats
```

**Response:**
```json
{
  "total_events": 1234,
  "events_by_type": {
    "telemetry": 456,
    "system": 123,
    "vision": 234
  },
  "events_by_status": {
    "info": 1000,
    "warning": 150,
    "error": 84
  },
  "events_last_hour": 45,
  "events_last_day": 567
}
```

#### Clear Old Logs
```
POST /api/audit/clear
Content-Type: application/json

{
  "days": 30
}
```

Deletes logs older than the specified number of days.

## Frontend

### Audit Logs Database Component

The new `AuditLogsDatabase` component provides:

1. **Statistics Dashboard**
   - Total events count
   - Events in last hour/24 hours
   - Breakdown by status (success, warning, error)

2. **Search & Filtering**
   - Text search across action and details
   - Filter by event type
   - Filter by status
   - Clear all filters button

3. **Data Table**
   - Sortable columns
   - Color-coded status indicators
   - Event type icons
   - Timestamps in local format

4. **Pagination**
   - Navigate through large result sets
   - Page numbers and navigation buttons
   - Shows current range of results

5. **Auto-Refresh**
   - Optional auto-refresh every 5 seconds
   - Toggle on/off

### Accessing the UI

1. Start the GCS server:
   ```bash
   cd TAIP
   python3 gcs_server.py
   ```

2. Open browser: `http://localhost:3000`

3. Navigate to the **"Audit Logs (Database)"** tab

## Integration Points

### GCS Server (`gcs_server.py`)
- Logs telemetry reception events
- Logs LCD control commands
- Logs client connections/disconnections
- Logs errors

### Main Application (`main.py`)
- Logs system startup/shutdown
- Logs mode selection (live/test)
- Logs initialization events

### GCS Client (`gcs_client.py`)
- Ready for client-side logging (not yet implemented)

## Database Management

### Location
The database is stored at: `/home/pi/EGH455/TAIP/audit_logs.db`

### Manual Inspection
```bash
sqlite3 /home/pi/EGH455/TAIP/audit_logs.db

# View recent logs
SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT 10;

# Count by event type
SELECT event_type, COUNT(*) FROM audit_logs GROUP BY event_type;

# Search for errors
SELECT * FROM audit_logs WHERE status='error' ORDER BY created_at DESC;
```

### Cleanup
Use the API endpoint or run manually:
```python
from audit_logger import get_audit_logger

logger = get_audit_logger()
deleted = logger.clear_old_logs(days=30)  # Delete logs older than 30 days
print(f"Deleted {deleted} old logs")
```

## Testing

### Test the Audit Logger Module
```bash
cd /home/pi/EGH455/TAIP
python3 audit_logger.py
```

This will create a test database with sample events and demonstrate all functionality.

### Test the API
```bash
# Start the server
python3 gcs_server.py

# In another terminal, test the endpoints
curl http://localhost:3000/api/audit/logs
curl http://localhost:3000/api/audit/stats
```

## Performance Considerations

- **Indexes**: Database has indexes on timestamp, event_type, status, and created_at for fast queries
- **Thread-Safe**: Uses locks to ensure safe concurrent access
- **Connection Pooling**: Uses context managers for efficient connection handling
- **Pagination**: API supports pagination to handle large result sets
- **Auto-Cleanup**: Can automatically delete old logs to manage database size

## Future Enhancements

Potential improvements:
1. Export logs to CSV/JSON
2. Advanced filtering (date ranges, multiple types)
3. Real-time log streaming via WebSocket
4. Log retention policies
5. Alert notifications for critical events
6. Grafana/visualization integration
