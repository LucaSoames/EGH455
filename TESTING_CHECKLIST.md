# Testing Checklist for Audit Logs Enhancement

## Pre-Testing Setup
- [ ] GCS server running on laptop at port 3000
- [ ] Pi connected to same network as laptop
- [ ] config.py has correct `GCS_LAPTOP_IP`
- [ ] Frontend built and deployed (npm run build completed)

## Test 1: Verify Data Model Changes
✅ **PASSED** - Ran test_drill_events.py successfully
- DrillEvent dataclass works
- PayloadData accepts drill_events
- JSON serialization works

## Test 2: Verify Drill Event Queueing (Pi Side)
```bash
# On Pi, test drilling.py changes
cd /home/pi/EGH455/TAIP
python3 -c "
from drilling import DrillController
d = DrillController()
print('Testing event queue...')
print(f'Initial pending events: {d.get_pending_events()}')
d._log_and_queue('Test Event 1', 'First test', 'info', test_id=1)
d._log_and_queue('Test Event 2', 'Second test', 'warning', test_id=2)
events = d.get_pending_events()
print(f'Retrieved {len(events)} events')
for e in events:
    print(f'  - {e.action}: {e.details} [{e.status}]')
events2 = d.get_pending_events()
print(f'After retrieval: {len(events2)} events (should be 0)')
"
```

**Expected Output:**
```
Testing event queue...
Initial pending events: []
Retrieved 2 events
  - Test Event 1: First test [info]
  - Test Event 2: Second test [warning]
After retrieval: 0 events (should be 0)
```

## Test 3: Verify Telemetry Payload Includes Drill Events
```bash
# On Pi, simulate telemetry with drill events
cd /home/pi/EGH455/TAIP
python3 -c "
from data_models import PayloadData, DrillEvent
from datetime import datetime
import json

event = DrillEvent('Test Action', 'Test details', 'info', {'key': 'value'})
payload = PayloadData(
    timestamp=datetime.now().isoformat(),
    gauge_pressure_bar=1.5,
    drill_events=[event]
)

from dataclasses import asdict
print(json.dumps(asdict(payload), indent=2))
"
```

**Expected:** JSON output includes `drill_events` array with one event

## Test 4: Verify GCS Server Logs Drill Events
```bash
# On laptop (GCS server)
# 1. Start server
cd /home/pi/EGH455/TAIP
python3 gcs_server.py --host 0.0.0.0 --port 3000

# 2. In another terminal, send test telemetry with drill event
curl -X POST http://localhost:3000/telemetry \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-01-13T12:00:00",
    "gauge_pressure_bar": 0.5,
    "yolo_detections": [],
    "aruco_markers": [],
    "environmental_data": null,
    "drill_events": [
      {
        "action": "Test Drill Event",
        "details": "Testing drill event reception",
        "status": "info",
        "metadata": {"test": true}
      }
    ]
  }'

# 3. Verify event was logged
curl http://localhost:3000/api/audit/logs?event_type=drill
```

**Expected:** Response includes the test drill event

## Test 5: Verify Frontend Time Range Search
1. **Open web interface:** http://192.168.86.24:3000
2. **Navigate to:** "Audit Logs (Database)" tab
3. **Test quick range buttons:**
   - [ ] Click "Today" → Shows only today's logs
   - [ ] Click "Yesterday" → Shows yesterday's logs
   - [ ] Click "Last 7 Days" → Shows last week's logs
   - [ ] Click "Last 30 Days" → Shows last month's logs
4. **Test manual date range:**
   - [ ] Set start date to 3 days ago, end date to today
   - [ ] Verify logs filtered correctly
5. **Test time range:**
   - [ ] Set start time to 09:00, end time to 17:00
   - [ ] Verify only events in that time range shown
6. **Test clear filters:**
   - [ ] Click "Clear Filters" button
   - [ ] Verify all filters reset

## Test 6: Verify Auto-Refresh (10 seconds)
1. **Open web interface:** Audit Logs tab
2. **Check countdown:**
   - [ ] Observe countdown: (10s), (9s), (8s), ..., (1s), (10s)
   - [ ] Verify refresh happens at 0 seconds
3. **Disable auto-refresh:**
   - [ ] Uncheck "Auto-refresh" checkbox
   - [ ] Verify countdown stops
   - [ ] Verify no automatic refreshes
4. **Re-enable:**
   - [ ] Check "Auto-refresh" checkbox
   - [ ] Verify countdown resumes

## Test 7: End-to-End Drill Event Flow
1. **Start GCS server on laptop:**
   ```bash
   cd /home/pi/EGH455/TAIP
   python3 gcs_server.py --host 0.0.0.0 --port 3000
   ```

2. **Start Pi application:**
   ```bash
   cd /home/pi/EGH455/TAIP
   python3 main.py
   ```

3. **Trigger drill event:**
   - Use test mode with low pressure (<1.0 bar)
   - OR manually test drilling.py

4. **Verify on web interface:**
   - [ ] Open http://192.168.86.24:3000
   - [ ] Go to "Audit Logs (Database)" tab
   - [ ] Filter by event type "Drill"
   - [ ] Verify drill events appear (Phase 1, 2, 3, 4)
   - [ ] Check timestamps are correct
   - [ ] Check status colors (warning=yellow, error=red, success=green)

## Test 8: Database Persistence
```bash
# On GCS laptop, check database directly
cd /home/pi/EGH455/TAIP
sqlite3 audit_logs.db "SELECT COUNT(*) FROM audit_logs WHERE event_type='drill';"
sqlite3 audit_logs.db "SELECT action, details, status FROM audit_logs WHERE event_type='drill' ORDER BY timestamp DESC LIMIT 5;"
```

**Expected:** Drill events visible in database

## Performance Verification
- [ ] Monitor CPU usage during operation (should not increase)
- [ ] Monitor network traffic (no extra requests for drill events)
- [ ] Verify telemetry rate remains at POST_TELEM_HZ (5 Hz)
- [ ] Verify frontend refresh rate is 10 seconds

## Known Issues / Edge Cases
- [ ] Test with empty drill_events (should work with default empty list)
- [ ] Test with many drill events (100+) in queue
- [ ] Test concurrent drill operations (thread safety)
- [ ] Test database size with 10,000+ events (pagination should handle)

## Success Criteria
✅ All tests above pass
✅ Drill events visible in web interface within 10 seconds of occurrence
✅ Time range filtering works correctly
✅ Auto-refresh countdown visible and functional
✅ No performance degradation
✅ No errors in console logs (Pi or server)

## Rollback Plan (if needed)
If issues arise:
1. Revert drilling.py to use direct log_drill() calls
2. Remove drill_events from PayloadData
3. Remove drill event extraction from gcs_server.py
4. Rebuild frontend with old AuditLogsDatabase.tsx
5. Restart services
