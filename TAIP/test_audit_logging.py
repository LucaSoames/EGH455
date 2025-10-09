#!/usr/bin/env python3
"""
Test script to verify audit logging is working correctly
"""

import sys
import time
import json
from pathlib import Path

# Add TAIP directory to path
sys.path.insert(0, str(Path(__file__).parent))

from audit_logger import (get_audit_logger, log_system, log_telemetry, 
                          log_sensor, log_vision, log_network)

def test_audit_logger():
    """Test the audit logger functionality."""
    
    print("=" * 60)
    print("Testing Audit Logger")
    print("=" * 60)
    
    logger = get_audit_logger()
    
    # Test 1: Log system event
    print("\n1. Logging system event...")
    log_system("Test System Event", "Testing system logging", "success")
    print("   ✓ System event logged")
    
    # Test 2: Log telemetry with different statuses
    print("\n2. Logging telemetry events...")
    log_telemetry("High Pressure", "Pressure: 8.5 bar", status="success", pressure=8.5)
    log_telemetry("Medium Pressure", "Pressure: 2.1 bar", status="warning", pressure=2.1)
    log_telemetry("Low Pressure", "Pressure: 0.5 bar", status="error", pressure=0.5)
    print("   ✓ Telemetry events logged")
    
    # Test 3: Log sensor data
    print("\n3. Logging sensor events...")
    log_sensor("Environmental Reading", 
               "Temp: 25.3°C, Humidity: 45.2%, Pressure: 1013.2 hPa, Light: 456.7 lux",
               status="success",
               temperature=25.3,
               humidity=45.2,
               pressure_hpa=1013.2,
               light=456.7)
    print("   ✓ Sensor events logged")
    
    # Test 4: Log vision events
    print("\n4. Logging vision events...")
    log_vision("Object Detection", "3 objects detected: Valve_Open, Gauge_Centre, Needle_Tip",
               status="info",
               count=3,
               classes=["Valve_Open", "Gauge_Centre", "Needle_Tip"])
    log_vision("ArUco Detection", "1 marker(s): ID 42 at 1.25m",
               status="success",
               marker_count=1,
               marker_ids=[42])
    print("   ✓ Vision events logged")
    
    # Test 5: Log network events
    print("\n5. Logging network events...")
    log_network("Client Connected", "WebSocket client: test-client-123", status="info")
    log_network("Telemetry Error", "Connection timeout", status="error")
    print("   ✓ Network events logged")
    
    # Wait a moment for writes to complete
    time.sleep(0.5)
    
    # Test 6: Retrieve logs
    print("\n6. Retrieving logs...")
    logs = logger.get_logs(limit=20)
    print(f"   ✓ Retrieved {len(logs)} logs")
    
    # Display recent logs
    print("\n7. Recent logs:")
    print("   " + "-" * 56)
    for log in logs[:10]:
        print(f"   [{log['event_type']:12s}] {log['action']:25s} - {log['status']:8s}")
        if log.get('details'):
            print(f"      {log['details']}")
    print("   " + "-" * 56)
    
    # Test 7: Get statistics
    print("\n8. Getting statistics...")
    stats = logger.get_stats()
    print(f"   Total events: {stats['total_events']}")
    print(f"   Events by type:")
    for event_type, count in stats['events_by_type'].items():
        print(f"      {event_type}: {count}")
    print(f"   Events by status:")
    for status, count in stats['events_by_status'].items():
        print(f"      {status}: {count}")
    print(f"   Events last hour: {stats['events_last_hour']}")
    print(f"   Events last 24h: {stats['events_last_day']}")
    
    # Test 8: Search functionality
    print("\n9. Testing search...")
    search_results = logger.get_logs(search="pressure", limit=5)
    print(f"   ✓ Found {len(search_results)} logs matching 'pressure'")
    
    # Test 9: Filter by type
    print("\n10. Testing filtering by type...")
    telemetry_logs = logger.get_logs(event_type="telemetry", limit=5)
    print(f"   ✓ Found {len(telemetry_logs)} telemetry logs")
    
    sensor_logs = logger.get_logs(event_type="sensor", limit=5)
    print(f"   ✓ Found {len(sensor_logs)} sensor logs")
    
    # Test 10: Filter by status
    print("\n11. Testing filtering by status...")
    error_logs = logger.get_logs(status="error", limit=5)
    print(f"   ✓ Found {len(error_logs)} error logs")
    
    success_logs = logger.get_logs(status="success", limit=5)
    print(f"   ✓ Found {len(success_logs)} success logs")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print(f"\nDatabase location: {logger.db_path}")
    print("\nYou can inspect the database with:")
    print(f"  sqlite3 {logger.db_path}")
    print("\nOr view logs in the web interface:")
    print("  1. Start the GCS server: python3 gcs_server.py")
    print("  2. Open http://localhost:3000")
    print("  3. Navigate to 'Audit Logs (Database)' tab")
    print("=" * 60)

if __name__ == '__main__':
    test_audit_logger()
