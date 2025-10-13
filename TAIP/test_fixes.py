#!/usr/bin/env python3
"""
Quick test to verify the bug fixes:
1. EnvironmentalData with correct field names
2. AuditLogger.log() method usage
"""

import sys
sys.path.insert(0, '/home/pi/EGH455/TAIP')

print("=" * 60)
print("Testing Bug Fixes")
print("=" * 60)

# Test 1: EnvironmentalData with correct fields
print("\n1. Testing EnvironmentalData with correct field names...")
try:
    from data_models import EnvironmentalData, GasReadings
    
    gas = GasReadings(
        reducing_ohms=1000.0,
        oxidising_ohms=2000.0,
        nh3_ohms=3000.0
    )
    
    env_data = EnvironmentalData(
        temperature_c=25.5,
        pressure_hpa=1013.25,
        humidity_rh=60.0,
        light_lux=150.0,
        pi_temperature_c=45.2,
        gas_readings=gas
    )
    
    print(f"   ✓ EnvironmentalData created successfully")
    print(f"   - Temperature: {env_data.temperature_c}°C")
    print(f"   - Pressure: {env_data.pressure_hpa} hPa")
    print(f"   - Humidity: {env_data.humidity_rh}%")
    print(f"   - Light: {env_data.light_lux} lux")
    print(f"   - Pi Temp: {env_data.pi_temperature_c}°C")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: AuditLogger.log() method
print("\n2. Testing AuditLogger.log() method...")
try:
    from audit_logger import get_audit_logger
    
    logger = get_audit_logger()
    
    # Check that log method exists
    if not hasattr(logger, 'log'):
        raise AttributeError("AuditLogger has no 'log' method")
    
    # Try to log a test event
    logger.log(
        event_type='drill',
        action='Test Action',
        details='Test details',
        status='info',
        metadata={'test_key': 'test_value'}
    )
    
    print(f"   ✓ AuditLogger.log() method exists and works")
    print(f"   - Method signature: log(event_type, action, details, status, **metadata)")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verify drill events can be serialized
print("\n3. Testing DrillEvent serialization...")
try:
    from data_models import DrillEvent, PayloadData
    from dataclasses import asdict
    import json
    from datetime import datetime
    
    event = DrillEvent(
        action='Drill Activation Triggered',
        details='Test drill event',
        status='warning',
        metadata={'pressure': 0.8, 'threshold': 1.0}
    )
    
    payload = PayloadData(
        timestamp=datetime.now().isoformat(),
        gauge_pressure_bar=0.8,
        environmental_data=env_data,
        drill_events=[event]
    )
    
    # Serialize to dict and JSON
    payload_dict = asdict(payload)
    payload_json = json.dumps(payload_dict, indent=2)
    
    print(f"   ✓ DrillEvent and PayloadData serialization works")
    print(f"   - Payload has {len(payload.drill_events)} drill event(s)")
    print(f"   - Environmental data included: {payload.environmental_data is not None}")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All fixes verified successfully!")
print("=" * 60)
print("\nBoth errors should now be fixed:")
print("  1. EnvironmentalData now accepts temperature_c, pressure_hpa, etc.")
print("  2. AuditLogger.log() method is correctly called (not log_event)")
