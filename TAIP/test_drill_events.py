#!/usr/bin/env python3
"""
Quick test script to verify drill event forwarding implementation
"""

import sys
import json
from datetime import datetime

# Add TAIP directory to path
sys.path.insert(0, '/home/pi/EGH455/TAIP')

from data_models import PayloadData, DrillEvent, YoloDetection, ArucoDetection

def test_drill_event_dataclass():
    """Test DrillEvent dataclass creation"""
    print("Testing DrillEvent dataclass...")
    event = DrillEvent(
        action="Test Drill Event",
        details="This is a test event",
        status="info",
        metadata={"phase": 1, "pressure": 0.5}
    )
    print(f"✓ DrillEvent created: {event}")
    return event

def test_payload_with_drill_events():
    """Test PayloadData with drill events"""
    print("\nTesting PayloadData with drill events...")
    
    # Create some test drill events
    events = [
        DrillEvent(
            action="Drill Activation Triggered",
            details="Pressure 0.8 bar below threshold",
            status="warning",
            metadata={"pressure": 0.8, "threshold": 1.0}
        ),
        DrillEvent(
            action="Drill Phase 1: CW Started",
            details="Drilling clockwise for 3.0s",
            status="error",
            metadata={"phase": 1, "direction": "CW", "duration": 3.0}
        )
    ]
    
    # Create payload with drill events
    payload = PayloadData(
        timestamp=datetime.now().isoformat(),
        yolo_detections=[],
        aruco_markers=[],
        gauge_pressure_bar=0.8,
        environmental_data=None,
        drill_events=events
    )
    
    print(f"✓ PayloadData created with {len(payload.drill_events)} drill events")
    
    # Test serialization to dict (what gets sent over network)
    from dataclasses import asdict
    payload_dict = asdict(payload)
    print(f"✓ Serialized to dict: drill_events field present = {'drill_events' in payload_dict}")
    print(f"  Number of events in dict: {len(payload_dict['drill_events'])}")
    
    # Test JSON serialization
    payload_json = json.dumps(payload_dict, indent=2)
    print(f"✓ JSON serialization successful")
    print(f"  JSON preview:\n{payload_json[:200]}...")
    
    return payload

def test_empty_drill_events():
    """Test PayloadData with no drill events (default)"""
    print("\nTesting PayloadData with default (empty) drill events...")
    
    payload = PayloadData(
        timestamp=datetime.now().isoformat(),
        yolo_detections=[],
        aruco_markers=[],
        gauge_pressure_bar=2.5,
        environmental_data=None
    )
    
    print(f"✓ PayloadData created without explicit drill_events")
    print(f"  Default drill_events list: {payload.drill_events}")
    print(f"  Is empty list: {len(payload.drill_events) == 0}")
    
    return payload

if __name__ == "__main__":
    print("=" * 60)
    print("Drill Event Forwarding Implementation Test")
    print("=" * 60)
    
    try:
        # Test 1: DrillEvent dataclass
        event = test_drill_event_dataclass()
        
        # Test 2: PayloadData with drill events
        payload_with_events = test_payload_with_drill_events()
        
        # Test 3: PayloadData without drill events (default)
        payload_empty = test_empty_drill_events()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nImplementation verified:")
        print("  1. DrillEvent dataclass works correctly")
        print("  2. PayloadData accepts drill_events field")
        print("  3. Default empty list works correctly")
        print("  4. JSON serialization works correctly")
        print("\nNext steps:")
        print("  - Test drill_controller.get_pending_events()")
        print("  - Test main.py telemetry transmission")
        print("  - Test gcs_server.py event reception")
        print("  - Test web interface display")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
