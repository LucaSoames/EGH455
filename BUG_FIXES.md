# Bug Fixes Summary

## Issues Reported

### Server Error:
```
[ERROR] Telemetry error: 'AuditLogger' object has no attribute 'log_event'
Traceback (most recent call last):
  File "gcs_server.py", line 244, in receive_telemetry
    audit_logger.log_event(
    ^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'AuditLogger' object has no attribute 'log_event'
```

### Client Error:
```
Environmental sensor error: EnvironmentalData.__init__() got an unexpected keyword argument 'temperature_c'
```

## Root Causes

### Issue 1: Incorrect Method Name
- **File:** `TAIP/gcs_server.py` (line 244)
- **Problem:** Code called `audit_logger.log_event()` but the method is actually named `log()`
- **Cause:** Copy-paste error when implementing drill event forwarding

### Issue 2: Incorrect EnvironmentalData Definition
- **File:** `TAIP/data_models.py`
- **Problem:** EnvironmentalData was simplified to generic field names (`temperature`, `humidity`, `pressure`, `light`, `created_at`) but the rest of the codebase uses specific names (`temperature_c`, `humidity_rh`, `pressure_hpa`, `light_lux`, `pi_temperature_c`, `gas_readings`)
- **Cause:** Incorrect merge or edit that simplified the dataclass

## Fixes Applied

### Fix 1: Corrected Method Call in gcs_server.py
**Line 244 - Changed from:**
```python
audit_logger.log_event(
    event_type='drill',
    action=action,
    details=details,
    status=status,
    **metadata  # Also incorrect - should be metadata=metadata
)
```

**To:**
```python
audit_logger.log(
    event_type='drill',
    action=action,
    details=details,
    status=status,
    metadata=metadata  # Pass as dict parameter, not **kwargs
)
```

**Explanation:** 
- The AuditLogger class has a `log()` method, not `log_event()`
- The method signature is: `log(event_type, action, details, status, metadata=None)`
- Metadata should be passed as a dictionary, not unpacked with **kwargs

### Fix 2: Restored Correct EnvironmentalData Definition
**In data_models.py - Changed from:**
```python
@dataclass
class EnvironmentalData:
    """Environmental data collected from Enviro+ sensors"""
    temperature: float
    humidity: float
    pressure: float
    light: float
    created_at: str
```

**To:**
```python
@dataclass
class EnvironmentalData:
    """Environmental data collected from Enviro+ sensors"""
    temperature_c: float
    pressure_hpa: float
    humidity_rh: float
    light_lux: float
    pi_temperature_c: Optional[float] = None  # Raspberry Pi CPU temperature
    gas_readings: Optional[GasReadings] = None
```

**Explanation:**
- Restored the original field names that match the sensors' actual units
- Added back optional fields for Pi CPU temperature and gas readings
- This matches what `enviro_lcd.py` expects when creating EnvironmentalData objects
- This matches what `gcs_server.py` expects when parsing the data

## Files Modified
1. `TAIP/gcs_server.py` - Fixed audit logger method call
2. `TAIP/data_models.py` - Restored correct EnvironmentalData definition

## Testing
Created and ran `test_fixes.py` which verified:
1. ✅ EnvironmentalData can be created with correct field names
2. ✅ AuditLogger.log() method works correctly
3. ✅ DrillEvent serialization works with environmental data

## Next Steps
1. Restart GCS server on laptop
2. Restart Pi client application
3. Verify both systems work without errors
4. Monitor for drill events appearing in web interface
