"""
Drilling Control Module for the EGH455 TAIP Subsystem

This module handles the GPIO-controlled drill activation based on gauge pressure readings.
It provides a clean interface for the main application to control the drill.
"""

import sys
import time
from typing import Optional, Deque
import config
import threading
from collections import deque

# Force system site-packages first (RPi.GPIO from python3-rpi-lgpio)
# This ensures we get the Pi-5 compatible version even when in a virtualenv
SYS_DIST = "/usr/lib/python3/dist-packages"
if SYS_DIST not in sys.path:
    sys.path.insert(0, SYS_DIST)

# Now we can safely import GPIO
try:
    import RPi.GPIO as GPIO
    IS_GPIO_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"WARNING: RPi.GPIO error: {e}. Running without GPIO support.")
    IS_GPIO_AVAILABLE = False

# Import audit logging
try:
    from audit_logger import log_drill, log_system
    from data_models import DrillEvent
    AUDIT_LOGGING_AVAILABLE = True
except ImportError:
    AUDIT_LOGGING_AVAILABLE = False
    print("⚠ GPIO library not available. Drill control will run in simulation mode.")

class DrillController:
    """Handles drill activation via GPIO using PWM for servo control."""
    
    def __init__(self):
        """Initialise GPIO and PWM for drill servo control."""
        self.gpio_available = IS_GPIO_AVAILABLE
        self.drill_active = False
        self.drilling_complete = False
        self.pwm = None
        self.drill_timer = None
        
        # Readings buffer for stability
        self.reading_buffer: Deque[float] = deque(maxlen=3)  # Store 3 consecutive readings
        self.stable_threshold_count = 0  # Count of consistent below-threshold readings
        
        # Queue for drill events to be sent to GCS
        self.pending_events = []
        self.events_lock = threading.Lock()
        
        if self.gpio_available:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                # Initialise with explicit LOW to prevent glitches
                GPIO.setup(config.DRILL_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
                
                # Set up PWM for servo control
                self.pwm = GPIO.PWM(config.DRILL_GPIO_PIN, config.PWM_FREQUENCY)
                # Move to the stop position on initialisation, then stop the signal.
                self.pwm.start(config.STOP_DUTY)
                time.sleep(0.5)  # Give servo time to move to initial position
                self.pwm.ChangeDutyCycle(0) # Stop sending signal to prevent jitter
                
                print(f"✓ GPIO pin {config.DRILL_GPIO_PIN} initialised with PWM for drilling")
                
                if AUDIT_LOGGING_AVAILABLE:
                    self._log_and_queue("Drill System Initialized", 
                                      f"GPIO pin {config.DRILL_GPIO_PIN} ready for drilling control",
                                      "success")
            except RuntimeError as e:
                print(f"GPIO unavailable ({e}). Continuing without drill control.")
                self.gpio_available = False
            except Exception as e:
                print(f"GPIO init error: {e}")
                self.gpio_available = False
    
    def _log_and_queue(self, action: str, details: str, status: str = "info", **metadata):
        """Helper method to log drill events both locally and queue for GCS."""
        if AUDIT_LOGGING_AVAILABLE:
            # Log locally to Pi's database
            log_drill(action, details, status, **metadata)
            
            # Queue for sending to GCS
            event = DrillEvent(action=action, details=details, status=status, metadata=metadata)
            with self.events_lock:
                self.pending_events.append(event)
    
    def get_pending_events(self):
        """Get and clear pending drill events for transmission to GCS."""
        with self.events_lock:
            events = self.pending_events.copy()
            self.pending_events.clear()
            return events
    
    def _set_continuous_pwm(self, duty_cycle: float) -> None:
        """Sets a continuous PWM signal. Used to start and run the drill."""
        if self.pwm:
            self.pwm.ChangeDutyCycle(duty_cycle)

    def _set_pwm_duty(self, duty_cycle: float, duration: float = 0.5) -> None:
        """
        Set PWM duty cycle for a short duration to move the servo, then stop the signal.
        This prevents servo jitter during idle periods.
        """
        if self.pwm:
            self.pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(duration)  # Allow time for servo to move to position
            self.pwm.ChangeDutyCycle(0) # Stop sending signal

    def control_drill(self, gauge_reading: Optional[float]):
        """Control drill activation based on pressure reading using PWM."""
        # If drilling is already active or complete for this cycle, do nothing
        if self.drilling_complete or self.drill_active:
            return
        
        # If no valid reading, we do nothing. The counter persists until a valid
        # reading comes in that is ABOVE the threshold. This makes the trigger
        # more robust to intermittent detection failures.
        if gauge_reading is None:
            return
            
        # A valid reading has been received. Check if it's below the threshold.
        if gauge_reading < config.DRILL_PRESSURE_THRESHOLD:
            self.stable_threshold_count += 1
        else:
            # If the pressure is above the threshold, reset the counter.
            self.stable_threshold_count = 0
                
        # Activate when we have consistent below-threshold readings
        if (self.stable_threshold_count >= config.DRILL_TRIGGER_COUNT and 
                self.gpio_available and not self.drill_active):
            self.drill_active = True
            
            if AUDIT_LOGGING_AVAILABLE:
                self._log_and_queue("Drill Activation Triggered",
                                   f"Pressure {gauge_reading:.2f} bar below threshold {config.DRILL_PRESSURE_THRESHOLD} bar for {config.DRILL_TRIGGER_COUNT} consecutive readings",
                                   "warning",
                                   pressure=gauge_reading,
                                   threshold=config.DRILL_PRESSURE_THRESHOLD,
                                   trigger_count=config.DRILL_TRIGGER_COUNT)
            
            self._start_drilling_sequence()
    
    def _start_drilling_sequence(self):
        """Start the drilling sequence: CW -> Stop -> CCW -> Stop."""
        try:
            if self.pwm:
                # Phase 1: Drill clockwise
                print(f"DRILL ACTIVATED - Phase 1: Drilling CW for {config.DRILL_DURATION_SEC} seconds")
                
                if AUDIT_LOGGING_AVAILABLE:
                    self._log_and_queue("Drill Phase 1: CW Started",
                                       f"Drilling clockwise for {config.DRILL_DURATION_SEC}s",
                                       "error",
                                       phase=1,
                                       direction="CW",
                                       duration=config.DRILL_DURATION_SEC)
                
                self._set_continuous_pwm(config.CW_DUTY)
                
                # Set up timer to transition to stop phase after CW duration
                self.drill_timer = threading.Timer(config.DRILL_DURATION_SEC, self._stop_between_phases)
                self.drill_timer.daemon = True
                self.drill_timer.start()
        except Exception as e:
            print(f"Error starting drill sequence: {e}")
            self._complete_drilling()  # Try to restore to safe state
    
    def _stop_between_phases(self):
        """Stop briefly between CW and CCW phases."""
        try:
            if self.pwm:
                # Stop the drill
                print("DRILL - Phase 2: Stopping briefly between CW and CCW")
                
                if AUDIT_LOGGING_AVAILABLE:
                    self._log_and_queue("Drill Phase 2: Stop Between Phases",
                                       "Brief stop between CW and CCW rotation",
                                       "warning",
                                       phase=2,
                                       direction="STOP")
                
                self._set_pwm_duty(config.STOP_DUTY, duration=1.0)  # Brief stop
                
                # Set up timer to start CCW phase
                self.drill_timer = threading.Timer(0.5, self._start_ccw_phase)
                self.drill_timer.daemon = True
                self.drill_timer.start()
        except Exception as e:
            print(f"Error in stop phase: {e}")
            self._complete_drilling()
    
    def _start_ccw_phase(self):
        """Start the counter-clockwise drilling phase."""
        try:
            if self.pwm:
                # Phase 3: Drill counter-clockwise
                print(f"DRILL - Phase 3: Drilling CCW for {config.DRILL_DURATION_SEC} seconds")
                
                if AUDIT_LOGGING_AVAILABLE:
                    self._log_and_queue("Drill Phase 3: CCW Started",
                                       f"Drilling counter-clockwise for {config.DRILL_DURATION_SEC}s",
                                       "error",
                                       phase=3,
                                       direction="CCW",
                                       duration=config.DRILL_DURATION_SEC)
                
                self._set_continuous_pwm(config.CCW_DUTY)
                
                # Set up timer to complete drilling after CCW duration
                self.drill_timer = threading.Timer(config.DRILL_DURATION_SEC, self._complete_drilling)
                self.drill_timer.daemon = True
                self.drill_timer.start()
        except Exception as e:
            print(f"Error starting CCW phase: {e}")
            self._complete_drilling()
    
    def _complete_drilling(self):
        """Complete the drilling sequence and reset to stopped position."""
        try:
            if self.pwm:
                # Phase 4: Stop and complete
                print("DRILL DEACTIVATED - Phase 4: Drilling sequence completed, returning to stop position")
                
                if AUDIT_LOGGING_AVAILABLE:
                    self._log_and_queue("Drill Phase 4: Sequence Complete",
                                       "Drilling sequence completed successfully, returning to stop position",
                                       "success",
                                       phase=4,
                                       direction="STOP")
                
                self._set_pwm_duty(config.STOP_DUTY)
                
                # Mark drilling as complete for this cycle
                self.drilling_complete = True
                self.drill_active = False
        except Exception as e:
            print(f"Error completing drill sequence: {e}")
    
    def reset_drill_state(self):
        """Reset drill state to allow new drilling cycle."""
        self.drilling_complete = False
        self.stable_threshold_count = 0
        self.reading_buffer.clear()
        
        if AUDIT_LOGGING_AVAILABLE:
            self._log_and_queue("Drill State Reset",
                               "Drill ready for new activation cycle",
                               "info")
    
    def close(self):
        """Clean up GPIO resources."""
        if self.gpio_available:
            try:
                # Cancel any pending timer
                if self.drill_timer and self.drill_timer.is_alive():
                    self.drill_timer.cancel()
                
                if self.pwm:
                    # Ensure servo is in stop position before cleanup
                    self._set_pwm_duty(config.STOP_DUTY)
                    self.pwm.stop()
                
                GPIO.cleanup(config.DRILL_GPIO_PIN)  # Only clean up our specific pin
                print("Drill GPIO resources released")
            except Exception as e:
                print(f"Error during GPIO cleanup: {e}")