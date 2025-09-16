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
            except RuntimeError as e:
                print(f"GPIO unavailable ({e}). Continuing without drill control.")
                self.gpio_available = False
            except Exception as e:
                print(f"GPIO init error: {e}")
                self.gpio_available = False
    
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
        
        # If no valid reading, reset counter
        if gauge_reading is None:
            self.stable_threshold_count = 0
            self.reading_buffer.clear()
            return
            
        # Add reading to buffer
        self.reading_buffer.append(gauge_reading)
        
        # Use simple average pressure from buffer
        if len(self.reading_buffer) == self.reading_buffer.maxlen:
            avg_pressure = sum(self.reading_buffer) / len(self.reading_buffer)
            
            # Check if average pressure is below threshold
            if avg_pressure < config.DRILL_PRESSURE_THRESHOLD:
                self.stable_threshold_count += 1
            else:
                self.stable_threshold_count = 0
                
            # Activate when we have consistent below-threshold readings
            if (self.stable_threshold_count >= config.DRILL_TRIGGER_COUNT and 
                    self.gpio_available and not self.drill_active):
                self.drill_active = True
                self._start_drilling_sequence()
    
    def _start_drilling_sequence(self):
        """Start the drilling sequence that runs for a fixed duration."""
        try:
            if self.pwm:
                # Activate drill
                print(f"DRILL ACTIVATED - Starting {config.DRILL_DURATION_SEC} second drilling sequence")
                self._set_pwm_duty(config.ACTIVE_DUTY)
                
                # Set up a timer to stop drilling after fixed duration
                self.drill_timer = threading.Timer(config.DRILL_DURATION_SEC, self._complete_drilling)
                self.drill_timer.daemon = True
                self.drill_timer.start()
        except Exception as e:
            print(f"Error starting drill sequence: {e}")
            self._complete_drilling()  # Try to restore to safe state
    
    def _complete_drilling(self):
        """Complete the drilling sequence and reset to stopped position."""
        try:
            if self.pwm:
                # Stop drill
                print("DRILL DEACTIVATED - Drilling sequence completed")
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