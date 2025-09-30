#!/usr/bin/env python3
"""
drilling_test_v2.py
Raspberry Pi 5 + venv friendly continuous-rotation servo control on GPIO 13.

This script forces the import of the Pi-5 compatible RPi.GPIO (provided by the
system package `python3-rpi-lgpio`) even when running inside a virtualenv.
It then turns the servo CW for 10 s, stop 1 s, CCW for 10 s, stop, and exits.
"""

import sys, time

# --- Force system site-packages first (RPi.GPIO from python3-rpi-lgpio) ---
# This avoids picking up an old wheel inside the venv.
SYS_DIST = "/usr/lib/python3/dist-packages"
if SYS_DIST not in sys.path:
    sys.path.insert(0, SYS_DIST)

import RPi.GPIO as GPIO  # now resolves to the Pi-5 compatible drop-in

# ===== CONFIG =====
SERVO_PIN = 13       # BCM pin
PWM_FREQUENCY = 50   # Hz

# Duty cycles for continuous rotation (tune for your servo if needed)
STOP_DUTY = 7.5      # ~1.5 ms pulse — stop
CW_DUTY   = 12.5     # ~2.5 ms pulse — full clockwise
CCW_DUTY  = 2.5      # ~0.5 ms pulse — full anticlockwise

# ===== SETUP =====
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
# Claim the line with an explicit initial level (prevents read-before-claim issues)
GPIO.setup(SERVO_PIN, GPIO.OUT, initial=GPIO.LOW)

pwm = GPIO.PWM(SERVO_PIN, PWM_FREQUENCY)
pwm.start(STOP_DUTY)
time.sleep(0.5)  # settle

try:
    print("Clockwise for 10 seconds…")
    pwm.ChangeDutyCycle(CCW_DUTY)
    time.sleep(10)

    print("Stop for 1 second…")
    pwm.ChangeDutyCycle(STOP_DUTY)
    time.sleep(1)

    print("Anticlockwise for 10 seconds…")
    pwm.ChangeDutyCycle(CW_DUTY)
    time.sleep(10)

    print("Stop.")
    pwm.ChangeDutyCycle(STOP_DUTY)
    time.sleep(0.5)

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    print("Cleaning up GPIO…")
    pwm.stop()
    GPIO.cleanup()
