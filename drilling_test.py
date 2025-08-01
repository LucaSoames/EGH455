#!/usr/bin/env python3
"""
Dual continuous‐rotation servo control script for Raspberry Pi.
Turns motor 1 clockwise for 3 s, then motor 2 clockwise for 3 s,
then motor 1 counter‐clockwise for 3 s. All parameters are configurable.
"""

import RPi.GPIO as GPIO
import time

# === CONFIGURABLE CONSTANTS ===
# GPIO pins (BCM numbering)
MOTOR1_PIN          = 12    # Motor 1 PWM pin
MOTOR2_PIN          = 13    # Motor 2 PWM pin

# PWM settings
PWM_FREQUENCY       = 50.0  # Hz, standard for hobby servos

# Duty-cycle calibration (percent)
STOP_DUTY_CYCLE     = 7.5   # ~1.5 ms pulse = no movement
MAX_FORWARD_DUTY    = 12.5  # ~2.5 ms pulse = full-speed forward
MAX_REVERSE_DUTY    = 2.5   # ~0.5 ms pulse = full-speed reverse

# Movement parameters
CW_SPEED_PERCENT    = 100.0  # clockwise speed (0 to +100)
CCW_SPEED_PERCENT   = -100.0 # counter-clockwise speed (0 to –100)
CW_DURATION         = 3.0    # seconds to run clockwise
CCW_DURATION        = 3.0    # seconds to run counter-clockwise
PAUSE_BETWEEN       = 0.5    # optional pause between actions

# === SETUP ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR1_PIN, GPIO.OUT)
GPIO.setup(MOTOR2_PIN, GPIO.OUT)

pwm1 = GPIO.PWM(MOTOR1_PIN, PWM_FREQUENCY)
pwm2 = GPIO.PWM(MOTOR2_PIN, PWM_FREQUENCY)

# Start both stopped
pwm1.start(STOP_DUTY_CYCLE)
pwm2.start(STOP_DUTY_CYCLE)

def speed_to_duty(speed_percent: float) -> float:
    """
    Map speed_percent in [-100,100] to a duty cycle percentage.
    Positive → forward (CW), negative → reverse (CCW).
    """
    s = max(-100.0, min(100.0, speed_percent))
    if s >= 0:
        # 0→STOP to +100→MAX_FORWARD
        return STOP_DUTY_CYCLE + (MAX_FORWARD_DUTY - STOP_DUTY_CYCLE) * (s / 100.0)
    else:
        # 0→STOP to -100→MAX_REVERSE
        return STOP_DUTY_CYCLE + (STOP_DUTY_CYCLE - MAX_REVERSE_DUTY) * (s / 100.0)

def set_speed(pwm, speed_percent: float) -> None:
    """Apply a speed_percent to the given PWM channel."""
    duty = speed_to_duty(speed_percent)
    pwm.ChangeDutyCycle(duty)

def main():
    try:
        # 1) Motor 1 clockwise
        print(f"Motor 1 → CW at {CW_SPEED_PERCENT}% for {CW_DURATION}s")
        set_speed(pwm1, CW_SPEED_PERCENT)
        time.sleep(CW_DURATION)
        set_speed(pwm1, 0.0)
        time.sleep(PAUSE_BETWEEN)

        # 2) Motor 2 clockwise
        print(f"Motor 2 → CW at {CW_SPEED_PERCENT}% for {CW_DURATION}s")
        set_speed(pwm2, CW_SPEED_PERCENT)
        time.sleep(CW_DURATION)
        set_speed(pwm2, 0.0)
        time.sleep(PAUSE_BETWEEN)

        # 3) Motor 1 counter-clockwise
        print(f"Motor 1 → CCW at {CCW_SPEED_PERCENT}% for {CCW_DURATION}s")
        set_speed(pwm1, CCW_SPEED_PERCENT)
        time.sleep(CCW_DURATION)
        set_speed(pwm1, 0.0)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Cleaning up GPIO and stopping PWM")
        pwm1.stop()
        pwm2.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
