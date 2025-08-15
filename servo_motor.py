#!/usr/bin/env python3
"""
Continuous-rotation servo control script for DF15RSMG using RPi.GPIO PWM.
Designed for drilling operations in sand with configurable parameters.
"""

import RPi.GPIO as GPIO
import time

# === Constants (edit these to calibrate your setup) ===
SERVO_PIN = 13            # BCM pin for PWM signal #Motor 2 is mapped to GPIO 13 #Motor 1 is mapped to GPIO 12
PWM_FREQUENCY = 50         # PWM frequency in Hz

# Duty cycles for continuous-rotation servo control
STOP_DUTY_CYCLE = 7.5      # % duty cycle to stop servo (~1.5 ms pulse)
MAX_FORWARD_DUTY = 12.5    # % duty cycle for full forward speed (~2.5 ms pulse)
MAX_REVERSE_DUTY = 2.5     # % duty cycle for full reverse speed (~0.5 ms pulse)

# Drilling operation parameters
DRILL_SPEED_PERCENT = 100   # % of max forward speed (0 to 100)
DRILL_DURATION = 5.0        # seconds to run motor during each drill
PAUSE_DURATION = 1.0        # seconds to pause between drills
NUM_DRILLS = 3              # number of drilling cycles

# === GPIO & PWM Setup ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, PWM_FREQUENCY)
pwm.start(STOP_DUTY_CYCLE)  # initialize stopped

# === Helper Functions ===
def speed_to_duty(speed_percent: float) -> float:
    """
    Convert a speed percentage (-100 to +100) to a PWM duty cycle.
    Positive: forward, Negative: reverse.
    """
    # Clamp speed
    speed = max(-100.0, min(100.0, speed_percent))
    if speed >= 0:
        # Linear map 0 -> STOP to 100 -> MAX_FORWARD
        return STOP_DUTY_CYCLE + (MAX_FORWARD_DUTY - STOP_DUTY_CYCLE) * (speed / 100.0)
    else:
        # Linear map -100 -> MAX_REVERSE to 0 -> STOP
        return STOP_DUTY_CYCLE + (STOP_DUTY_CYCLE - MAX_REVERSE_DUTY) * (speed / 100.0)


def set_speed(speed_percent: float) -> None:
    """
    Set servo rotation speed.
    """
    duty = speed_to_duty(speed_percent)
    pwm.ChangeDutyCycle(duty)


def drill_sequence() -> None:
    """
    Perform the defined number of drilling cycles.
    """
    try:
        for cycle in range(1, NUM_DRILLS + 1):
            print(f"Drill cycle {cycle}/{NUM_DRILLS}: Running at {DRILL_SPEED_PERCENT}% speed")
            set_speed(DRILL_SPEED_PERCENT)
            time.sleep(DRILL_DURATION)

            print("Pausing motor")
            set_speed(0)
            time.sleep(PAUSE_DURATION)
    except KeyboardInterrupt:
        print("Drilling interrupted by user")
    finally:
        print("Cleanup: stopping PWM and resetting GPIO")
        pwm.stop()
        GPIO.cleanup()


# === Main Execution ===
if __name__ == "__main__":
    drill_sequence()
