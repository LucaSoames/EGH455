#!/usr/bin/env python3
import pigpio, time

PIN1, PIN2 = 12, 13       # you can use 12/13 again here
pi = pigpio.pi(); assert pi.connected

def stop(pin): pi.set_servo_pulsewidth(pin, 1500)  # tune stop ~1500 µs

try:
    pi.set_servo_pulsewidth(PIN1, 2500); time.sleep(3)  # M1 CW
    stop(PIN1); time.sleep(0.5)

    pi.set_servo_pulsewidth(PIN2, 2500); time.sleep(3)  # M2 CW
    stop(PIN2); time.sleep(0.5)

    pi.set_servo_pulsewidth(PIN1,  500); time.sleep(3)  # M1 CCW
    stop(PIN1)
finally:
    pi.set_servo_pulsewidth(PIN1, 0)
    pi.set_servo_pulsewidth(PIN2, 0)
    pi.stop()
