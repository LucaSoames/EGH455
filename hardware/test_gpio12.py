# /home/pi/test_gpio12.py
import RPi.GPIO as GPIO, time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(12, GPIO.OUT, initial=GPIO.LOW)  # <- important
for _ in range(4):
    GPIO.output(12, 1); time.sleep(0.5)
    GPIO.output(12, 0); time.sleep(0.5)
GPIO.cleanup()