"""
Sensor Calibration Script for Enviro+ MiCS-6814
- Place the sensor in clean air (outdoors or well-ventilated room).
- Run this script for at least 1 minute.
- The script will average the resistance readings and print recommended Ro values.
- Use the datasheet graphs to estimate Rs/Ro vs PPM for each gas.
- Calculate the constants A and B for the linear function y = Ax + B, where PPM = A * (Rs/Ro) + B.
- Update the constants in `display_ip.py` with the calculated values for PPM estimation.

The MiCS-6814 datasheet provides Rs/Ro vs PPM graphs for each gas. See:
https://cdn.sparkfun.com/datasheets/Sensors/Biometric/MiCS-6814.pdf
"""

import time
from enviroplus import gas

DURATION = 60  # seconds to average readings
INTERVAL = 1   # seconds between readings
WARMUP = 10 * 60  # minutes to ignore at start

red_list = []
ox_list = []
nh3_list = []

print("Calibrating... Please keep the sensor in clean air.")
print(f"Collecting data for {DURATION} seconds.")

start = time.time()
warmup_done = False
while time.time() - start < DURATION + WARMUP:
    readings = gas.read_all()
    elapsed = time.time() - start
    if not warmup_done and elapsed > WARMUP:
        print("\nWarmup finished. Recording calibration data now...\n")
        warmup_done = True
    if elapsed > WARMUP:
        red_list.append(readings.reducing)
        ox_list.append(readings.oxidising)
        nh3_list.append(readings.nh3)
    print(f"Reducing: {readings.reducing:.2f}, Oxidising: {readings.oxidising:.2f}, NH3: {readings.nh3:.2f}")
    time.sleep(INTERVAL)

ro_red = sum(red_list) / len(red_list)
ro_ox = sum(ox_list) / len(ox_list)
ro_nh3 = sum(nh3_list) / len(nh3_list)

print("\nCalibration complete!")
print(f"Recommended baseline resistances (Ro) for clean air:")
print(f"RO_RED = {ro_red:.2f}  # Ohms, reducing gases")
print(f"RO_OX  = {ro_ox:.2f}  # Ohms, oxidising gases")
print(f"RO_NH3 = {ro_nh3:.2f}  # Ohms, NH3")