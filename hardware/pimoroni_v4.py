#!/usr/bin/env python3
"""
enviro_plus_temp_humidity_airquality.py

Continuously read and print temperature, humidity, and air-quality data
from the Pimoroni Enviro+ HAT’s BME280 and MICS6814 sensors.
"""

import time
from datetime import datetime

from smbus2     import SMBus
from bme280     import BME280
from enviroplus import gas

# === CONFIGURATION ===
I2C_BUS       = 1      # Default I²C bus on Raspberry Pi
READ_INTERVAL = 2.0    # Seconds between readings

# === SETUP ===
bus = SMBus(I2C_BUS)
bme = BME280(i2c_dev=bus)

def read_temp_humidity():
    """
    Reads temperature and humidity from the BME280.
    Returns a tuple (temperature_c, humidity_percent).
    """
    return bme.get_temperature(), bme.get_humidity()

def read_gas():
    """
    Reads the MICS6814 gas sensor.
    Returns a tuple (reducing_ohm, oxidising_ohm, nh3_ohm).
    """
    reading = gas.read_all()
    return reading.reducing, reading.oxidising, reading.nh3

def main():
    print("Enviro+ Temp/Humidity & Air-Quality Monitor (Ctrl+C to exit)\n")
    try:
        while True:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # BME280
            temp_c, hum_pct = read_temp_humidity()

            # Gas sensor
            r_reducing, r_oxidising, r_nh3 = read_gas()

            # Print it all
            print(
                f"[{ts}] "
                f"T={temp_c:.2f}°C  H={hum_pct:.2f}%  "
                f"Gas=[R⁺={r_reducing:.0f}Ω, Oₓ={r_oxidising:.0f}Ω, NH₃={r_nh3:.0f}Ω]"
            )

            time.sleep(READ_INTERVAL)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        bus.close()

if __name__ == "__main__":
    main()
