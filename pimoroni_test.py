#!/usr/bin/env python3
"""
temp_humidity.py

Continuously read and print only temperature and humidity
from the BME280 sensor on a Pimoroni Enviro+ HAT.
"""

import time
from bme280 import BME280
from smbus2 import SMBus
from datetime import datetime

# === CONFIGURATION ===
I2C_BUS       = 1      # Default I²C bus on Raspberry Pi
READ_INTERVAL = 2.0    # Seconds between readings

# === SETUP ===
bus   = SMBus(I2C_BUS)
bme   = BME280(i2c_dev=bus)

def read_temp_humidity():
    """
    Reads temperature and humidity from the BME280.
    Returns a tuple (temperature_c, humidity_percent).
    """
    temp = bme.get_temperature()
    hum  = bme.get_humidity()
    return temp, hum

def main():
    print("Enviro+ Temperature & Humidity Monitor (Ctrl+C to exit)\n")
    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temp_c, hum_pct = read_temp_humidity()
            print(f"[{timestamp}] Temperature: {temp_c:.2f} °C   Humidity: {hum_pct:.2f} %")
            time.sleep(READ_INTERVAL)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        bus.close()

if __name__ == "__main__":
    main()
