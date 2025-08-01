# pimoroni_data.py
#!/usr/bin/env python3

"""
enviro_plus_temp_humidity_airquality_light.py

Continuously read and print temperature, humidity, air-quality, and light
from the Pimoroni Enviro+ HAT’s BME280, MICS6814, and LTR-559 sensors,
and store readings in the SQLite database.
"""

import time
from datetime import datetime

from smbus2     import SMBus
from bme280     import BME280
from enviroplus import gas
from ltr559     import LTR559

import db
db.init_db()

# === CONFIGURATION ===
I2C_BUS       = 1      # Default I²C bus on Raspberry Pi
READ_INTERVAL = 2.0    # Seconds between readings

# === SETUP ===
bus = SMBus(I2C_BUS)
bme = BME280(i2c_dev=bus)
ltr = LTR559()           # instantiate light/proximity sensor

def read_temp_humidity():
    """Return (°C, %RH) from BME280."""
    return bme.get_temperature(), bme.get_humidity()

def read_gas():
    """Return (reducing, oxidising, nh3) resistances (Ω) from MICS6814."""
    reading = gas.read_all()
    return reading.reducing, reading.oxidising, reading.nh3

def read_light():
    """Return (lux, proximity) from LTR-559."""
    return ltr.get_lux(), ltr.get_proximity()

def main():
    print("Enviro+ Temp/Humidity, Air-Quality & Light Monitor (Ctrl+C to exit)\n")
    try:
        while True:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Read sensors
            temp_c, hum_pct = read_temp_humidity()
            r_reducing, r_oxidising, r_nh3 = read_gas()
            lux, proximity = read_light()

            # Save to DB
            db.insert_reading(
                ts,
                temp_c,
                hum_pct,
                r_reducing,
                r_oxidising,
                r_nh3,
                lux,
                proximity
            )

            # Print to console
            print(
                f"[{ts}] "
                f"T={temp_c:.2f}°C  H={hum_pct:.2f}%  "
                f"Gas=[R⁺={r_reducing:.0f}Ω, Oₓ={r_oxidising:.0f}Ω, NH₃={r_nh3:.0f}Ω]  "
                f"Light={lux:.1f} lux  Pr={proximity}"
            )

            time.sleep(READ_INTERVAL)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        bus.close()

if __name__ == "__main__":
    main()
