#!/usr/bin/env python3
"""
enviro_plus_all.py

Continuously read and print all data streams from the Pimoroni Enviro+ HAT,
using /dev/serial0 for the PMS5003 UART.
"""

import time
from datetime import datetime

from smbus2      import SMBus
from bme280      import BME280
from ltr559      import LTR559
from enviroplus  import gas
from pms5003     import PMS5003, ReadTimeoutError

# === CONFIGURATION ===
I2C_BUS       = 1
READ_INTERVAL = 2.0
PMS_PORT      = "/dev/serial0"

def main():
    print("Enviro+ All-Sensors Monitor (Ctrl+C to exit)\n")
    try:
        with SMBus(I2C_BUS) as bus:
            bme = BME280(i2c_dev=bus)
            ltr = LTR559()
            pms = PMS5003(device=PMS_PORT)

            # allow gas sensor heater to stabilise
            time.sleep(5.0)

            while True:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # BME280
                temp_c   = bme.get_temperature()
                hum_pct  = bme.get_humidity()
                pres_hpa = bme.get_pressure()

                # LTR-559
                lux       = ltr.get_lux()
                proximity = ltr.get_proximity()

                # MICS6814 gas (new API)
                gas_reading = gas.read_all()
                r_reducing   = gas_reading.reducing
                r_oxidising  = gas_reading.oxidising
                r_nh3        = gas_reading.nh3

                # PMS5003
                try:
                    p = pms.read()
                    pm1  = p.pm_ug_per_m3(1.0)
                    pm25 = p.pm_ug_per_m3(2.5)
                    pm10 = p.pm_ug_per_m3(10)
                except ReadTimeoutError:
                    pm1 = pm25 = pm10 = None

                print(
                    f"[{ts}] "
                    f"T={temp_c:.2f}°C  H={hum_pct:.2f}%  P={pres_hpa:.2f}hPa  "
                    f"L={lux:.1f}lux  Pr={proximity}  "
                    f"Gas=[R⁺={r_reducing:.0f}Ω,Oₓ={r_oxidising:.0f}Ω,NH₃={r_nh3:.0f}Ω]  "
                    f"PM₁={pm1 or 'N/A'}µg/m³  "
                    f"PM₂.₅={pm25 or 'N/A'}µg/m³  "
                    f"PM₁₀={pm10 or 'N/A'}µg/m³"
                )

                time.sleep(READ_INTERVAL)

    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()
