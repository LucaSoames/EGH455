#!/usr/bin/env python3

"""
Enviro+ LCD Display Script

This script cycles between three display pages on the LCD:
  1. The Raspberry Pi's IP address
  2. Environmental metrics (temperature, humidity, pressure, light)
  3. Gas sensor readings (CO, NO2, NH3) in PPM, using a calculated calibration

You can cycle between pages by tapping near the proximity sensor on the Enviro+.

USAGE WITH CRONTAB:
-------------------
This script has been set to run automatically at boot using a crontab entry:
    @reboot /home/pi/venvs/depthai_env/bin/python /home/pi/EGH455/display_ip.py

As this script runs in an infinite loop to continuously update the LCD,
you must stop it before running any other script that uses the LCD display.

TO STOP THIS SCRIPT SO YOU CAN USE THE LCD WITH ANOTHER SCRIPT:
---------------------------------------------------------------
1. Edit crontab:
       crontab -e
2. Comment out the @reboot line for display_ip.py by adding a # at the start:
       #@reboot /home/pi/venvs/depthai_env/bin/python /home/pi/EGH455/display_ip.py
3. Save and exit the editor.
4. Reboot the Raspberry Pi:
       sudo reboot
5. You can now safely run your own script that uses the LCD display.

CALIBRATION:
------------
- Before using the gas PPM display, calibrate your sensor in clean air using the provided sensor_calibration.py script.
- Update the RO_RED, RO_OX, RO_NH3 values in this script with your calibration results.
- Use the MiCS-6814 datasheet graphs to estimate Rs/Ro vs PPM for each gas and calculate the A and B coefficients for each channel.

For more details, see the Pimoroni Enviro+ documentation and the MiCS-6814 datasheet.
"""

import time
import socket
import st7735
from fonts.ttf import RobotoMedium as UserFont
from PIL import Image, ImageDraw, ImageFont
from ltr559 import LTR559
from bme280 import BME280
from enviroplus import gas
import sys

# --- Calibration values (replace after calibration)
RO_RED = 451379.96      # Ohms, baseline for reducing gases
RO_OX = 11485.55        # Ohms, baseline for oxidising gases
RO_NH3 = 347942.92      # Ohms, baseline for Ammonia (NH3)

# --- Linear coefficients for y = a*x + b (calculated from datasheet)
# RED - Carbon Monoxide (CO):   ppm = 300 * (Rs/Ro) - 300
# OX - Nitrogen Dioxide (NO₂):  ppm = 0.25 * (Rs/Ro) - 0.25
# NH3 - Ammonia (NH₃):          ppm = -3 * (Rs/Ro) + 3
A_RED, B_RED = 300.0,  -300.0
A_OX,  B_OX  = 0.25,   -0.25
A_NH3, B_NH3 = -3.0,     3.0

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = 'No IP'
    finally:
        s.close()
    return IP

# Wait for a valid IP address (max 30 seconds)
ip = get_ip()
wait_time = 0
while (ip == 'No IP' or ip.startswith('127.')) and wait_time < 30:
    time.sleep(2)
    wait_time += 2
    ip = get_ip()

print(f"IP: {ip}")  # Print to terminal

disp = st7735.ST7735(
    port=0,
    cs=1,
    dc="GPIO9",
    backlight="GPIO12",
    rotation=270,
    spi_speed_hz=10000000
)
disp.begin()
disp.set_backlight(1)  # Ensure backlight is ON

WIDTH = disp.width
HEIGHT = disp.height
font_size = 22
font = ImageFont.truetype(UserFont, font_size)
text_colour1 = (255, 0, 0)      # Red for first line
text_colour2 = (255, 255, 255)  # White for second line
back_colour = (0, 0, 0)         # Black

img = Image.new("RGB", (WIDTH, HEIGHT), color=(0, 0, 0))
draw = ImageDraw.Draw(img)

message1 = "IP Address:"
message2 = ip

# Calculate positions for two lines
x1, y1, x2, y2 = font.getbbox(message1)
size_x1 = x2 - x1
size_y1 = y2 - y1

x3, y3, x4, y4 = font.getbbox(message2)
size_x2 = x4 - x3
size_y2 = y4 - y3

x_msg1 = (WIDTH - size_x1) / 2
x_msg2 = (WIDTH - size_x2) / 2
line_spacing = 16  # Space between lines (adjust as needed)
total_height = size_y1 + line_spacing + size_y2
y_start = (HEIGHT - total_height) / 2

y_msg1 = y_start
y_msg2 = y_start + size_y1 + line_spacing

ltr559 = LTR559()
bme = BME280()

def draw_ip():
    draw.rectangle((0, 0, WIDTH, HEIGHT), back_colour)
    draw.text((x_msg1, y_msg1), message1, font=font, fill=text_colour1)
    draw.text((x_msg2, y_msg2), message2, font=font, fill=text_colour2)
    disp.display(img)

def draw_env():
    temp = bme.get_temperature()
    hum = bme.get_humidity()
    pres = bme.get_pressure()
    light = ltr559.get_lux()
    draw.rectangle((0, 0, WIDTH, HEIGHT), back_colour)
    draw.text((10, 5), f"Temp: {temp:.1f}°C", font=font, fill=text_colour2)
    draw.text((10, 30), f"Hum: {hum:.1f}%", font=font, fill=text_colour2)
    draw.text((10, 55), f"Pres: {pres:.1f}hPa", font=font, fill=text_colour2)
    draw.text((10, 80), f"Light: {light:.1f}lx", font=font, fill=text_colour2)
    disp.display(img)

def ppm_from_rs_ro(rs, ro, a, b):
    ratio = rs / ro if ro else 0
    ppm = a * ratio + b
    return max(ppm, 0)  # No negative ppm

def draw_gas():
    readings = gas.read_all()
    # Calculate PPM using Rs/Ro and linear function
    red_ppm = ppm_from_rs_ro(readings.reducing, RO_RED, A_RED, B_RED)
    ox_ppm = ppm_from_rs_ro(readings.oxidising, RO_OX, A_OX, B_OX)
    nh3_ppm = ppm_from_rs_ro(readings.nh3, RO_NH3, A_NH3, B_NH3)
    draw.rectangle((0, 0, WIDTH, HEIGHT), back_colour)
    draw.text((10, 5), f"RED: {red_ppm:.1f} ppm", font=font, fill=text_colour2)
    draw.text((10, 30), f"OX: {ox_ppm:.1f} ppm", font=font, fill=text_colour2)
    draw.text((10, 55), f"NH₃: {nh3_ppm:.1f} ppm", font=font, fill=text_colour2)
    disp.display(img)

pages = [draw_ip, draw_env, draw_gas]
page = 0
last_tap = 0
tap_delay = 0.7  # seconds

try:
    while True:
        proximity = ltr559.get_proximity()
        if proximity > 1500 and (time.time() - last_tap) > tap_delay:
            page = (page + 1) % len(pages)
            last_tap = time.time()
        pages[page]()
        time.sleep(0.1)
except KeyboardInterrupt:
    disp.set_backlight(0)
    sys.exit(0)
