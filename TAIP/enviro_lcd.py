#!/usr/bin/env python3
import time
import cv2
from typing import Optional
from data_models import EnvironmentalData

# Import exactly like the working display_ip.py - NO gpiod imports at all
try:
    import st7735
    from fonts.ttf import RobotoMedium as UserFont
    from PIL import Image, ImageDraw, ImageFont
    from bme280 import BME280
    from ltr559 import LTR559  # Import class directly, not module
    
    IS_ENVIRO_AVAILABLE = True
    print("✓ Enviro+ libraries imported successfully")
except ImportError as e:
    print(f"WARNING: Enviro+ libraries not found. Running without LCD/sensor support.")
    print(f" -> Import error detail: {e}")
    IS_ENVIRO_AVAILABLE = False
    # Stubs to avoid NameError
    Image = None
    ImageDraw = None
    ImageFont = None
    UserFont = None
    st7735 = None
    BME280 = None
    LTR559 = None

class EnvironmentalSensors:
    """Handles Enviro+ board sensors."""
    def __init__(self):
        self.bme = None
        self.ltr = None
        if IS_ENVIRO_AVAILABLE:
            try:
                self.bme = BME280()
                self.ltr = LTR559()  # Instantiate the class directly
                print("✓ Environmental sensors initialised")
            except Exception as e:
                print(f"Environmental sensor init failed: {e}")

    def get_readings(self) -> Optional[EnvironmentalData]:
        if not self.bme or not self.ltr:
            return None
        try:
            return EnvironmentalData(
                temperature_c=self.bme.get_temperature(),
                pressure_hpa=self.bme.get_pressure(),
                humidity_rh=self.bme.get_humidity(),
                light_lux=self.ltr.get_lux()
            )
        except Exception as e:
            print(f"Environmental sensor error: {e}")
            return None

    def get_proximity(self) -> int:
        if not self.ltr:
            return 0
        try:
            return self.ltr.get_proximity()
        except Exception:
            return 0

class LCDDisplay:
    """Handles the ST7735 LCD display on the Enviro+ board."""
    def __init__(self):
        self.lcd = None
        self.current_mode = 0
        self.last_tap_time = 0

        if IS_ENVIRO_AVAILABLE:
            try:
                self.lcd = st7735.ST7735(
                    port=0,
                    cs=1,
                    dc="GPIO9",
                    backlight="GPIO12",
                    rotation=270,
                    spi_speed_hz=10000000
                )
                self.lcd.begin()
                self.lcd.set_backlight(1)
                self.image = Image.new("RGB", (self.lcd.width, self.lcd.height), color=(0, 0, 0))
                self.draw = ImageDraw.Draw(self.image)
                self.font = ImageFont.truetype(UserFont, 14)
                print("✓ LCD display initialised")
            except Exception as e:
                print(f"LCD setup failed: {e}")
                self.lcd = None

    def update_mode(self, proximity: int):
        now = time.time()
        if proximity > 1500 and (now - self.last_tap_time) > 0.5:
            self.current_mode = (self.current_mode + 1) % 3
            self.last_tap_time = now
            print(f"LCD mode changed to: {self.current_mode}")

    def update_display(self, ip_address: str, frame, detections,
                       env_data, gauge_pressure, is_file_mode: bool):
        if not self.lcd:
            return
        try:
            # Clear the drawing canvas
            self.draw.rectangle((0, 0, self.lcd.width, self.lcd.height), (0, 0, 0))

            if self.current_mode == 0:
                # Mode 0: IP Address and System Info
                self.draw.text((5, 5), "IP:", fill=(255, 255, 0), font=self.font)
                self.draw.text((5, 23), ip_address, fill=(255, 255, 255), font=self.font)
                mode_text = "File" if is_file_mode else "Live"
                self.draw.text((5, 41), f"Mode: {mode_text}", fill=(0, 255, 255), font=self.font)
                if gauge_pressure is not None:
                    self.draw.text((5, 59), f"P: {gauge_pressure:.1f} bar",
                                   fill=(255, 128, 0), font=self.font)
                else:
                    self.draw.text((5, 59), "P: N/A", fill=(128, 128, 128), font=self.font)
                
                det_count = len(detections) if detections else 0
                self.draw.text((5, 77), f"Dets: {det_count}", fill=(0, 255, 0), font=self.font)
                
                self.lcd.display(self.image)

            elif self.current_mode == 1:
                # Mode 1: Live Camera Feed
                if frame is not None:
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    img_pil = img_pil.resize((self.lcd.width, self.lcd.height))
                    self.lcd.display(img_pil)
                else:
                    self.draw.text((5, 40), "No Frame", fill=(255, 0, 0), font=self.font)
                    self.lcd.display(self.image)

            elif self.current_mode == 2:
                # Mode 2: Environmental Data
                if env_data:
                    self.draw.text((5, 5), "Environment:", fill=(255, 255, 0), font=self.font)
                    self.draw.text((5, 25), f"Temp: {env_data.temperature_c:.1f}°C",
                                   fill=(255, 255, 255), font=self.font)
                    self.draw.text((5, 45), f"Hum: {env_data.humidity_rh:.1f}%",
                                   fill=(255, 255, 255), font=self.font)
                    self.draw.text((5, 65), f"Press: {env_data.pressure_hpa:.1f} hPa",
                                   fill=(255, 255, 255), font=self.font)
                    self.draw.text((5, 85), f"Light: {env_data.light_lux:.1f} lux",
                                   fill=(255, 255, 255), font=self.font)
                else:
                    self.draw.text((5, 40), "No Env Data", fill=(255, 0, 0), font=self.font)
                
                self.lcd.display(self.image)

        except Exception as e:
            print(f"LCD update error: {e}")

    def close(self):
        """Properly close the LCD and turn off backlight."""
        if self.lcd:
            try:
                # Clear the screen first
                print("Clearing LCD display...")
                if hasattr(self, 'draw') and hasattr(self, 'image'):
                    self.draw.rectangle((0, 0, self.lcd.width, self.lcd.height), (0, 0, 0))
                    self.lcd.display(self.image)
                
                # Turn off backlight
                print("Turning off LCD backlight...")
                self.lcd.set_backlight(0)
                
                # Small delay to ensure commands are processed
                time.sleep(0.1)
                print("✓ LCD closed successfully")
            except Exception as e:
                print(f"Error closing LCD: {e}")
            finally:
                self.lcd = None