import time
import cv2
from typing import Optional

from data_models import EnvironmentalData

# Conditional import for Pimoroni Enviro+ libraries
try:
    from bme280 import BME280
    from ltr559 import LTR559
    import st7735
    from PIL import Image, ImageDraw, ImageFont
    from fonts.ttf import RobotoMedium as UserFont
    IS_ENVIRO_AVAILABLE = True
except Exception as e:
    print("WARNING: Enviro+ libraries not found. Running without LCD/sensor support.")
    print(f" -> Import error detail: {e}")
    IS_ENVIRO_AVAILABLE = False
    # Stubs to avoid NameError if PIL not available
    Image = None
    ImageDraw = None
    ImageFont = None
    UserFont = None
    st7735 = None

class EnvironmentalSensors:
    """Handles Enviro+ board sensors."""
    def __init__(self):
        self.bme = None
        self.ltr = None
        if IS_ENVIRO_AVAILABLE:
            try:
                self.bme = BME280()
                self.ltr = LTR559()
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
        return self.ltr.get_proximity() if self.ltr else 0

class LCDDisplay:
    """Handles the ST7735 LCD display on the Enviro+ board."""
    def __init__(self):
        self.lcd = None
        self.current_mode = 0
        self.last_tap_time = 0

        if IS_ENVIRO_AVAILABLE and st7735 and Image:
            try:
                self.lcd = st7735.ST7735(port=0, cs=1, dc="GPIO9",
                                         backlight="GPIO12", rotation=270)
                self.lcd.begin()
                self.image = Image.new("RGB", (self.lcd.width, self.lcd.height), color=(0, 0, 0))
                self.draw = ImageDraw.Draw(self.image)
                # UserFont is a path-like object exported by Pimoroni fonts package
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

    def update_display(self, ip_address: str, frame, detections,
                       env_data, gauge_pressure, is_file_mode: bool):
        if not self.lcd:
            return
        try:
            self.draw.rectangle((0, 0, self.lcd.width, self.lcd.height), (0, 0, 0))

            if self.current_mode == 0:
                self.draw.text((5, 5), "IP:", fill=(255, 255, 0), font=self.font)
                self.draw.text((5, 23), ip_address, fill=(255, 255, 255), font=self.font)
                mode_text = "File" if is_file_mode else "Live"
                self.draw.text((5, 41), f"Mode: {mode_text}", fill=(0, 255, 255), font=self.font)
                if gauge_pressure is not None:
                    self.draw.text((5, 59), f"P: {gauge_pressure:.1f} bar",
                                   fill=(255, 128, 0), font=self.font)

            elif self.current_mode == 1 and frame is not None:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_pil = img_pil.resize((self.lcd.width, self.lcd.height))
                self.image = img_pil

            elif self.current_mode == 2 and env_data:
                self.draw.text((5, 5), "Environment:", fill=(255, 255, 0), font=self.font)
                self.draw.text((5, 25), f"Temp: {env_data.temperature_c:.1f}°C",
                               fill=(255, 255, 255), font=self.font)
                self.draw.text((5, 45), f"Hum: {env_data.humidity_rh:.1f}%",
                               fill=(255, 255, 255), font=self.font)
                self.draw.text((5, 65), f"Press: {env_data.pressure_hpa:.1f} hPa",
                               fill=(255, 255, 255), font=self.font)
                self.draw.text((5, 85), f"Light: {env_data.light_lux:.1f} lux",
                               fill=(255, 255, 255), font=self.font)

            self.lcd.display(self.image)
        except Exception as e:
            print(f"LCD update error: {e}")

    def close(self):
        if self.lcd:
            try:
                self.lcd.set_backlight(0)
            except Exception:
                pass