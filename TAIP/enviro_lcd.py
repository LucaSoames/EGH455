#!/usr/bin/env python3
import time
import cv2
from typing import Optional
from data_models import EnvironmentalData, GasReadings

# Import exactly like the working display_ip.py - NO gpiod imports at all
try:
    import st7735
    from fonts.ttf import RobotoMedium as UserFont
    from PIL import Image, ImageDraw, ImageFont
    from bme280 import BME280
    from ltr559 import LTR559
    from enviroplus import gas  # Import gas sensor module
    
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
    gas = None

class EnvironmentalSensors:
    """Handles Enviro+ board sensors."""
    def __init__(self):
        self.bme = None
        self.ltr = None
        self.gas = None
        if IS_ENVIRO_AVAILABLE:
            try:
                self.bme = BME280()
                self.ltr = LTR559()
                self.gas = gas  # Gas sensor module
                print("✓ Environmental sensors initialised (BME280, LTR559, MICS6814)")
            except Exception as e:
                print(f"Environmental sensor init failed: {e}")

    def get_readings(self) -> Optional[EnvironmentalData]:
        if not self.bme or not self.ltr:
            return None
        try:
            # Get gas readings
            gas_data = None
            if self.gas:
                try:
                    readings = self.gas.read_all()
                    gas_data = GasReadings(
                        reducing_ohms=readings.reducing,
                        oxidising_ohms=readings.oxidising,
                        nh3_ohms=readings.nh3
                    )
                except Exception as e:
                    print(f"Gas sensor read error: {e}")
            
            return EnvironmentalData(
                temperature_c=self.bme.get_temperature(),
                pressure_hpa=self.bme.get_pressure(),
                humidity_rh=self.bme.get_humidity(),
                light_lux=self.ltr.get_lux(),
                gas_readings=gas_data
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
    """Handles the ST7735 LCD display with visual tab navigation."""
    
    # Display constants
    TAB_HEIGHT = 15
    TAB_COUNT = 4
    
    # Color scheme
    COLOR_BG = (0, 0, 0)
    COLOR_TAB_ACTIVE = (0, 120, 255)
    COLOR_TAB_INACTIVE = (40, 40, 40)
    COLOR_TAB_TEXT_ACTIVE = (255, 255, 255)
    COLOR_TAB_TEXT_INACTIVE = (100, 100, 100)
    COLOR_HEADER = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_VALUE = (0, 255, 255)
    COLOR_WARNING = (255, 128, 0)
    COLOR_ERROR = (255, 0, 0)
    COLOR_SUCCESS = (0, 255, 0)
    
    def __init__(self):
        self.lcd = None
        self.current_mode = 0
        self.last_tap_time = 0
        self.image = None
        self.draw = None
        self.font_small = None
        self.font_medium = None

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
                self.image = Image.new("RGB", (self.lcd.width, self.lcd.height), color=self.COLOR_BG)
                self.draw = ImageDraw.Draw(self.image)
                
                # Two font sizes for better layout
                self.font_small = ImageFont.truetype(UserFont, 11)
                self.font_medium = ImageFont.truetype(UserFont, 14)
                
                print(f"✓ LCD display initialised ({self.lcd.width}x{self.lcd.height})")
            except Exception as e:
                print(f"LCD setup failed: {e}")
                self.lcd = None

    def update_mode(self, proximity: int):
        """Cycle through display modes when proximity sensor detects hand wave."""
        now = time.time()
        if proximity > 1500 and (now - self.last_tap_time) > 0.5:
            self.current_mode = (self.current_mode + 1) % self.TAB_COUNT
            self.last_tap_time = now
            print(f"LCD mode changed to: {self.current_mode} ({self._get_tab_name(self.current_mode)})")

    def _get_tab_name(self, mode: int) -> str:
        """Get display name for each tab."""
        names = ["Info", "Cam", "Env", "Gas"]
        return names[mode] if mode < len(names) else "?"

    def _draw_tab_bar(self):
        """Draw the tab navigation bar at the top of the display."""
        tab_width = self.lcd.width // self.TAB_COUNT
        
        for i in range(self.TAB_COUNT):
            x_start = i * tab_width
            is_active = (i == self.current_mode)
            
            # Draw tab background
            tab_color = self.COLOR_TAB_ACTIVE if is_active else self.COLOR_TAB_INACTIVE
            self.draw.rectangle(
                [(x_start, 0), (x_start + tab_width - 1, self.TAB_HEIGHT)],
                fill=tab_color
            )
            
            # Draw tab border
            if is_active:
                self.draw.rectangle(
                    [(x_start, 0), (x_start + tab_width - 1, self.TAB_HEIGHT)],
                    outline=(255, 255, 255),
                    width=1
                )
            
            # Draw tab text (centered)
            tab_name = self._get_tab_name(i)
            text_color = self.COLOR_TAB_TEXT_ACTIVE if is_active else self.COLOR_TAB_TEXT_INACTIVE
            
            # Calculate text position for centering
            bbox = self.draw.textbbox((0, 0), tab_name, font=self.font_small)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x_start + (tab_width - text_width) // 2
            text_y = (self.TAB_HEIGHT - text_height) // 2 - 1
            
            self.draw.text((text_x, text_y), tab_name, fill=text_color, font=self.font_small)

    def _draw_content_area(self):
        """Clear the content area below the tab bar."""
        self.draw.rectangle(
            [(0, self.TAB_HEIGHT), (self.lcd.width, self.lcd.height)],
            fill=self.COLOR_BG
        )

    def update_display(self, ip_address: str, frame, detections,
                       env_data, gauge_pressure, is_file_mode: bool):
        """Update the LCD display with current data."""
        if not self.lcd:
            return
        
        try:
            # Clear entire display
            self.draw.rectangle((0, 0, self.lcd.width, self.lcd.height), self.COLOR_BG)
            
            # Draw tab bar
            self._draw_tab_bar()
            
            # Draw content based on current mode
            content_y_start = self.TAB_HEIGHT + 3
            
            if self.current_mode == 0:
                self._draw_info_tab(content_y_start, ip_address, gauge_pressure, 
                                   detections, is_file_mode)
            
            elif self.current_mode == 1:
                self._draw_camera_tab(content_y_start, frame)
            
            elif self.current_mode == 2:
                self._draw_environment_tab(content_y_start, env_data)
            
            elif self.current_mode == 3:
                self._draw_gas_tab(content_y_start, env_data)
            
            # Display the final image
            self.lcd.display(self.image)
            
        except Exception as e:
            print(f"LCD update error: {e}")

    def _draw_info_tab(self, y_start: int, ip_address: str, gauge_pressure, 
                       detections, is_file_mode: bool):
        """Tab 0: System Information"""
        y = y_start
        line_spacing = 18
        
        # IP Address
        self.draw.text((5, y), "IP:", fill=self.COLOR_HEADER, font=self.font_medium)
        y += line_spacing
        self.draw.text((5, y), ip_address, fill=self.COLOR_TEXT, font=self.font_small)
        y += line_spacing
        
        # Mode
        mode_text = "File Mode" if is_file_mode else "Live Camera"
        mode_color = self.COLOR_WARNING if is_file_mode else self.COLOR_SUCCESS
        self.draw.text((5, y), f"Mode: {mode_text}", fill=mode_color, font=self.font_small)
        y += line_spacing
        
        # Gauge Pressure
        if gauge_pressure is not None:
            pressure_text = f"Pressure: {gauge_pressure:.1f} bar"
            pressure_color = self.COLOR_WARNING if gauge_pressure < 3.0 else self.COLOR_SUCCESS
            self.draw.text((5, y), pressure_text, fill=pressure_color, font=self.font_small)
        else:
            self.draw.text((5, y), "Pressure: N/A", fill=self.COLOR_ERROR, font=self.font_small)
        y += line_spacing
        
        # Detection Count
        det_count = len(detections) if detections else 0
        det_color = self.COLOR_SUCCESS if det_count > 0 else self.COLOR_TEXT
        self.draw.text((5, y), f"Detections: {det_count}", fill=det_color, font=self.font_small)

    def _draw_camera_tab(self, y_start: int, frame):
        """Tab 1: Live Camera Feed"""
        if frame is not None:
            try:
                # Convert frame to PIL Image
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Calculate scaling to fit content area
                content_height = self.lcd.height - y_start
                aspect_ratio = img_pil.width / img_pil.height
                
                if aspect_ratio > (self.lcd.width / content_height):
                    # Width-constrained
                    new_width = self.lcd.width
                    new_height = int(new_width / aspect_ratio)
                else:
                    # Height-constrained
                    new_height = content_height
                    new_width = int(new_height * aspect_ratio)
                
                img_pil = img_pil.resize((new_width, new_height))
                
                # Center the image
                x_offset = (self.lcd.width - new_width) // 2
                y_offset = y_start + (content_height - new_height) // 2
                
                # Paste onto black background
                self.image.paste(img_pil, (x_offset, y_offset))
                
            except Exception as e:
                print(f"Camera preview error: {e}")
                self._draw_content_area()
                self.draw.text((5, y_start + 30), "Camera Error", 
                             fill=self.COLOR_ERROR, font=self.font_medium)
        else:
            self._draw_content_area()
            self.draw.text((5, y_start + 30), "No Frame Available", 
                         fill=self.COLOR_ERROR, font=self.font_medium)

    def _draw_environment_tab(self, y_start: int, env_data):
        """Tab 2: Environmental Sensors (BME280 + LTR559)"""
        y = y_start
        line_spacing = 17
        
        if env_data:
            # Temperature
            self.draw.text((5, y), "Temperature:", fill=self.COLOR_HEADER, font=self.font_small)
            self.draw.text((100, y), f"{env_data.temperature_c:.1f}°C", 
                         fill=self.COLOR_VALUE, font=self.font_small)
            y += line_spacing
            
            # Humidity
            self.draw.text((5, y), "Humidity:", fill=self.COLOR_HEADER, font=self.font_small)
            self.draw.text((100, y), f"{env_data.humidity_rh:.1f}%", 
                         fill=self.COLOR_VALUE, font=self.font_small)
            y += line_spacing
            
            # Pressure
            self.draw.text((5, y), "Pressure:", fill=self.COLOR_HEADER, font=self.font_small)
            self.draw.text((100, y), f"{env_data.pressure_hpa:.1f}", 
                         fill=self.COLOR_VALUE, font=self.font_small)
            y += line_spacing
            self.draw.text((100, y), "hPa", fill=self.COLOR_TEXT, font=self.font_small)
            y += line_spacing
            
            # Light
            self.draw.text((5, y), "Light:", fill=self.COLOR_HEADER, font=self.font_small)
            self.draw.text((100, y), f"{env_data.light_lux:.1f} lux", 
                         fill=self.COLOR_VALUE, font=self.font_small)
        else:
            self.draw.text((5, y + 30), "No Environmental Data", 
                         fill=self.COLOR_ERROR, font=self.font_medium)

    def _draw_gas_tab(self, y_start: int, env_data):
        """Tab 3: Gas Sensor Readings (MICS6814)"""
        y = y_start
        line_spacing = 17
        
        if env_data and env_data.gas_readings:
            gas = env_data.gas_readings
            
            # Reducing gases (CO, H2S, NH3)
            self.draw.text((5, y), "Reducing:", fill=self.COLOR_HEADER, font=self.font_small)
            self.draw.text((100, y), f"{gas.reducing_ohms:.0f}Ω", 
                         fill=self.COLOR_VALUE, font=self.font_small)
            y += line_spacing
            self.draw.text((5, y), "(CO,H2S,NH3)", fill=self.COLOR_TEXT, font=self.font_small)
            y += line_spacing + 3
            
            # Oxidising gases (NO2, NO, O3)
            self.draw.text((5, y), "Oxidising:", fill=self.COLOR_HEADER, font=self.font_small)
            self.draw.text((100, y), f"{gas.oxidising_ohms:.0f}Ω", 
                         fill=self.COLOR_VALUE, font=self.font_small)
            y += line_spacing
            self.draw.text((5, y), "(NO2,NO,O3)", fill=self.COLOR_TEXT, font=self.font_small)
            y += line_spacing + 3
            
            # NH3 channel
            self.draw.text((5, y), "NH3:", fill=self.COLOR_HEADER, font=self.font_small)
            self.draw.text((100, y), f"{gas.nh3_ohms:.0f}Ω", 
                         fill=self.COLOR_VALUE, font=self.font_small)
            y += line_spacing
            self.draw.text((5, y), "(NH3,H2,EtOH)", fill=self.COLOR_TEXT, font=self.font_small)
            
        else:
            self.draw.text((5, y + 30), "No Gas Data", 
                         fill=self.COLOR_ERROR, font=self.font_medium)
            y += 50
            self.draw.text((5, y), "Gas sensor warmup", 
                         fill=self.COLOR_TEXT, font=self.font_small)
            y += line_spacing
            self.draw.text((5, y), "takes ~5 minutes", 
                         fill=self.COLOR_TEXT, font=self.font_small)

    def close(self):
        """Properly close the LCD and turn off backlight."""
        if self.lcd:
            try:
                print("Clearing LCD display...")
                if hasattr(self, 'draw') and hasattr(self, 'image'):
                    self.draw.rectangle((0, 0, self.lcd.width, self.lcd.height), (0, 0, 0))
                    self.lcd.display(self.image)
                
                print("Turning off LCD backlight...")
                self.lcd.set_backlight(0)
                
                time.sleep(0.1)
                print("✓ LCD closed successfully")
            except Exception as e:
                print(f"Error closing LCD: {e}")
            finally:
                self.lcd = None