#!/usr/bin/env python3
import time
import cv2
import os
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
    
    def _get_pi_temperature(self) -> Optional[float]:
        """Get Raspberry Pi CPU temperature."""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millidegrees = int(f.read().strip())
                return temp_millidegrees / 1000.0
        except Exception as e:
            print(f"Failed to read Pi temperature: {e}")
            return None
    
    def _calculate_gas_ppm(self, rs_ohms: float, ro_ohms: float, a: float, b: float) -> float:
        """Convert resistance ratio to PPM using linear calibration."""
        if ro_ohms <= 0:
            return 0.0
        ratio = rs_ohms / ro_ohms
        ppm = a * ratio + b
        return max(ppm, 0.0)  # No negative PPM
    
    def get_readings(self) -> Optional[EnvironmentalData]:
        if not self.bme or not self.ltr:
            return None
        try:
            # Import config here to avoid circular dependency
            import config
            
            # Get gas readings with PPM conversion
            gas_data = None
            if self.gas:
                try:
                    readings = self.gas.read_all()
                    
                    # Calculate PPM values using calibration from display_ip.py
                    reducing_ppm = self._calculate_gas_ppm(
                        readings.reducing, config.RO_RED, config.A_RED, config.B_RED
                    )
                    oxidising_ppm = self._calculate_gas_ppm(
                        readings.oxidising, config.RO_OX, config.A_OX, config.B_OX
                    )
                    nh3_ppm = self._calculate_gas_ppm(
                        readings.nh3, config.RO_NH3, config.A_NH3, config.B_NH3
                    )
                    
                    gas_data = GasReadings(
                        reducing_ohms=readings.reducing,
                        oxidising_ohms=readings.oxidising,
                        nh3_ohms=readings.nh3,
                        reducing_ppm=reducing_ppm,
                        oxidising_ppm=oxidising_ppm,
                        nh3_ppm=nh3_ppm
                    )
                except Exception as e:
                    print(f"Gas sensor read error: {e}")
            
            # Get Pi temperature
            pi_temp = self._get_pi_temperature()
            
            return EnvironmentalData(
                temperature_c=self.bme.get_temperature(),
                pressure_hpa=self.bme.get_pressure(),
                humidity_rh=self.bme.get_humidity(),
                light_lux=self.ltr.get_lux(),
                pi_temperature_c=pi_temp,
                gas_readings=gas_data
            )
        except Exception as e:
            print(f"Environmental sensor error: {e}")
            return None
    
    def get_proximity(self) -> int:
        """Get proximity sensor value for LCD mode switching."""
        if self.ltr:
            try:
                return self.ltr.get_proximity()
            except Exception:
                pass
        return 0


class LCDDisplay:
    """Handles the ST7735 LCD display with 3-tab navigation."""
    
    # Display constants
    TAB_HEIGHT = 15
    TAB_COUNT = 3  # Reduced from 4 to 3
    
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
    
    def set_tab(self, tab_index: int):
        """Set LCD tab programmatically (called from GCS command)."""
        if 0 <= tab_index < self.TAB_COUNT:
            self.current_mode = tab_index
            print(f"LCD tab set to: {self.current_mode} ({self._get_tab_name(self.current_mode)})")
    
    def update_mode(self, proximity: int):
        """Cycle through display modes when proximity sensor detects hand wave."""
        now = time.time()
        if proximity > 1500 and (now - self.last_tap_time) > 0.5:
            self.current_mode = (self.current_mode + 1) % self.TAB_COUNT
            self.last_tap_time = now
            print(f"LCD mode changed to: {self.current_mode} ({self._get_tab_name(self.current_mode)})")

    def _get_tab_name(self, mode: int) -> str:
        """Get display name for each tab."""
        names = ["IP", "CAM", "TEMP"]
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
    
    def update_display(self, ip_address: str, frame_with_detections, env_data, 
                       detections=None, aruco_markers=None):
        """Update the LCD display with current data.
        
        Args:
            ip_address: System IP address
            frame_with_detections: RGB frame with YOLO detections already drawn
            env_data: EnvironmentalData object with all sensor readings
            detections: List of YoloDetection objects (for camera tab)
            aruco_markers: List of ArucoDetection objects (for camera tab)
        """
        if not self.lcd:
            return
        
        try:
            # Store detections for camera tab
            self._last_detections = detections or []
            self._last_aruco_markers = aruco_markers or []
            
            # Clear entire display
            self.draw.rectangle((0, 0, self.lcd.width, self.lcd.height), self.COLOR_BG)
            
            # Draw tab bar
            self._draw_tab_bar()
            
            # Draw content based on current mode
            content_y_start = self.TAB_HEIGHT + 3
            
            if self.current_mode == 0:
                self._draw_ip_tab(content_y_start, ip_address)
            
            elif self.current_mode == 1:
                self._draw_camera_tab(content_y_start, frame_with_detections)
            
            elif self.current_mode == 2:
                self._draw_temp_tab(content_y_start, env_data)
            
            # Display the final image
            self.lcd.display(self.image)
            
        except Exception as e:
            print(f"LCD update error: {e}")
    
    def _draw_ip_tab(self, y_start: int, ip_address: str):
        """Tab 0: IP Address Display"""
        self._draw_content_area()
        
        # Center the text vertically and horizontally in content area
        content_height = self.lcd.height - y_start
        
        # Use larger font for IP display
        font_large = ImageFont.truetype(UserFont, 18)
        
        # Calculate text sizes
        header_bbox = self.draw.textbbox((0, 0), "IP Address:", font=font_large)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        
        ip_bbox = self.draw.textbbox((0, 0), ip_address, font=font_large)
        ip_width = ip_bbox[2] - ip_bbox[0]
        ip_height = ip_bbox[3] - ip_bbox[1]
        
        # Calculate vertical centering
        line_spacing = 10
        total_height = header_height + line_spacing + ip_height
        start_y = y_start + (content_height - total_height) // 2
        
        # Draw header centered horizontally
        header_x = (self.lcd.width - header_width) // 2
        self.draw.text((header_x, start_y), "IP Address:", 
                      fill=(255, 255, 255), font=font_large)
        
        # Draw IP address centered horizontally (all white)
        ip_x = (self.lcd.width - ip_width) // 2
        ip_y = start_y + header_height + line_spacing
        self.draw.text((ip_x, ip_y), ip_address, 
                      fill=(255, 255, 255), font=font_large)

    def _draw_camera_tab(self, y_start: int, frame_with_detections):
        """Tab 1: Live Camera Feed with Detection Overlay + Detection List"""
        import config
        
        if frame_with_detections is not None:
            try:
                # Convert frame to PIL Image
                img_pil = Image.fromarray(cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB))
                
                # Scale to take ~1/3 of width, aligned left
                content_height = self.lcd.height - y_start
                target_width = self.lcd.width // 3
                aspect_ratio = img_pil.width / img_pil.height
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
                
                # Ensure it fits vertically
                if new_height > content_height:
                    new_height = content_height
                    new_width = int(new_height * aspect_ratio)
                
                img_pil = img_pil.resize((new_width, new_height))
                
                # Paste on left side
                self.image.paste(img_pil, (5, y_start))
                
                # Draw detection info on the right side
                text_x = new_width + 15
                text_y = y_start + 5
                line_height = 15
                
                # Get detections from the last update
                if hasattr(self, '_last_detections'):
                    detections = self._last_detections
                    
                    # Count detections by class
                    detection_counts = {}
                    for det in detections:
                        detection_counts[det.label] = detection_counts.get(det.label, 0) + 1
                    
                    # Draw detected classes with colors
                    for label, count in detection_counts.items():
                        color = config.DETECTION_COLOURS.get(label, config.DETECTION_COLOURS["default"])
                        text = f"{label}: {count}"
                        self.draw.text((text_x, text_y), text, 
                                     fill=color, font=self.font_small)
                        text_y += line_height
                
                # Draw ArUco marker info
                if hasattr(self, '_last_aruco_markers') and self._last_aruco_markers:
                    text_y += 5  # Add spacing
                    self.draw.text((text_x, text_y), "ArUco Markers:", 
                                 fill=self.COLOR_HEADER, font=self.font_small)
                    text_y += line_height
                    
                    for marker in self._last_aruco_markers[:3]:  # Show up to 3 markers
                        marker_text = f"ID {marker.marker_id}"
                        self.draw.text((text_x, text_y), marker_text, 
                                     fill=(0, 255, 0), font=self.font_small)
                        text_y += line_height
                        
                        # Show distance
                        dist_text = f"  {marker.distance_m:.2f}m"
                        self.draw.text((text_x, text_y), dist_text, 
                                     fill=self.COLOR_VALUE, font=self.font_small)
                        text_y += line_height
                
            except Exception as e:
                print(f"Camera preview error: {e}")
                self._draw_content_area()
                self.draw.text((5, y_start + 30), "Camera Error", 
                             fill=self.COLOR_ERROR, font=self.font_medium)
        else:
            self._draw_content_area()
            self.draw.text((5, y_start + 30), "No Frame Available", 
                         fill=self.COLOR_ERROR, font=self.font_medium)
    
    def _draw_temp_tab(self, y_start: int, env_data):
        """Tab 2: Temperature Readings (Enviro+ BME280 + Pi CPU)"""
        y = y_start + 10
        line_spacing = 25
        
        if env_data:
            # Enviro+ Temperature
            temp_color = self.COLOR_VALUE
            if env_data.temperature_c > 40:
                temp_color = self.COLOR_WARNING
            elif env_data.temperature_c > 50:
                temp_color = self.COLOR_ERROR
            
            self.draw.text((5, y), f"Temperature: {env_data.temperature_c:.1f} °C", 
                         fill=temp_color, font=self.font_medium)
            y += line_spacing
            
            # Pi CPU Temperature
            if env_data.pi_temperature_c is not None:
                pi_temp_color = self.COLOR_VALUE
                if env_data.pi_temperature_c > 70:
                    pi_temp_color = self.COLOR_WARNING
                elif env_data.pi_temperature_c > 80:
                    pi_temp_color = self.COLOR_ERROR
                
                self.draw.text((5, y), f"RPi CPU Temp: {env_data.pi_temperature_c:.1f} °C", 
                             fill=pi_temp_color, font=self.font_medium)
            else:
                self.draw.text((5, y), "RPi CPU Temp: N/A", 
                             fill=self.COLOR_ERROR, font=self.font_medium)
        else:
            self.draw.text((5, y + 20), "No Temperature Data", 
                         fill=self.COLOR_ERROR, font=self.font_medium)
    
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