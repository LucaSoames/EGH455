#!/usr/bin/env python3
"""
Hardware Data Bridge Service
Connects Raspberry Pi hardware sensors to the main UAV backend system.

This service:
1. Reads sensor data from the local SQLite database
2. Transforms it to match backend TelemetryData format
3. Sends data to the main backend via HTTP API
4. Handles video streaming integration
5. Manages real-time data synchronization
"""

import json
import time
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import threading
import logging
import os

# Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
HARDWARE_DB_PATH = "sensor_data.db"
UAV_ID = int(os.environ.get("UAV_ID", "1"))  # Default UAV ID
MISSION_ID = int(os.environ.get("MISSION_ID", "1"))  # Default Mission ID
BRIDGE_INTERVAL = 2.0  # seconds between data transmissions
MAX_RETRIES = 3
RETRY_DELAY = 5.0

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HardwareBridge:
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
        self.last_processed_id = 0
        self.running = False
        self.auth_token = None
        
    def authenticate(self):
        """Authenticate with the backend system"""
        try:
            auth_data = {
                "username": os.environ.get("BRIDGE_USERNAME", "admin"),
                "password": os.environ.get("BRIDGE_PASSWORD", "admin123")
            }
            
            response = self.session.post(
                f"{BACKEND_URL}/api/auth/login",
                json=auth_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.auth_token = data['data']['access_token']
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.auth_token}'
                    })
                    logger.info("Successfully authenticated with backend")
                    return True
                    
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            
        return False
    
    def get_connection(self):
        """Get SQLite connection to hardware database"""
        try:
            conn = sqlite3.connect(HARDWARE_DB_PATH, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to hardware database: {e}")
            return None
    
    def get_new_sensor_readings(self) -> list:
        """Get new sensor readings since last processed ID"""
        conn = self.get_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM readings WHERE id > ? ORDER BY id ASC",
                (self.last_processed_id,)
            )
            readings = cursor.fetchall()
            
            if readings:
                self.last_processed_id = readings[-1]['id']
                
            return [dict(row) for row in readings]
            
        except Exception as e:
            logger.error(f"Failed to fetch sensor readings: {e}")
            return []
        finally:
            conn.close()
    
    def convert_gas_to_ppm(self, gas_reducing: float, gas_oxidising: float, gas_nh3: float) -> Dict[str, float]:
        """Convert gas resistance values to approximate PPM concentrations"""
        # Simplified conversion - in production, use calibrated formulas
        # These are rough approximations based on typical MICS6814 characteristics
        
        # CO (reducing gas) - typical range 1-1000 ppm
        co_ppm = max(0, 1000 - (gas_reducing - 10000) / 100) if gas_reducing > 0 else 0
        
        # NO2 (oxidising gas) - typical range 0.1-10 ppm  
        no2_ppm = max(0, (20000 - gas_oxidising) / 1000) if gas_oxidising > 0 else 0
        
        # NH3 (ammonia) - typical range 1-500 ppm
        nh3_ppm = max(0, 500 - (gas_nh3 - 5000) / 50) if gas_nh3 > 0 else 0
        
        # Estimate CO2 based on combined readings (very rough approximation)
        co2_ppm = min(5000, max(400, 400 + (co_ppm * 2) + (no2_ppm * 50)))
        
        return {
            "co": round(co_ppm, 1),
            "co2": round(co2_ppm, 1),
            "no2": round(no2_ppm, 2),
            "nh3": round(nh3_ppm, 1),
            "pm25": 0.0,  # Not measured by current hardware
            "pm10": 0.0   # Not measured by current hardware
        }
    
    def calculate_aqi(self, air_quality: Dict[str, float]) -> int:
        """Calculate Air Quality Index from pollutant concentrations"""
        # Simplified AQI calculation based on EPA standards
        aqi_values = []
        
        # CO AQI (8-hour average simulation)
        co = air_quality.get('co', 0)
        if co <= 4.4:
            co_aqi = co * 50 / 4.4
        elif co <= 9.4:
            co_aqi = 50 + (co - 4.4) * 50 / 5
        elif co <= 12.4:
            co_aqi = 100 + (co - 9.4) * 50 / 3
        else:
            co_aqi = min(300, 150 + (co - 12.4) * 100 / 15.4)
        aqi_values.append(co_aqi)
        
        # NO2 AQI (1-hour average)
        no2 = air_quality.get('no2', 0)
        if no2 <= 0.053:
            no2_aqi = no2 * 50 / 0.053
        elif no2 <= 0.1:
            no2_aqi = 50 + (no2 - 0.053) * 50 / 0.047
        else:
            no2_aqi = min(200, 100 + (no2 - 0.1) * 100 / 0.36)
        aqi_values.append(no2_aqi)
        
        return int(max(aqi_values)) if aqi_values else 0
    
    def transform_sensor_data(self, reading: Dict) -> Dict:
        """Transform hardware sensor reading to backend telemetry format"""
        try:
            # Parse timestamp
            timestamp_str = reading['timestamp']
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            
            # Convert gas readings to PPM
            air_quality = self.convert_gas_to_ppm(
                reading['gas_reducing'],
                reading['gas_oxidising'], 
                reading['gas_nh3']
            )
            
            # Calculate AQI
            aqi = self.calculate_aqi(air_quality)
            
            # Environmental data
            environmental = {
                "temperature": round(reading['temperature'], 2),
                "humidity": round(reading['humidity'], 2),
                "pressure": 1013.25,  # Standard pressure - add barometric sensor if needed
                "light_intensity": round(reading['light'], 1),
                "proximity": reading['proximity']
            }
            
            # Create telemetry data payload
            telemetry_data = {
                "uav_id": UAV_ID,
                "mission_id": MISSION_ID,
                "latitude": -27.4975,  # Example coordinates - replace with actual GPS
                "longitude": 153.0137,
                "altitude": 50.0,      # Example altitude
                "heading": 0.0,        # Example heading
                "speed": 0.0,          # Stationary for ground testing
                "vertical_speed": 0.0,
                "battery_level": 85.0, # Example battery level
                "signal_strength": -45.0, # Example signal strength
                "gps_satellites": 12,
                "system_status": "normal",
                "temperature": environmental["temperature"],
                "wind_speed": 0.0,
                "wind_direction": 0.0,
                "timestamp": timestamp.isoformat(),
                
                # Extended data for hardware integration
                "air_quality_data": {
                    **air_quality,
                    "aqi": aqi,
                    "gas_readings_raw": {
                        "reducing": reading['gas_reducing'],
                        "oxidising": reading['gas_oxidising'],
                        "nh3": reading['gas_nh3']
                    }
                },
                
                "environmental_data": environmental,
                
                "hardware_status": {
                    "sensors_online": True,
                    "camera_online": True,
                    "servo_online": True,
                    "last_reading_id": reading['id']
                }
            }
            
            return telemetry_data
            
        except Exception as e:
            logger.error(f"Failed to transform sensor data: {e}")
            return None
    
    def send_telemetry_data(self, telemetry_data: Dict) -> bool:
        """Send telemetry data to backend"""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    f"{BACKEND_URL}/api/telemetry",
                    json=telemetry_data
                )
                
                if response.status_code == 201:
                    logger.debug("Telemetry data sent successfully")
                    
                    # Emit real-time WebSocket updates for different data types
                    self.emit_realtime_updates(telemetry_data)
                    
                    return True
                elif response.status_code == 401:
                    logger.warning("Authentication expired, re-authenticating...")
                    if self.authenticate():
                        continue  # Retry with new token
                    else:
                        logger.error("Re-authentication failed")
                        return False
                else:
                    logger.warning(f"Failed to send telemetry: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Network error sending telemetry (attempt {attempt + 1}): {e}")
                
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                
        return False
    
    def emit_realtime_updates(self, telemetry_data: Dict):
        """Emit real-time WebSocket updates for different data types"""
        try:
            # Emit environmental sensor updates
            if telemetry_data.get('environmental_data'):
                self.emit_environmental_update(telemetry_data)
            
            # Emit air quality updates
            if telemetry_data.get('air_quality_data'):
                self.emit_air_quality_update(telemetry_data)
                
            # Emit hardware status updates
            if telemetry_data.get('hardware_status'):
                self.emit_hardware_status_update(telemetry_data)
                
        except Exception as e:
            logger.error(f"Failed to emit real-time updates: {e}")
    
    def emit_environmental_update(self, telemetry_data: Dict):
        """Send environmental update via WebSocket"""
        try:
            update_data = {
                'uav_id': telemetry_data['uav_id'],
                'timestamp': telemetry_data['timestamp'],
                'environmental': telemetry_data['environmental_data'],
                'location': {
                    'latitude': telemetry_data['latitude'],
                    'longitude': telemetry_data['longitude'],
                    'altitude': telemetry_data['altitude']
                }
            }
            
            # Use requests to trigger WebSocket emission via backend API
            self.session.post(
                f"{BACKEND_URL}/api/websocket/emit/environmental",
                json=update_data,
                timeout=2
            )
            
        except Exception as e:
            logger.debug(f"Environmental WebSocket update failed: {e}")
    
    def emit_air_quality_update(self, telemetry_data: Dict):
        """Send air quality update via WebSocket"""
        try:
            update_data = {
                'uav_id': telemetry_data['uav_id'],
                'timestamp': telemetry_data['timestamp'],
                'air_quality': telemetry_data['air_quality_data'],
                'location': {
                    'latitude': telemetry_data['latitude'],
                    'longitude': telemetry_data['longitude'],
                    'altitude': telemetry_data['altitude']
                }
            }
            
            self.session.post(
                f"{BACKEND_URL}/api/websocket/emit/air-quality",
                json=update_data,
                timeout=2
            )
            
        except Exception as e:
            logger.debug(f"Air quality WebSocket update failed: {e}")
    
    def emit_hardware_status_update(self, telemetry_data: Dict):
        """Send hardware status update via WebSocket"""
        try:
            update_data = {
                'uav_id': telemetry_data['uav_id'],
                'timestamp': telemetry_data['timestamp'],
                'status': telemetry_data['hardware_status']
            }
            
            self.session.post(
                f"{BACKEND_URL}/api/websocket/emit/hardware-status",
                json=update_data,
                timeout=2
            )
            
        except Exception as e:
            logger.debug(f"Hardware status WebSocket update failed: {e}")
    
    def run_bridge_loop(self):
        """Main bridge loop - runs continuously"""
        logger.info("Starting hardware bridge loop...")
        
        # Initial authentication
        if not self.authenticate():
            logger.error("Initial authentication failed - bridge will not start")
            return
            
        self.running = True
        
        while self.running:
            try:
                # Get new sensor readings
                readings = self.get_new_sensor_readings()
                
                if readings:
                    logger.info(f"Processing {len(readings)} new sensor readings")
                    
                    for reading in readings:
                        # Transform to backend format
                        telemetry_data = self.transform_sensor_data(reading)
                        
                        if telemetry_data:
                            # Send to backend
                            success = self.send_telemetry_data(telemetry_data)
                            if success:
                                logger.debug(f"Processed reading ID {reading['id']}")
                            else:
                                logger.warning(f"Failed to send reading ID {reading['id']}")
                else:
                    logger.debug("No new sensor readings")
                
                # Wait for next iteration
                time.sleep(BRIDGE_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Bridge interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in bridge loop: {e}")
                time.sleep(BRIDGE_INTERVAL)
        
        self.running = False
        logger.info("Hardware bridge stopped")
    
    def start(self):
        """Start the bridge service in a separate thread"""
        if self.running:
            logger.warning("Bridge is already running")
            return
            
        self.bridge_thread = threading.Thread(target=self.run_bridge_loop, daemon=True)
        self.bridge_thread.start()
        logger.info("Hardware bridge started in background thread")
    
    def stop(self):
        """Stop the bridge service"""
        self.running = False
        if hasattr(self, 'bridge_thread'):
            self.bridge_thread.join(timeout=5.0)
        logger.info("Hardware bridge stopped")

def main():
    """Main entry point for standalone bridge service"""
    bridge = HardwareBridge()
    
    try:
        bridge.run_bridge_loop()
    except KeyboardInterrupt:
        logger.info("Bridge service interrupted")
    finally:
        bridge.stop()

if __name__ == "__main__":
    main()