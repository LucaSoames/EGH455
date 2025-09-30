import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Grid,
} from '@mui/material';
import {
  Battery1Bar,
  Battery2Bar,
  Battery3Bar,
  Battery4Bar,
  Battery5Bar,
  Battery6Bar,
  BatteryFull,
  BatteryAlert,
  NetworkWifi1Bar,
  NetworkWifi2Bar,
  NetworkWifi3Bar,
  NetworkWifi,
  FlightTakeoff,
  Speed,
  Height,
  Thermostat,
} from '@mui/icons-material';

interface TelemetryData {
  id: number;
  uav_id: number;
  mission_id?: number;
  latitude: number;
  longitude: number;
  altitude: number;
  heading: number;
  speed: number;
  vertical_speed: number;
  battery_level: number;
  signal_strength: number;
  gps_satellites: number;
  system_status: 'normal' | 'warning' | 'error';
  error_messages?: string;
  temperature?: number;
  wind_speed?: number;
  wind_direction?: number;
  timestamp: string;
}

interface TelemetryWidgetProps {
  data: TelemetryData;
  uavModel: string;
  serialNumber: string;
}

const TelemetryWidget: React.FC<TelemetryWidgetProps> = ({ data, uavModel, serialNumber }) => {
  const getBatteryIcon = (level: number) => {
    if (level >= 90) return <BatteryFull color="success" />;
    if (level >= 75) return <Battery6Bar color="success" />;
    if (level >= 62) return <Battery5Bar color="success" />;
    if (level >= 50) return <Battery4Bar color="warning" />;
    if (level >= 37) return <Battery3Bar color="warning" />;
    if (level >= 25) return <Battery2Bar color="error" />;
    if (level >= 12) return <Battery1Bar color="error" />;
    return <BatteryAlert color="error" />;
  };

  const getSignalIcon = (strength: number) => {
    if (strength >= 75) return <NetworkWifi color="success" />;
    if (strength >= 50) return <NetworkWifi3Bar color="success" />;
    if (strength >= 25) return <NetworkWifi2Bar color="warning" />;
    return <NetworkWifi1Bar color="error" />;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h6" component="div">
              {uavModel}
            </Typography>
            <Typography color="textSecondary" variant="body2">
              {serialNumber}
            </Typography>
          </Box>
          <Chip 
            label={data.system_status.toUpperCase()}
            color={getStatusColor(data.system_status) as any}
            size="small"
          />
        </Box>

        <Grid container spacing={2}>
          {/* Battery Level */}
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              {getBatteryIcon(data.battery_level)}
              <Typography variant="body2" sx={{ ml: 1 }}>
                {data.battery_level}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={data.battery_level} 
              color={data.battery_level > 25 ? 'success' : 'error'}
              sx={{ height: 6, borderRadius: 3 }}
            />
          </Grid>

          {/* Signal Strength */}
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              {getSignalIcon(data.signal_strength)}
              <Typography variant="body2" sx={{ ml: 1 }}>
                {data.signal_strength}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={data.signal_strength} 
              color={data.signal_strength > 25 ? 'primary' : 'error'}
              sx={{ height: 6, borderRadius: 3 }}
            />
          </Grid>

          {/* Altitude */}
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Height fontSize="small" color="primary" />
              <Box sx={{ ml: 1 }}>
                <Typography variant="body2" color="textSecondary">
                  Altitude
                </Typography>
                <Typography variant="h6">
                  {data.altitude.toFixed(0)}m
                </Typography>
              </Box>
            </Box>
          </Grid>

          {/* Speed */}
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Speed fontSize="small" color="primary" />
              <Box sx={{ ml: 1 }}>
                <Typography variant="body2" color="textSecondary">
                  Speed
                </Typography>
                <Typography variant="h6">
                  {data.speed.toFixed(1)} m/s
                </Typography>
              </Box>
            </Box>
          </Grid>

          {/* GPS Satellites */}
          <Grid item xs={4}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FlightTakeoff fontSize="small" color="primary" />
              <Box sx={{ ml: 1 }}>
                <Typography variant="body2" color="textSecondary">
                  GPS Sats
                </Typography>
                <Typography variant="h6">
                  {data.gps_satellites}
                </Typography>
              </Box>
            </Box>
          </Grid>

          {/* Temperature */}
          {data.temperature && (
            <Grid item xs={6}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Thermostat fontSize="small" color="primary" />
                <Box sx={{ ml: 1 }}>
                  <Typography variant="body2" color="textSecondary">
                    Temperature
                  </Typography>
                  <Typography variant="h6">
                    {data.temperature.toFixed(1)}°C
                  </Typography>
                </Box>
              </Box>
            </Grid>
          )}

          {/* Position */}
          <Grid item xs={12}>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
              Position
            </Typography>
            <Typography variant="body2">
              {data.latitude.toFixed(6)}°, {data.longitude.toFixed(6)}°
            </Typography>
            <Typography variant="body2">
              Heading: {data.heading.toFixed(0)}°
            </Typography>
          </Grid>

          {/* Timestamp */}
          <Grid item xs={12}>
            <Typography variant="caption" color="textSecondary">
              Last update: {formatTime(data.timestamp)}
            </Typography>
          </Grid>
        </Grid>

        {data.error_messages && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="error">
              {data.error_messages}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default TelemetryWidget;