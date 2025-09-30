import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Alert,
  LinearProgress,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Button,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ReferenceLine,
} from 'recharts';
import {
  Air,
  DeviceThermostat,
  Opacity,
  Speed,
  LightMode,
  Warning,
  CheckCircle,
  Error as ErrorIcon,
  TrendingUp,
  TrendingDown,
  Refresh,
  Download,
  Notifications,
  NotificationsOff,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';
import axios from 'axios';

interface EnvironmentalReading {
  id: number;
  uav_id: number;
  mission_id?: number;
  timestamp: string;
  location: {
    latitude: number;
    longitude: number;
    altitude: number;
  };
  sensors: {
    // Air Quality Sensors
    co2: number;          // CO2 concentration (ppm)
    co: number;           // Carbon Monoxide (ppm) 
    no2: number;          // Nitrogen Dioxide (ppm)
    so2: number;          // Sulfur Dioxide (ppm)
    o3: number;           // Ozone (ppm)
    pm25: number;         // PM2.5 particles (μg/m³)
    pm10: number;         // PM10 particles (μg/m³)
    voc: number;          // Volatile Organic Compounds (ppm)
    
    // Environmental Sensors
    temperature: number;   // Temperature (°C)
    humidity: number;      // Humidity (%)
    pressure: number;      // Atmospheric pressure (hPa)
    light_intensity: number; // Light intensity (lux)
    uv_index: number;      // UV Index
    wind_speed: number;    // Wind speed (m/s)
    wind_direction: number; // Wind direction (degrees)
  };
  air_quality_index: number; // Overall AQI (0-500)
  hazard_level: 'safe' | 'moderate' | 'unhealthy' | 'hazardous' | 'dangerous';
}

interface EnvironmentalSensorDashboardProps {
  uavId?: number;
  missionId?: number;
  autoRefresh?: boolean;
}

const EnvironmentalSensorDashboard: React.FC<EnvironmentalSensorDashboardProps> = ({
  uavId,
  missionId,
  autoRefresh = true
}) => {
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('6h');
  const [realTimeUpdates, setRealTimeUpdates] = useState(autoRefresh);
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [selectedSensor, setSelectedSensor] = useState<string>('all');

  // Fetch environmental data from hardware API
  const { data: environmentalData = [], isLoading: envLoading } = useQuery({
    queryKey: ['hardware-environmental', uavId, timeRange],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (uavId) params.append('uav_id', uavId.toString());
      
      // Convert time range to hours
      const hours = timeRange === '1h' ? 1 : timeRange === '6h' ? 6 : timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 1;
      params.append('hours', hours.toString());
      
      try {
        const response = await axios.get(`/api/hardware/environmental?${params}`);
        return response.data.data || [];
      } catch (error) {
        console.warn('Hardware environmental API not available, using fallback');
        return [];
      }
    },
    refetchInterval: realTimeUpdates ? 5000 : false,
  });

  // Fetch air quality data from hardware API  
  const { data: airQualityData = [], isLoading: airLoading } = useQuery({
    queryKey: ['hardware-air-quality', uavId, timeRange],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (uavId) params.append('uav_id', uavId.toString());
      
      const hours = timeRange === '1h' ? 1 : timeRange === '6h' ? 6 : timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 1;
      params.append('hours', hours.toString());
      
      try {
        const response = await axios.get(`/api/hardware/air-quality?${params}`);
        return response.data.data || [];
      } catch (error) {
        console.warn('Hardware air quality API not available, using fallback');
        return [];
      }
    },
    refetchInterval: realTimeUpdates ? 5000 : false,
  });

  // Combine hardware data into the expected format
  const readings = React.useMemo(() => {
    const combined: EnvironmentalReading[] = [];
    
    // Merge environmental and air quality data by timestamp
    const envMap = new Map<string, any>();
    environmentalData.forEach((item: any) => {
      const key = item.timestamp;
      envMap.set(key, item);
    });
    
    airQualityData.forEach((item: any) => {
      const key = item.timestamp;
      const envItem = envMap.get(key);
      
      if (envItem || item.air_quality || item.environmental) {
        const reading: EnvironmentalReading = {
          id: Date.parse(key),
          uav_id: item.uav_id,
          timestamp: key,
          location: item.location || { latitude: 0, longitude: 0, altitude: 0 },
          sensors: {
            // Air quality from hardware sensors
            co2: item.air_quality?.co2 || 400,
            co: item.air_quality?.co || 0,
            no2: item.air_quality?.no2 || 0,
            so2: 0, // Not available in current hardware
            o3: 0,  // Not available in current hardware  
            pm25: item.air_quality?.pm25 || 0,
            pm10: item.air_quality?.pm10 || 0,
            voc: 0, // Could be derived from gas readings
            
            // Environmental from hardware sensors
            temperature: (envItem?.environmental?.temperature || item.environmental?.temperature) || 20,
            humidity: (envItem?.environmental?.humidity || item.environmental?.humidity) || 50,
            pressure: (envItem?.environmental?.pressure || item.environmental?.pressure) || 1013.25,
            light_intensity: (envItem?.environmental?.light_intensity || item.environmental?.light_intensity) || 1000,
            uv_index: 5, // Not available in current hardware
            wind_speed: 0, // Not available in current hardware
            wind_direction: 0, // Not available in current hardware
          },
          air_quality_index: item.air_quality?.aqi || 50,
          hazard_level: (item.air_quality?.aqi || 50) <= 50 ? 'safe' : 
                       (item.air_quality?.aqi || 50) <= 100 ? 'moderate' :
                       (item.air_quality?.aqi || 50) <= 150 ? 'unhealthy' : 'hazardous'
        };
        
        combined.push(reading);
      }
    });
    
    return combined.sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
  }, [environmentalData, airQualityData]);

  const isLoading = envLoading || airLoading;
  const refetch = () => {
    // Refetch would be handled by React Query automatically
  };

  const latestReading = readings[readings.length - 1];

  const getHazardColor = (level: string): 'success' | 'info' | 'warning' | 'error' => {
    switch (level) {
      case 'safe': return 'success';
      case 'moderate': return 'info';
      case 'unhealthy': return 'warning';
      case 'hazardous': case 'dangerous': return 'error';
      default: return 'info';
    }
  };

  const getAQIColor = (aqi: number) => {
    if (aqi <= 50) return '#4caf50';    // Good - Green
    if (aqi <= 100) return '#ffeb3b';   // Moderate - Yellow
    if (aqi <= 150) return '#ff9800';   // Unhealthy for Sensitive Groups - Orange
    if (aqi <= 200) return '#f44336';   // Unhealthy - Red
    if (aqi <= 300) return '#9c27b0';   // Very Unhealthy - Purple
    return '#795548';                   // Hazardous - Maroon
  };

  const getSensorStatus = (sensorType: string, value: number) => {
    const thresholds: { [key: string]: { warning: number; danger: number; unit: string } } = {
      co2: { warning: 1000, danger: 5000, unit: 'ppm' },
      co: { warning: 35, danger: 200, unit: 'ppm' },
      no2: { warning: 100, danger: 200, unit: 'ppm' },
      so2: { warning: 75, danger: 185, unit: 'ppm' },
      pm25: { warning: 35, danger: 75, unit: 'μg/m³' },
      pm10: { warning: 150, danger: 250, unit: 'μg/m³' },
      temperature: { warning: 35, danger: 40, unit: '°C' },
      humidity: { warning: 80, danger: 95, unit: '%' },
    };

    const threshold = thresholds[sensorType];
    if (!threshold) return { status: 'normal', color: 'success' };

    if (value >= threshold.danger) return { status: 'danger', color: 'error' };
    if (value >= threshold.warning) return { status: 'warning', color: 'warning' };
    return { status: 'normal', color: 'success' };
  };

  const formatSensorData = (readings: EnvironmentalReading[]) => {
    return readings.map(reading => ({
      ...reading,
      time: format(new Date(reading.timestamp), 'HH:mm'),
      date: format(new Date(reading.timestamp), 'MMM dd'),
    }));
  };

  const chartData = formatSensorData(readings);

  const hazardousReadings = readings.filter(reading => 
    reading.hazard_level === 'hazardous' || reading.hazard_level === 'dangerous'
  );

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          <Air sx={{ mr: 1, verticalAlign: 'middle' }} />
          Environmental Monitoring System
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <FormControlLabel
            control={
              <Switch
                checked={realTimeUpdates}
                onChange={(e) => setRealTimeUpdates(e.target.checked)}
              />
            }
            label="Real-time"
          />
          <FormControlLabel
            control={
              <Switch
                checked={alertsEnabled}
                onChange={(e) => setAlertsEnabled(e.target.checked)}
              />
            }
            label={alertsEnabled ? <Notifications /> : <NotificationsOff />}
          />
          <IconButton onClick={() => refetch()}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Hazard Alerts */}
      {alertsEnabled && hazardousReadings.length > 0 && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <strong>HAZARDOUS CONDITIONS DETECTED!</strong>
          {' '}{hazardousReadings.length} hazardous reading(s) in the current time range. 
          Immediate safety protocols required.
        </Alert>
      )}

      {latestReading && latestReading.hazard_level !== 'safe' && (
        <Alert 
          severity={getHazardColor(latestReading.hazard_level)} 
          sx={{ mb: 3 }}
        >
          Current hazard level: <strong>{latestReading.hazard_level.toUpperCase()}</strong>
          {' '}(AQI: {latestReading.air_quality_index})
        </Alert>
      )}

      {/* Current Status Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ color: latestReading ? getAQIColor(latestReading.air_quality_index) : 'inherit' }}>
                {latestReading ? latestReading.air_quality_index : '--'}
              </Typography>
              <Typography color="textSecondary">
                Air Quality Index
              </Typography>
              {latestReading && (
                <Chip 
                  label={latestReading.hazard_level.toUpperCase()}
                  color={getHazardColor(latestReading.hazard_level)}
                  size="small"
                  sx={{ mt: 1 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {latestReading ? latestReading.sensors.temperature.toFixed(1) : '--'}°C
              </Typography>
              <Typography color="textSecondary">
                Temperature
              </Typography>
              {latestReading && (
                <Chip 
                  label={getSensorStatus('temperature', latestReading.sensors.temperature).status}
                  color={getSensorStatus('temperature', latestReading.sensors.temperature).color as any}
                  size="small"
                  sx={{ mt: 1 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info">
                {latestReading ? latestReading.sensors.humidity.toFixed(1) : '--'}%
              </Typography>
              <Typography color="textSecondary">
                Humidity
              </Typography>
              {latestReading && (
                <Chip 
                  label={getSensorStatus('humidity', latestReading.sensors.humidity).status}
                  color={getSensorStatus('humidity', latestReading.sensors.humidity).color as any}
                  size="small"
                  sx={{ mt: 1 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning">
                {latestReading ? latestReading.sensors.co2.toFixed(0) : '--'}
              </Typography>
              <Typography color="textSecondary">
                CO2 (ppm)
              </Typography>
              {latestReading && (
                <Chip 
                  label={getSensorStatus('co2', latestReading.sensors.co2).status}
                  color={getSensorStatus('co2', latestReading.sensors.co2).color as any}
                  size="small"
                  sx={{ mt: 1 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Controls */}
      <Grid container spacing={2} sx={{ mb: 3 }} alignItems="center">
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => setTimeRange(e.target.value as any)}
            >
              <MenuItem value="1h">Last Hour</MenuItem>
              <MenuItem value="6h">Last 6 Hours</MenuItem>
              <MenuItem value="24h">Last 24 Hours</MenuItem>
              <MenuItem value="7d">Last 7 Days</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Focus Sensor</InputLabel>
            <Select
              value={selectedSensor}
              label="Focus Sensor"
              onChange={(e) => setSelectedSensor(e.target.value)}
            >
              <MenuItem value="all">All Sensors</MenuItem>
              <MenuItem value="air_quality">Air Quality</MenuItem>
              <MenuItem value="temperature">Temperature</MenuItem>
              <MenuItem value="humidity">Humidity</MenuItem>
              <MenuItem value="pressure">Pressure</MenuItem>
              <MenuItem value="light">Light Intensity</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Button 
            variant="outlined" 
            startIcon={<Download />}
            fullWidth
            onClick={() => alert('Environmental data export initiated')}
          >
            Export Data
          </Button>
        </Grid>
      </Grid>

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Charts */}
      <Grid container spacing={3}>
        {/* Air Quality Over Time */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Air Quality Index Trend
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <ChartTooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="air_quality_index"
                    stroke="#ff7300"
                    fill="url(#colorAQI)"
                    name="Air Quality Index"
                  />
                  <defs>
                    <linearGradient id="colorAQI" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff7300" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#ff7300" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <ReferenceLine y={50} stroke="green" strokeDasharray="5 5" label="Good" />
                  <ReferenceLine y={100} stroke="yellow" strokeDasharray="5 5" label="Moderate" />
                  <ReferenceLine y={150} stroke="orange" strokeDasharray="5 5" label="Unhealthy" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Sensor Readings Radar */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Sensor Status
              </Typography>
              {latestReading ? (
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={[
                    { sensor: 'CO2', value: Math.min(latestReading.sensors.co2 / 20, 100), fullMark: 100 },
                    { sensor: 'PM2.5', value: Math.min(latestReading.sensors.pm25 * 2, 100), fullMark: 100 },
                    { sensor: 'Temp', value: Math.min(latestReading.sensors.temperature * 2.5, 100), fullMark: 100 },
                    { sensor: 'Humidity', value: latestReading.sensors.humidity, fullMark: 100 },
                    { sensor: 'Pressure', value: Math.min((latestReading.sensors.pressure - 900) / 2, 100), fullMark: 100 },
                    { sensor: 'Light', value: Math.min(latestReading.sensors.light_intensity / 100, 100), fullMark: 100 },
                  ]}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="sensor" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="Current Values"
                      dataKey="value"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.6}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body2" color="textSecondary">
                    No current readings available
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Hazardous Gas Concentrations */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Hazardous Gas Concentrations
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <ChartTooltip />
                  <Legend />
                  <Line type="monotone" dataKey="sensors.co2" stroke="#ff7300" name="CO2 (ppm)" dot={false} />
                  <Line type="monotone" dataKey="sensors.co" stroke="#ff0000" name="CO (ppm)" dot={false} />
                  <Line type="monotone" dataKey="sensors.no2" stroke="#8884d8" name="NO2 (ppm)" dot={false} />
                  <Line type="monotone" dataKey="sensors.so2" stroke="#82ca9d" name="SO2 (ppm)" dot={false} />
                  <ReferenceLine y={1000} stroke="#ff7300" strokeDasharray="5 5" label="CO2 Limit" />
                  <ReferenceLine y={35} stroke="#ff0000" strokeDasharray="5 5" label="CO Limit" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Environmental Conditions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Environmental Conditions
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <ChartTooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="sensors.temperature" stroke="#ff7300" name="Temperature (°C)" />
                  <Line yAxisId="right" type="monotone" dataKey="sensors.humidity" stroke="#82ca9d" name="Humidity (%)" />
                  <Line yAxisId="left" type="monotone" dataKey="sensors.light_intensity" stroke="#8dd1e1" name="Light (lux/10)" 
                        strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Particulate Matter */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Particulate Matter Levels
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <ChartTooltip />
                  <Legend />
                  <Bar dataKey="sensors.pm25" fill="#82ca9d" name="PM2.5 (μg/m³)" />
                  <Bar dataKey="sensors.pm10" fill="#8dd1e1" name="PM10 (μg/m³)" />
                  <ReferenceLine y={35} stroke="#82ca9d" strokeDasharray="5 5" label="PM2.5 Limit" />
                  <ReferenceLine y={150} stroke="#8dd1e1" strokeDasharray="5 5" label="PM10 Limit" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default EnvironmentalSensorDashboard;