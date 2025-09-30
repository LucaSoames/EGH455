import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  Switch,
  FormControlLabel,
  LinearProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
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
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { format, subHours, subDays, subWeeks } from 'date-fns';
import axios from 'axios';

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
  system_status: string;
  error_messages?: string;
  temperature: number;
  wind_speed: number;
  wind_direction: number;
  timestamp: string;
  air_quality?: {
    co2: number;
    co: number;
    no2: number;
    o3: number;
    pm25: number;
    pm10: number;
    humidity: number;
    pressure: number;
    light_intensity: number;
  };
}

interface UAV {
  id: number;
  serial_number: string;
  model: string;
  status: string;
}

type TimeRange = '1h' | '6h' | '24h' | '7d';
type MetricType = 'air_quality' | 'environmental' | 'flight' | 'system';

const TelemetryVisualizationPage: React.FC = () => {
  const [selectedUAV, setSelectedUAV] = useState<number | 'all'>('all');
  const [timeRange, setTimeRange] = useState<TimeRange>('6h');
  const [metricType, setMetricType] = useState<MetricType>('air_quality');
  const [realTimeUpdates, setRealTimeUpdates] = useState(true);

  const { data: uavs } = useQuery({
    queryKey: ['uavs'],
    queryFn: async () => {
      const response = await axios.get('/api/uavs');
      return response.data.data as UAV[];
    },
  });

  const { data: telemetryData, isLoading } = useQuery({
    queryKey: ['telemetry', selectedUAV, timeRange],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (selectedUAV !== 'all') {
        params.append('uav_id', selectedUAV.toString());
      }
      params.append('timeRange', timeRange);
      
      const response = await axios.get(`/api/telemetry/historical?${params}`);
      return response.data.data as TelemetryData[];
    },
    refetchInterval: realTimeUpdates ? 5000 : false,
  });

  const getTimeRangeHours = (range: TimeRange): number => {
    switch (range) {
      case '1h': return 1;
      case '6h': return 6;
      case '24h': return 24;
      case '7d': return 168;
      default: return 6;
    }
  };

  const formatTelemetryData = (data: TelemetryData[] = []) => {
    return data.map(item => ({
      ...item,
      formattedTime: format(new Date(item.timestamp), 'HH:mm:ss'),
      formattedDate: format(new Date(item.timestamp), 'MMM dd, HH:mm'),
    }));
  };

  const getAirQualityStatus = (data?: TelemetryData['air_quality']) => {
    if (!data) return { status: 'unknown', color: 'default', message: 'No data available' };
    
    const { co2, co, no2, pm25 } = data;
    
    // Air quality thresholds (simplified)
    if (co2 > 1000 || co > 35 || no2 > 200 || pm25 > 75) {
      return { status: 'poor', color: 'error', message: 'Poor air quality detected' };
    } else if (co2 > 800 || co > 20 || no2 > 100 || pm25 > 35) {
      return { status: 'moderate', color: 'warning', message: 'Moderate air quality' };
    } else {
      return { status: 'good', color: 'success', message: 'Good air quality' };
    }
  };

  const renderAirQualityCharts = () => {
    const data = formatTelemetryData(telemetryData?.filter(d => d.air_quality));
    
    if (!data.length) {
      return (
        <Alert severity="info">
          No air quality data available for the selected time range
        </Alert>
      );
    }

    const latestData = data[data.length - 1]?.air_quality;
    const airQualityStatus = getAirQualityStatus(latestData);

    return (
      <Grid container spacing={3}>
        {/* Air Quality Status */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">Current Air Quality Status</Typography>
                <Chip
                  icon={airQualityStatus.status === 'good' ? <CheckCircle /> : 
                        airQualityStatus.status === 'moderate' ? <Warning /> : <ErrorIcon />}
                  label={airQualityStatus.message}
                  color={airQualityStatus.color as any}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Gas Concentrations */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Air sx={{ mr: 1, verticalAlign: 'middle' }} />
                Gas Concentrations (ppm)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="formattedTime" 
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis />
                  <Tooltip labelFormatter={(label) => `Time: ${label}`} />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="air_quality.co2" 
                    stroke="#ff7300" 
                    name="CO2 (ppm)"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="air_quality.co" 
                    stroke="#ff0000" 
                    name="CO (ppm)"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="air_quality.no2" 
                    stroke="#8884d8" 
                    name="NO2 (ppm)"
                    dot={false}
                    strokeWidth={2}
                  />
                  <ReferenceLine y={1000} stroke="#ff7300" strokeDasharray="5 5" label="CO2 Limit" />
                  <ReferenceLine y={35} stroke="#ff0000" strokeDasharray="5 5" label="CO Limit" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Particulate Matter */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Particulate Matter (μg/m³)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="formattedTime" tick={{ fontSize: 12 }} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="air_quality.pm25"
                    stackId="1"
                    stroke="#82ca9d"
                    fill="#82ca9d"
                    name="PM2.5"
                  />
                  <Area
                    type="monotone"
                    dataKey="air_quality.pm10"
                    stackId="1"
                    stroke="#8dd1e1"
                    fill="#8dd1e1"
                    name="PM10"
                  />
                  <ReferenceLine y={35} stroke="#82ca9d" strokeDasharray="5 5" label="PM2.5 Limit" />
                  <ReferenceLine y={150} stroke="#8dd1e1" strokeDasharray="5 5" label="PM10 Limit" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderEnvironmentalCharts = () => {
    const data = formatTelemetryData(telemetryData);
    
    return (
      <Grid container spacing={3}>
        {/* Temperature and Humidity */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <DeviceThermostat sx={{ mr: 1, verticalAlign: 'middle' }} />
                Temperature & Humidity
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="formattedTime" />
                  <YAxis yAxisId="temp" orientation="left" />
                  <YAxis yAxisId="humidity" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line
                    yAxisId="temp"
                    type="monotone"
                    dataKey="temperature"
                    stroke="#ff7300"
                    name="Temperature (°C)"
                    dot={false}
                  />
                  <Line
                    yAxisId="humidity"
                    type="monotone"
                    dataKey="air_quality.humidity"
                    stroke="#82ca9d"
                    name="Humidity (%)"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Pressure and Light */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <LightMode sx={{ mr: 1, verticalAlign: 'middle' }} />
                Pressure & Light Intensity
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="formattedTime" />
                  <YAxis yAxisId="pressure" orientation="left" />
                  <YAxis yAxisId="light" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line
                    yAxisId="pressure"
                    type="monotone"
                    dataKey="air_quality.pressure"
                    stroke="#8884d8"
                    name="Pressure (hPa)"
                    dot={false}
                  />
                  <Line
                    yAxisId="light"
                    type="monotone"
                    dataKey="air_quality.light_intensity"
                    stroke="#ffbb33"
                    name="Light (lux)"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderFlightCharts = () => {
    const data = formatTelemetryData(telemetryData);
    
    return (
      <Grid container spacing={3}>
        {/* Altitude and Speed */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Flight Parameters
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="formattedTime" />
                  <YAxis yAxisId="altitude" orientation="left" />
                  <YAxis yAxisId="speed" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line
                    yAxisId="altitude"
                    type="monotone"
                    dataKey="altitude"
                    stroke="#8884d8"
                    name="Altitude (m)"
                    dot={false}
                  />
                  <Line
                    yAxisId="speed"
                    type="monotone"
                    dataKey="speed"
                    stroke="#82ca9d"
                    name="Speed (m/s)"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Battery and Signal */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="formattedTime" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="battery_level"
                    stroke="#ff7300"
                    name="Battery (%)"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="signal_strength"
                    stroke="#8dd1e1"
                    name="Signal Strength (%)"
                    dot={false}
                  />
                  <ReferenceLine y={20} stroke="#ff7300" strokeDasharray="5 5" label="Low Battery" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderSystemCharts = () => {
    const data = formatTelemetryData(telemetryData);
    
    return (
      <Grid container spacing={3}>
        {/* System Status Distribution */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={data.reduce((acc, curr) => {
                  const existing = acc.find(item => item.status === curr.system_status);
                  if (existing) {
                    existing.count += 1;
                  } else {
                    acc.push({ status: curr.system_status, count: 1 });
                  }
                  return acc;
                }, [] as { status: string; count: number }[])}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="status" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* GPS Satellites */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                GPS Satellite Count
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="formattedTime" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="gps_satellites"
                    stroke="#82ca9d"
                    name="GPS Satellites"
                    dot={false}
                  />
                  <ReferenceLine y={4} stroke="#ff0000" strokeDasharray="5 5" label="Minimum GPS" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderMetricCharts = () => {
    switch (metricType) {
      case 'air_quality':
        return renderAirQualityCharts();
      case 'environmental':
        return renderEnvironmentalCharts();
      case 'flight':
        return renderFlightCharts();
      case 'system':
        return renderSystemCharts();
      default:
        return renderAirQualityCharts();
    }
  };

  return (
    <Container maxWidth="xl">
      <Paper sx={{ p: 3 }}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h4" gutterBottom>
            Advanced Telemetry Visualization
          </Typography>
          <Typography variant="body1" color="textSecondary" sx={{ mb: 3 }}>
            Real-time air quality monitoring and environmental sensor data analysis
          </Typography>

          {/* Controls */}
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>UAV</InputLabel>
                <Select
                  value={selectedUAV}
                  label="UAV"
                  onChange={(e) => setSelectedUAV(e.target.value as number | 'all')}
                >
                  <MenuItem value="all">All UAVs</MenuItem>
                  {uavs?.map((uav) => (
                    <MenuItem key={uav.id} value={uav.id}>
                      {uav.serial_number} - {uav.model}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Time Range</InputLabel>
                <Select
                  value={timeRange}
                  label="Time Range"
                  onChange={(e) => setTimeRange(e.target.value as TimeRange)}
                >
                  <MenuItem value="1h">Last Hour</MenuItem>
                  <MenuItem value="6h">Last 6 Hours</MenuItem>
                  <MenuItem value="24h">Last 24 Hours</MenuItem>
                  <MenuItem value="7d">Last 7 Days</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Data Type</InputLabel>
                <Select
                  value={metricType}
                  label="Data Type"
                  onChange={(e) => setMetricType(e.target.value as MetricType)}
                >
                  <MenuItem value="air_quality">Air Quality</MenuItem>
                  <MenuItem value="environmental">Environmental</MenuItem>
                  <MenuItem value="flight">Flight Data</MenuItem>
                  <MenuItem value="system">System Health</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={realTimeUpdates}
                    onChange={(e) => setRealTimeUpdates(e.target.checked)}
                  />
                }
                label="Real-time Updates"
              />
            </Grid>
          </Grid>
        </Box>

        {/* Loading Indicator */}
        {isLoading && <LinearProgress sx={{ mb: 2 }} />}

        {/* Charts */}
        {renderMetricCharts()}
      </Paper>
    </Container>
  );
};

export default TelemetryVisualizationPage;