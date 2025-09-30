import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Chip,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Button,
  Badge,
  Divider,
  Paper,
} from '@mui/material';
import {
  Flight,
  Speed,
  Height,
  BatteryFull,
  NetworkWifi,
  Thermostat,
  Air,
  Navigation,
  Warning,
  Error,
  CheckCircle,
  Pause,
  PlayArrow,
  Settings,
  Fullscreen,
  FullscreenExit,
  Timeline,
  TrendingUp,
  TrendingDown,
  Remove,
} from '@mui/icons-material';
import { useSocket } from '../contexts/SocketContext';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  ReferenceLine,
} from 'recharts';
import { format } from 'date-fns';

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
  flight_mode?: string;
  payload_status?: string;
}

interface RealTimeTelemetryDashboardProps {
  uavId?: number;
  maxDataPoints?: number;
  updateInterval?: number;
  showAlerts?: boolean;
  compactMode?: boolean;
}

interface MetricCard {
  key: keyof TelemetryData;
  label: string;
  icon: React.ReactNode;
  unit: string;
  format: (value: number) => string;
  thresholds: {
    critical?: { min?: number; max?: number };
    warning?: { min?: number; max?: number };
  };
  chartColor: string;
}

const METRIC_CONFIGS: MetricCard[] = [
  {
    key: 'altitude',
    label: 'Altitude',
    icon: <Height />,
    unit: 'm',
    format: (v) => v.toFixed(1),
    thresholds: {
      warning: { max: 100 },
      critical: { max: 120 },
    },
    chartColor: '#8884d8',
  },
  {
    key: 'speed',
    label: 'Speed',
    icon: <Speed />,
    unit: 'm/s',
    format: (v) => v.toFixed(1),
    thresholds: {
      warning: { max: 15 },
      critical: { max: 20 },
    },
    chartColor: '#82ca9d',
  },
  {
    key: 'battery_level',
    label: 'Battery',
    icon: <BatteryFull />,
    unit: '%',
    format: (v) => v.toFixed(0),
    thresholds: {
      critical: { min: 20 },
      warning: { min: 30 },
    },
    chartColor: '#ffc658',
  },
  {
    key: 'signal_strength',
    label: 'Signal',
    icon: <NetworkWifi />,
    unit: '%',
    format: (v) => v.toFixed(0),
    thresholds: {
      critical: { min: 30 },
      warning: { min: 50 },
    },
    chartColor: '#ff7300',
  },
];

const RealTimeTelemetryDashboard: React.FC<RealTimeTelemetryDashboardProps> = ({
  uavId,
  maxDataPoints = 50,
  updateInterval = 1000,
  showAlerts = true,
  compactMode = false,
}) => {
  const [paused, setPaused] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<string>('altitude');
  const [dataBuffer, setDataBuffer] = useState<(TelemetryData & { timestamp_formatted: string })[]>([]);
  const [alerts, setAlerts] = useState<{ id: string; message: string; severity: 'warning' | 'error'; timestamp: Date }[]>([]);
  
  const { telemetryData, latestTelemetry, isConnected, joinTelemetryUpdates } = useSocket();
  const dataUpdateRef = useRef<NodeJS.Timeout>();

  // Join telemetry updates on mount
  useEffect(() => {
    joinTelemetryUpdates();
  }, [joinTelemetryUpdates]);

  // Process incoming telemetry data
  useEffect(() => {
    if (paused || !isConnected) return;

    const processData = () => {
      const relevantData = uavId 
        ? telemetryData.filter(d => d.uav_id === uavId)
        : telemetryData;

      const processedData = relevantData
        .slice(0, maxDataPoints)
        .map(data => ({
          ...data,
          timestamp_formatted: format(new Date(data.timestamp), 'HH:mm:ss'),
        }));

      setDataBuffer(processedData);

      // Check for alerts
      if (showAlerts && processedData.length > 0) {
        const latest = processedData[0];
        checkForAlerts(latest);
      }
    };

    processData();
    dataUpdateRef.current = setInterval(processData, updateInterval);

    return () => {
      if (dataUpdateRef.current) {
        clearInterval(dataUpdateRef.current);
      }
    };
  }, [telemetryData, uavId, maxDataPoints, updateInterval, paused, isConnected, showAlerts]);

  const checkForAlerts = (data: TelemetryData) => {
    const newAlerts: typeof alerts = [];

    METRIC_CONFIGS.forEach(config => {
      const value = data[config.key] as number;
      if (typeof value !== 'number') return;

      const { critical, warning } = config.thresholds;

      if (critical) {
        if ((critical.min !== undefined && value < critical.min) || 
            (critical.max !== undefined && value > critical.max)) {
          newAlerts.push({
            id: `critical-${config.key}-${Date.now()}`,
            message: `Critical ${config.label}: ${config.format(value)}${config.unit}`,
            severity: 'error',
            timestamp: new Date(),
          });
        }
      } else if (warning) {
        if ((warning.min !== undefined && value < warning.min) || 
            (warning.max !== undefined && value > warning.max)) {
          newAlerts.push({
            id: `warning-${config.key}-${Date.now()}`,
            message: `Warning ${config.label}: ${config.format(value)}${config.unit}`,
            severity: 'warning',
            timestamp: new Date(),
          });
        }
      }
    });

    if (data.system_status === 'error' && data.error_messages) {
      newAlerts.push({
        id: `system-error-${Date.now()}`,
        message: `System Error: ${data.error_messages}`,
        severity: 'error',
        timestamp: new Date(),
      });
    }

    if (newAlerts.length > 0) {
      setAlerts(prev => [...newAlerts, ...prev].slice(0, 10)); // Keep last 10 alerts
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'normal': return <CheckCircle color="success" />;
      case 'warning': return <Warning color="warning" />;
      case 'error': return <Error color="error" />;
      default: return <CheckCircle />;
    }
  };

  const getTrendIndicator = (current: number, previous: number) => {
    const diff = current - previous;
    if (Math.abs(diff) < 0.1) return <Remove color="disabled" fontSize="small" />;
    return diff > 0 
      ? <TrendingUp color="success" fontSize="small" />
      : <TrendingDown color="error" fontSize="small" />;
  };

  const getMetricStatus = (config: MetricCard, value: number) => {
    const { critical, warning } = config.thresholds;
    
    if (critical) {
      if ((critical.min !== undefined && value < critical.min) || 
          (critical.max !== undefined && value > critical.max)) {
        return 'error';
      }
    }
    
    if (warning) {
      if ((warning.min !== undefined && value < warning.min) || 
          (warning.max !== undefined && value > warning.max)) {
        return 'warning';
      }
    }
    
    return 'success';
  };

  const latestData = dataBuffer[0];
  const previousData = dataBuffer[1];
  const chartData = dataBuffer.slice().reverse(); // Reverse for chronological order

  return (
    <Box sx={{ height: fullscreen ? '100vh' : 'auto', overflow: 'auto' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant={compactMode ? "h6" : "h5"}>
            Real-time Telemetry
            {uavId && ` - UAV ${uavId}`}
          </Typography>
          <Chip
            icon={isConnected ? <CheckCircle /> : <Error />}
            label={isConnected ? 'Live' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            variant="outlined"
          />
          {latestData && (
            <Chip
              icon={getStatusIcon(latestData.system_status)}
              label={latestData.system_status.toUpperCase()}
              color={getStatusColor(latestData.system_status) as any}
              variant="filled"
            />
          )}
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <FormControlLabel
            control={
              <Switch
                checked={!paused}
                onChange={(e) => setPaused(!e.target.checked)}
                size="small"
              />
            }
            label="Live"
          />
          <Badge badgeContent={alerts.length} color="error">
            <IconButton size="small">
              <Warning />
            </IconButton>
          </Badge>
          <IconButton size="small" onClick={() => setFullscreen(!fullscreen)}>
            {fullscreen ? <FullscreenExit /> : <Fullscreen />}
          </IconButton>
        </Box>
      </Box>

      {/* Connection Status */}
      {!isConnected && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Connection lost. Telemetry data is not updating.
        </Alert>
      )}

      {/* Alerts */}
      {showAlerts && alerts.length > 0 && (
        <Card sx={{ mb: 2 }}>
          <CardContent sx={{ py: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Active Alerts ({alerts.length})
            </Typography>
            <Box sx={{ maxHeight: 100, overflow: 'auto' }}>
              {alerts.slice(0, 3).map(alert => (
                <Alert 
                  key={alert.id} 
                  severity={alert.severity} 
                  sx={{ mb: 0.5, py: 0 }}
                  onClose={() => setAlerts(prev => prev.filter(a => a.id !== alert.id))}
                >
                  <Typography variant="body2">
                    {alert.message}
                  </Typography>
                </Alert>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      <Grid container spacing={compactMode ? 1 : 2}>
        {/* Metric Cards */}
        {METRIC_CONFIGS.map(config => {
          const currentValue = latestData?.[config.key] as number;
          const previousValue = previousData?.[config.key] as number;
          const status = currentValue !== undefined ? getMetricStatus(config, currentValue) : 'default';
          
          return (
            <Grid item xs={6} md={3} key={config.key}>
              <Card 
                sx={{ 
                  cursor: 'pointer',
                  border: selectedMetric === config.key ? 2 : 0,
                  borderColor: 'primary.main',
                }}
                onClick={() => setSelectedMetric(config.key)}
              >
                <CardContent sx={{ py: compactMode ? 1 : 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        color: status === 'error' ? 'error.main' : 
                               status === 'warning' ? 'warning.main' : 'success.main'
                      }}>
                        {config.icon}
                      </Box>
                      <Box sx={{ ml: 1 }}>
                        <Typography color="textSecondary" variant="body2">
                          {config.label}
                        </Typography>
                        <Typography variant={compactMode ? "h6" : "h5"}>
                          {currentValue !== undefined 
                            ? `${config.format(currentValue)}${config.unit}`
                            : '--'
                          }
                        </Typography>
                      </Box>
                    </Box>
                    {previousValue !== undefined && currentValue !== undefined && (
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        {getTrendIndicator(currentValue, previousValue)}
                        <Typography variant="caption" color="textSecondary">
                          {config.format(Math.abs(currentValue - previousValue))}
                        </Typography>
                      </Box>
                    )}
                  </Box>
                  
                  {/* Mini progress bar for percentage metrics */}
                  {(config.key === 'battery_level' || config.key === 'signal_strength') && 
                   currentValue !== undefined && (
                    <LinearProgress
                      variant="determinate"
                      value={currentValue}
                      color={status as any}
                      sx={{ mt: 1, height: 6, borderRadius: 3 }}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>
          );
        })}

        {/* Additional Info Cards */}
        {latestData && (
          <React.Fragment>
            <Grid item xs={6} md={3}>
              <Card>
                <CardContent sx={{ py: compactMode ? 1 : 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Navigation sx={{ fontSize: 32, color: 'info.main', mr: 1 }} />
                    <Box>
                      <Typography color="textSecondary" variant="body2">
                        GPS Satellites
                      </Typography>
                      <Typography variant={compactMode ? "h6" : "h5"}>
                        {latestData.gps_satellites}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={6} md={3}>
              <Card>
                <CardContent sx={{ py: compactMode ? 1 : 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Flight sx={{ fontSize: 32, color: 'primary.main', mr: 1 }} />
                    <Box>
                      <Typography color="textSecondary" variant="body2">
                        Heading
                      </Typography>
                      <Typography variant={compactMode ? "h6" : "h5"}>
                        {latestData.heading}°
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {latestData.temperature !== undefined && (
              <Grid item xs={6} md={3}>
                <Card>
                  <CardContent sx={{ py: compactMode ? 1 : 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Thermostat sx={{ fontSize: 32, color: 'secondary.main', mr: 1 }} />
                      <Box>
                        <Typography color="textSecondary" variant="body2">
                          Temperature
                        </Typography>
                        <Typography variant={compactMode ? "h6" : "h5"}>
                          {latestData.temperature}°C
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {latestData.wind_speed !== undefined && (
              <Grid item xs={6} md={3}>
                <Card>
                  <CardContent sx={{ py: compactMode ? 1 : 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Air sx={{ fontSize: 32, color: 'info.main', mr: 1 }} />
                      <Box>
                        <Typography color="textSecondary" variant="body2">
                          Wind Speed
                        </Typography>
                        <Typography variant={compactMode ? "h6" : "h5"}>
                          {latestData.wind_speed} m/s
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </React.Fragment>
        )}
      </Grid>

      {/* Real-time Chart */}
      <Card sx={{ mt: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              {METRIC_CONFIGS.find(c => c.key === selectedMetric)?.label} Timeline
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              {METRIC_CONFIGS.map(config => (
                <Button
                  key={config.key}
                  size="small"
                  variant={selectedMetric === config.key ? 'contained' : 'outlined'}
                  onClick={() => setSelectedMetric(config.key)}
                  startIcon={config.icon}
                >
                  {config.label}
                </Button>
              ))}
            </Box>
          </Box>

          <Box sx={{ height: compactMode ? 200 : 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp_formatted" 
                  tick={{ fontSize: 12 }}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <RechartsTooltip 
                  labelFormatter={(value) => `Time: ${value}`}
                  formatter={(value, name) => [
                    typeof value === 'number' ? value.toFixed(2) : value,
                    name
                  ]}
                />
                <Area
                  type="monotone"
                  dataKey={selectedMetric}
                  stroke={METRIC_CONFIGS.find(c => c.key === selectedMetric)?.chartColor || '#8884d8'}
                  fill={METRIC_CONFIGS.find(c => c.key === selectedMetric)?.chartColor || '#8884d8'}
                  fillOpacity={0.6}
                  strokeWidth={2}
                />
                
                {/* Reference lines for thresholds */}
                {METRIC_CONFIGS.find(c => c.key === selectedMetric)?.thresholds.critical?.max && (
                  <ReferenceLine
                    y={METRIC_CONFIGS.find(c => c.key === selectedMetric)?.thresholds.critical?.max}
                    stroke="red"
                    strokeDasharray="5 5"
                    label="Critical Max"
                  />
                )}
                {METRIC_CONFIGS.find(c => c.key === selectedMetric)?.thresholds.critical?.min && (
                  <ReferenceLine
                    y={METRIC_CONFIGS.find(c => c.key === selectedMetric)?.thresholds.critical?.min}
                    stroke="red"
                    strokeDasharray="5 5"
                    label="Critical Min"
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>

      {/* Position and Mission Info */}
      {latestData && (
        <Card sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Current Position & Status
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Position Information
                  </Typography>
                  <Typography variant="body2">
                    <strong>Latitude:</strong> {latestData.latitude.toFixed(6)}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Longitude:</strong> {latestData.longitude.toFixed(6)}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Vertical Speed:</strong> {latestData.vertical_speed.toFixed(2)} m/s
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Mission Information
                  </Typography>
                  <Typography variant="body2">
                    <strong>Mission ID:</strong> {latestData.mission_id || 'None'}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Flight Mode:</strong> {latestData.flight_mode || 'Unknown'}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Payload Status:</strong> {latestData.payload_status || 'Unknown'}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Last Update:</strong> {format(new Date(latestData.timestamp), 'HH:mm:ss')}
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default RealTimeTelemetryDashboard;