import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  IconButton,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Switch,
  FormControlLabel,
  LinearProgress,
  Tooltip,
  Divider,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  Build,
  Settings,
  Speed,
  Timer,
  Tune,
  CameraAlt,
  Sensors,
  Warning,
  CheckCircle,
  Error,
  RotateLeft,
  RotateRight,
  Power,
  PowerOff,
} from '@mui/icons-material';
import { useQuery, useMutation } from '@tanstack/react-query';
import axios from 'axios';

interface HardwareStatus {
  sensors_online: boolean;
  camera_online: boolean;
  servo_online: boolean;
  last_reading_id?: number;
}

interface DrillingData {
  command?: string;
  duration?: number;
  timestamp?: string;
  status?: string;
  servo_position?: number;
  pressure_reading?: number;
  valve_state?: 'open' | 'closed';
  gauge_pressure?: number;
  target_pressure?: number;
  ready_to_drill?: boolean;
}

interface UAV {
  id: number;
  serial_number: string;
  model: string;
  status: string;
}

interface HardwareControlInterfaceProps {
  uavId?: number;
}

const HardwareControlInterface: React.FC<HardwareControlInterfaceProps> = ({
  uavId
}) => {
  const [selectedUAV, setSelectedUAV] = useState<number | null>(uavId || null);
  const [drillingDuration, setDrillingDuration] = useState(10);
  const [calibrationOpen, setCalibrationOpen] = useState(false);
  const [sensorType, setSensorType] = useState('all');

  const { data: uavs } = useQuery({
    queryKey: ['uavs'],
    queryFn: async () => {
      const response = await axios.get('/api/uavs');
      return response.data.data as UAV[];
    },
  });

  const { data: hardwareStatus, refetch: refetchStatus } = useQuery({
    queryKey: ['hardware-status', selectedUAV],
    queryFn: async () => {
      if (!selectedUAV) return null;
      const response = await axios.get(`/api/hardware/status?uav_id=${selectedUAV}`);
      return response.data.data;
    },
    enabled: !!selectedUAV,
    refetchInterval: 5000,
  });

  const { data: drillingData } = useQuery({
    queryKey: ['drilling-data', selectedUAV],
    queryFn: async () => {
      if (!selectedUAV) return null;
      const response = await axios.get(`/api/hardware/drilling?uav_id=${selectedUAV}`);
      return response.data.data as DrillingData | null;
    },
    enabled: !!selectedUAV,
    refetchInterval: 2000,
  });

  const drillingMutation = useMutation({
    mutationFn: async ({ action, duration }: { action: string; duration?: number }) => {
      if (!selectedUAV) throw new window.Error('No UAV selected');
      const response = await axios.post('/api/hardware/drilling/control', {
        uav_id: selectedUAV,
        action,
        duration: duration || drillingDuration,
      });
      return response.data;
    },
    onSuccess: () => {
      refetchStatus();
    },
  });

  const calibrationMutation = useMutation({
    mutationFn: async ({ sensor_type }: { sensor_type: string }) => {
      if (!selectedUAV) throw new window.Error('No UAV selected');
      const response = await axios.post('/api/hardware/calibrate', {
        uav_id: selectedUAV,
        sensor_type,
      });
      return response.data;
    },
    onSuccess: () => {
      setCalibrationOpen(false);
      refetchStatus();
    },
  });

  const handleDrillingControl = (action: string) => {
    drillingMutation.mutate({ 
      action, 
      duration: action !== 'stop' ? drillingDuration : undefined 
    });
  };

  const handleCalibrate = () => {
    calibrationMutation.mutate({ sensor_type: sensorType });
  };

  const getStatusIcon = (online: boolean) => {
    return online ? (
      <CheckCircle color="success" />
    ) : (
      <Error color="error" />
    );
  };

  const getStatusColor = (online: boolean) => {
    return online ? 'success' : 'error';
  };

  const getSystemHealthColor = () => {
    if (!hardwareStatus?.hardware_status) return 'default';
    
    const { sensors_online, camera_online, servo_online } = hardwareStatus.hardware_status;
    
    if (sensors_online && camera_online && servo_online) return 'success';
    if (sensors_online || camera_online || servo_online) return 'warning';
    return 'error';
  };

  const getDrillingStatusChip = () => {
    if (!drillingData) return <Chip label="No Data" color="default" size="small" />;
    
    const status = drillingData.status || 'idle';
    const colors: Record<string, 'warning' | 'primary' | 'success' | 'error' | 'default'> = {
      'pending': 'warning',
      'active': 'primary',
      'completed': 'success',
      'error': 'error',
      'idle': 'default'
    };

    return <Chip label={status.toUpperCase()} color={colors[status] || 'default'} size="small" />;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Hardware Control Interface
      </Typography>

      {/* UAV Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            UAV Selection
          </Typography>
          <FormControl fullWidth>
            <InputLabel>Select UAV</InputLabel>
            <Select
              value={selectedUAV || ''}
              onChange={(e) => setSelectedUAV(Number(e.target.value) || null)}
              label="Select UAV"
            >
              {uavs?.map((uav) => (
                <MenuItem key={uav.id} value={uav.id}>
                  {uav.serial_number} - {uav.model} ({uav.status})
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </CardContent>
      </Card>

      {selectedUAV && (
        <Grid container spacing={3}>
          {/* Hardware Status */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Hardware Status
                  </Typography>
                  <IconButton onClick={() => refetchStatus()} size="small">
                    <Refresh />
                  </IconButton>
                </Box>

                {hardwareStatus ? (
                  <Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Typography variant="body2" sx={{ mr: 2 }}>
                        System Health:
                      </Typography>
                      <Chip 
                        label={hardwareStatus.system_health?.toUpperCase() || 'UNKNOWN'} 
                        color={getSystemHealthColor()}
                        size="small"
                      />
                    </Box>

                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Sensors sx={{ mr: 1 }} />
                          <Typography variant="body2" sx={{ mr: 1 }}>
                            Sensors:
                          </Typography>
                          {getStatusIcon(hardwareStatus.hardware_status?.sensors_online)}
                        </Box>
                      </Grid>
                      <Grid item xs={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <CameraAlt sx={{ mr: 1 }} />
                          <Typography variant="body2" sx={{ mr: 1 }}>
                            Camera:
                          </Typography>
                          {getStatusIcon(hardwareStatus.hardware_status?.camera_online)}
                        </Box>
                      </Grid>
                      <Grid item xs={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Build sx={{ mr: 1 }} />
                          <Typography variant="body2" sx={{ mr: 1 }}>
                            Servo:
                          </Typography>
                          {getStatusIcon(hardwareStatus.hardware_status?.servo_online)}
                        </Box>
                      </Grid>
                      <Grid item xs={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Timer sx={{ mr: 1 }} />
                          <Typography variant="body2">
                            Last Update: {new Date(hardwareStatus.timestamp).toLocaleTimeString()}
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </Box>
                ) : (
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <CircularProgress size={20} sx={{ mr: 2 }} />
                    <Typography>Loading hardware status...</Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Drilling Control */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Drilling Control
                </Typography>

                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" gutterBottom>
                    Status: {getDrillingStatusChip()}
                  </Typography>
                  
                  {drillingData?.ready_to_drill && (
                    <Alert severity="success" sx={{ mt: 1 }}>
                      System ready for drilling operation
                    </Alert>
                  )}
                  
                  {drillingData?.valve_state && (
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Valve: <strong>{drillingData.valve_state.toUpperCase()}</strong>
                    </Typography>
                  )}
                  
                  {drillingData?.gauge_pressure !== undefined && (
                    <Typography variant="body2">
                      Pressure: <strong>{drillingData.gauge_pressure.toFixed(1)} PSI</strong>
                      {drillingData.target_pressure && (
                        <span> / {drillingData.target_pressure} PSI target</span>
                      )}
                    </Typography>
                  )}
                </Box>

                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom>
                    Duration: {drillingDuration} seconds
                  </Typography>
                  <Slider
                    value={drillingDuration}
                    onChange={(_, value) => setDrillingDuration(value as number)}
                    min={1}
                    max={60}
                    marks={[
                      { value: 5, label: '5s' },
                      { value: 10, label: '10s' },
                      { value: 30, label: '30s' },
                      { value: 60, label: '60s' },
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Box>

                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <Button
                      fullWidth
                      variant="contained"
                      color="success"
                      startIcon={<RotateRight />}
                      onClick={() => handleDrillingControl('start')}
                      disabled={drillingMutation.isPending}
                    >
                      Start
                    </Button>
                  </Grid>
                  <Grid item xs={4}>
                    <Button
                      fullWidth
                      variant="contained"
                      color="warning"
                      startIcon={<RotateLeft />}
                      onClick={() => handleDrillingControl('reverse')}
                      disabled={drillingMutation.isPending}
                    >
                      Reverse
                    </Button>
                  </Grid>
                  <Grid item xs={4}>
                    <Button
                      fullWidth
                      variant="contained"
                      color="error"
                      startIcon={<Stop />}
                      onClick={() => handleDrillingControl('stop')}
                      disabled={drillingMutation.isPending}
                    >
                      Stop
                    </Button>
                  </Grid>
                </Grid>

                {drillingMutation.isPending && (
                  <LinearProgress sx={{ mt: 2 }} />
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* System Controls */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Controls
                </Typography>

                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6} md={3}>
                    <Button
                      fullWidth
                      variant="outlined"
                      startIcon={<Tune />}
                      onClick={() => setCalibrationOpen(true)}
                    >
                      Calibrate Sensors
                    </Button>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Button
                      fullWidth
                      variant="outlined"
                      startIcon={<Refresh />}
                      onClick={() => refetchStatus()}
                    >
                      Refresh Status
                    </Button>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Calibration Dialog */}
      <Dialog open={calibrationOpen} onClose={() => setCalibrationOpen(false)}>
        <DialogTitle>Sensor Calibration</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Sensor Type</InputLabel>
            <Select
              value={sensorType}
              onChange={(e) => setSensorType(e.target.value)}
              label="Sensor Type"
            >
              <MenuItem value="all">All Sensors</MenuItem>
              <MenuItem value="gas">Gas Sensors</MenuItem>
              <MenuItem value="environmental">Environmental Sensors</MenuItem>
              <MenuItem value="camera">Camera</MenuItem>
            </Select>
          </FormControl>
          <Typography variant="body2" sx={{ mt: 2 }}>
            This will initiate the calibration process for the selected sensors. 
            Ensure the environment is stable during calibration.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCalibrationOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleCalibrate}
            disabled={calibrationMutation.isPending}
            variant="contained"
          >
            {calibrationMutation.isPending ? 'Calibrating...' : 'Start Calibration'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default HardwareControlInterface;