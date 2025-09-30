import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  Alert,
  Button,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  GpsFixed,
  Warning,
  CheckCircle,
  Error as ErrorIcon,
  Visibility,
  VisibilityOff,
  LocationOn,
  AccessTime,
  Speed,
  Height,
  MyLocation,
  ZoomIn,
  FilterList,
  Notifications,
  NotificationsOff,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';
import axios from 'axios';

export interface DetectedTarget {
  id: number;
  uav_id: number;
  mission_id?: number;
  target_type: 'person' | 'vehicle' | 'structure' | 'unknown' | 'hazard';
  confidence: number;
  coordinates: {
    latitude: number;
    longitude: number;
    altitude: number;
  };
  bounding_box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  image_url?: string;
  detection_time: string;
  status: 'active' | 'confirmed' | 'false_positive' | 'investigating';
  priority: 'low' | 'medium' | 'high' | 'critical';
  description?: string;
  estimated_size?: {
    width: number;
    height: number;
  };
  movement_vector?: {
    speed: number;
    direction: number;
  };
}

interface TargetDetectionInterfaceProps {
  uavId?: number;
  missionId?: number;
  onTargetSelected?: (target: DetectedTarget) => void;
}

const TargetDetectionInterface: React.FC<TargetDetectionInterfaceProps> = ({
  uavId,
  missionId,
  onTargetSelected
}) => {
  const [selectedTarget, setSelectedTarget] = useState<DetectedTarget | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [priorityFilter, setPriorityFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const { data: targets = [], isLoading, refetch } = useQuery({
    queryKey: ['detected-targets', uavId, missionId, statusFilter, priorityFilter, typeFilter],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (uavId) params.append('uav_id', uavId.toString());
      if (missionId) params.append('mission_id', missionId.toString());
      if (statusFilter !== 'all') params.append('status', statusFilter);
      if (priorityFilter !== 'all') params.append('priority', priorityFilter);
      if (typeFilter !== 'all') params.append('type', typeFilter);
      
      const response = await axios.get(`/api/target-detection?${params}`);
      return response.data.data as DetectedTarget[];
    },
    refetchInterval: autoRefresh ? 3000 : false, // Refresh every 3 seconds for real-time detection
  });

  // Filter targets based on current filters
  const filteredTargets = targets.filter(target => {
    if (statusFilter !== 'all' && target.status !== statusFilter) return false;
    if (priorityFilter !== 'all' && target.priority !== priorityFilter) return false;
    if (typeFilter !== 'all' && target.target_type !== typeFilter) return false;
    return true;
  });

  const handleTargetClick = (target: DetectedTarget) => {
    setSelectedTarget(target);
    onTargetSelected?.(target);
  };

  const getTargetTypeIcon = (type: string) => {
    switch (type) {
      case 'person': return '👤';
      case 'vehicle': return '🚗';
      case 'structure': return '🏢';
      case 'hazard': return '⚠️';
      default: return '❓';
    }
  };

  const getStatusColor = (status: string): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (status) {
      case 'active': return 'primary';
      case 'confirmed': return 'success';
      case 'false_positive': return 'error';
      case 'investigating': return 'warning';
      default: return 'default';
    }
  };

  const getPriorityColor = (priority: string): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (priority) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'default';
      default: return 'default';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const formatCoordinates = (coords: DetectedTarget['coordinates']) => {
    return `${coords.latitude.toFixed(6)}, ${coords.longitude.toFixed(6)} (${coords.altitude.toFixed(1)}m)`;
  };

  const recentTargets = targets.filter(target => 
    new Date(target.detection_time) > new Date(Date.now() - 60000) // Last minute
  );

  const highPriorityTargets = targets.filter(target => 
    target.priority === 'critical' || target.priority === 'high'
  );

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        <GpsFixed sx={{ mr: 1, verticalAlign: 'middle' }} />
        Target Detection System
      </Typography>

      {/* Statistics Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {targets.length}
              </Typography>
              <Typography color="textSecondary" variant="body2">
                Total Targets
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="error">
                {highPriorityTargets.length}
              </Typography>
              <Typography color="textSecondary" variant="body2">
                High Priority
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="secondary">
                {recentTargets.length}
              </Typography>
              <Typography color="textSecondary" variant="body2">
                Recent (1m)
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success">
                {targets.filter(t => t.status === 'confirmed').length}
              </Typography>
              <Typography color="textSecondary" variant="body2">
                Confirmed
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Alerts */}
      {alertsEnabled && highPriorityTargets.length > 0 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <strong>{highPriorityTargets.length} high-priority target(s) detected!</strong>
          {' '}Immediate attention required for mission safety.
        </Alert>
      )}

      {recentTargets.length > 0 && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {recentTargets.length} new target(s) detected in the last minute.
        </Alert>
      )}

      {/* Controls */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Status</InputLabel>
              <Select
                value={statusFilter}
                label="Status"
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <MenuItem value="all">All Status</MenuItem>
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="confirmed">Confirmed</MenuItem>
                <MenuItem value="investigating">Investigating</MenuItem>
                <MenuItem value="false_positive">False Positive</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Priority</InputLabel>
              <Select
                value={priorityFilter}
                label="Priority"
                onChange={(e) => setPriorityFilter(e.target.value)}
              >
                <MenuItem value="all">All Priority</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="low">Low</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Type</InputLabel>
              <Select
                value={typeFilter}
                label="Type"
                onChange={(e) => setTypeFilter(e.target.value)}
              >
                <MenuItem value="all">All Types</MenuItem>
                <MenuItem value="person">Person</MenuItem>
                <MenuItem value="vehicle">Vehicle</MenuItem>
                <MenuItem value="structure">Structure</MenuItem>
                <MenuItem value="hazard">Hazard</MenuItem>
                <MenuItem value="unknown">Unknown</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={3}>
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
              }
              label="Auto Refresh"
            />
          </Grid>

          <Grid item xs={12} md={3}>
            <FormControlLabel
              control={
                <Switch
                  checked={alertsEnabled}
                  onChange={(e) => setAlertsEnabled(e.target.checked)}
                />
              }
              label={alertsEnabled ? "Alerts On" : "Alerts Off"}
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Loading Indicator */}
      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Targets Table */}
      <Paper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Target</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Confidence</TableCell>
                <TableCell>Priority</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Location</TableCell>
                <TableCell>Detection Time</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredTargets.map((target) => (
                <TableRow
                  key={target.id}
                  hover
                  onClick={() => handleTargetClick(target)}
                  sx={{ 
                    cursor: 'pointer',
                    backgroundColor: selectedTarget?.id === target.id ? 'rgba(25, 118, 210, 0.08)' : 'inherit'
                  }}
                >
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <span style={{ fontSize: '1.2em' }}>
                        {getTargetTypeIcon(target.target_type)}
                      </span>
                      <Typography variant="body2">
                        Target #{target.id}
                      </Typography>
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Chip 
                      label={target.target_type}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      label={`${(target.confidence * 100).toFixed(1)}%`}
                      color={getConfidenceColor(target.confidence)}
                      size="small"
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      label={target.priority.toUpperCase()}
                      color={getPriorityColor(target.priority)}
                      size="small"
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      label={target.status}
                      color={getStatusColor(target.status)}
                      size="small"
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Tooltip title={formatCoordinates(target.coordinates)}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LocationOn fontSize="small" color="action" />
                        <Typography variant="body2">
                          {target.coordinates.latitude.toFixed(4)}, {target.coordinates.longitude.toFixed(4)}
                        </Typography>
                      </Box>
                    </Tooltip>
                  </TableCell>
                  
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <AccessTime fontSize="small" color="action" />
                      <Typography variant="body2">
                        {format(new Date(target.detection_time), 'HH:mm:ss')}
                      </Typography>
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Tooltip title="View Details">
                      <IconButton size="small" onClick={() => handleTargetClick(target)}>
                        <ZoomIn />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Track Location">
                      <IconButton size="small">
                        <MyLocation />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {filteredTargets.length === 0 && !isLoading && (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography variant="body1" color="textSecondary">
              No targets detected matching current filters
            </Typography>
            <Button 
              variant="outlined" 
              onClick={() => refetch()}
              sx={{ mt: 2 }}
            >
              Refresh Detection
            </Button>
          </Box>
        )}
      </Paper>

      {/* Selected Target Details */}
      {selectedTarget && (
        <Paper sx={{ mt: 3, p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Target Details - #{selectedTarget.id}
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>Basic Information</Typography>
              <Typography variant="body2"><strong>Type:</strong> {selectedTarget.target_type}</Typography>
              <Typography variant="body2"><strong>Confidence:</strong> {(selectedTarget.confidence * 100).toFixed(1)}%</Typography>
              <Typography variant="body2"><strong>Priority:</strong> {selectedTarget.priority}</Typography>
              <Typography variant="body2"><strong>Status:</strong> {selectedTarget.status}</Typography>
              <Typography variant="body2"><strong>Detection Time:</strong> {format(new Date(selectedTarget.detection_time), 'PPpp')}</Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>Location Information</Typography>
              <Typography variant="body2"><strong>Coordinates:</strong> {formatCoordinates(selectedTarget.coordinates)}</Typography>
              <Typography variant="body2"><strong>Bounding Box:</strong> {selectedTarget.bounding_box.width} x {selectedTarget.bounding_box.height}px</Typography>
              {selectedTarget.estimated_size && (
                <Typography variant="body2">
                  <strong>Estimated Size:</strong> {selectedTarget.estimated_size.width.toFixed(1)} x {selectedTarget.estimated_size.height.toFixed(1)}m
                </Typography>
              )}
              {selectedTarget.movement_vector && (
                <Typography variant="body2">
                  <strong>Movement:</strong> {selectedTarget.movement_vector.speed.toFixed(1)} m/s at {selectedTarget.movement_vector.direction}°
                </Typography>
              )}
            </Grid>
          </Grid>
          
          {selectedTarget.description && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Description</Typography>
              <Typography variant="body2">{selectedTarget.description}</Typography>
            </Box>
          )}
          
          {selectedTarget.image_url && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Captured Image</Typography>
              <img 
                src={selectedTarget.image_url} 
                alt={`Target ${selectedTarget.id}`}
                style={{ maxWidth: '100%', maxHeight: '300px', border: '1px solid #ddd', borderRadius: '4px' }}
              />
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
};

export default TargetDetectionInterface;