import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  IconButton,
  Switch,
  FormControlLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Tooltip,
  TextField,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Download,
  Storage,
  AccessTime,
  Speed,
  Memory,
  CloudUpload,
  FilterList,
  Clear,
  Refresh,
} from '@mui/icons-material';
import { format } from 'date-fns';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';

interface DataLogEntry {
  id: number;
  timestamp: string;
  uav_id: number;
  mission_id?: number;
  data_type: 'telemetry' | 'sensor' | 'video' | 'target' | 'environmental' | 'system';
  data_source: string;
  data_payload: Record<string, any>;
  file_path?: string;
  file_size?: number;
  checksum?: string;
  logged_at: string;
}

interface LoggingSession {
  id: number;
  session_name: string;
  uav_id: number;
  mission_id?: number;
  start_time: string;
  end_time?: string;
  status: 'active' | 'paused' | 'stopped';
  total_entries: number;
  total_size_bytes: number;
  data_types: string[];
}

interface RealTimeDataLoggerProps {
  uavId?: number;
  missionId?: number;
  autoStart?: boolean;
}

const RealTimeDataLogger: React.FC<RealTimeDataLoggerProps> = ({
  uavId,
  missionId,
  autoStart = false
}) => {
  const [isLogging, setIsLogging] = useState(autoStart);
  const [sessionName, setSessionName] = useState(`Session_${format(new Date(), 'yyyyMMdd_HHmmss')}`);
  const [logLevel, setLogLevel] = useState<'all' | 'critical' | 'warnings' | 'errors'>('all');
  const [dataTypes, setDataTypes] = useState<string[]>(['telemetry', 'sensor', 'environmental']);
  const [compressionEnabled, setCompressionEnabled] = useState(true);
  const [realTimeDisplay, setRealTimeDisplay] = useState(true);
  const [maxDisplayEntries, setMaxDisplayEntries] = useState(100);
  const [bufferSize, setBufferSize] = useState(1000);
  
  const queryClient = useQueryClient();
  const logBuffer = useRef<DataLogEntry[]>([]);
  const sessionId = useRef<number | null>(null);

  // Current logging session
  const { data: currentSession, isLoading: sessionLoading } = useQuery({
    queryKey: ['logging-session', sessionId.current],
    queryFn: async () => {
      if (!sessionId.current) return null;
      const response = await axios.get(`/api/data-logging/sessions/${sessionId.current}`);
      return response.data.data as LoggingSession;
    },
    enabled: !!sessionId.current,
    refetchInterval: isLogging ? 5000 : false,
  });

  // Recent log entries
  const { data: logEntries = [], isLoading: entriesLoading } = useQuery({
    queryKey: ['log-entries', sessionId.current, realTimeDisplay],
    queryFn: async () => {
      if (!sessionId.current) return [];
      const params = new URLSearchParams();
      params.append('session_id', sessionId.current.toString());
      params.append('limit', maxDisplayEntries.toString());
      if (logLevel !== 'all') params.append('level', logLevel);
      
      const response = await axios.get(`/api/data-logging/entries?${params}`);
      return response.data.data as DataLogEntry[];
    },
    enabled: !!sessionId.current && realTimeDisplay,
    refetchInterval: isLogging && realTimeDisplay ? 2000 : false,
  });

  // Start logging session
  const startLoggingMutation = useMutation({
    mutationFn: async (config: {
      session_name: string;
      uav_id?: number;
      mission_id?: number;
      data_types: string[];
      log_level: string;
      buffer_size: number;
      compression: boolean;
    }) => {
      const response = await axios.post('/api/data-logging/sessions', config);
      return response.data;
    },
    onSuccess: (data) => {
      sessionId.current = data.session_id;
      setIsLogging(true);
      queryClient.invalidateQueries({ queryKey: ['logging-session'] });
    },
    onError: (error) => {
      console.error('Failed to start logging session:', error);
      alert('Failed to start logging session');
    },
  });

  // Stop logging session
  const stopLoggingMutation = useMutation({
    mutationFn: async (sessionId: number) => {
      const response = await axios.post(`/api/data-logging/sessions/${sessionId}/stop`);
      return response.data;
    },
    onSuccess: () => {
      setIsLogging(false);
      queryClient.invalidateQueries({ queryKey: ['logging-session'] });
    },
  });

  // Export log data
  const exportLogsMutation = useMutation({
    mutationFn: async (sessionId: number) => {
      const response = await axios.get(`/api/data-logging/sessions/${sessionId}/export`, {
        responseType: 'blob'
      });
      return response.data;
    },
    onSuccess: (data) => {
      // Create download link
      const url = window.URL.createObjectURL(new Blob([data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${sessionName}_${format(new Date(), 'yyyyMMdd_HHmmss')}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    },
  });

  const handleStartLogging = () => {
    if (!sessionName.trim()) {
      alert('Please enter a session name');
      return;
    }

    startLoggingMutation.mutate({
      session_name: sessionName,
      uav_id: uavId,
      mission_id: missionId,
      data_types: dataTypes,
      log_level: logLevel,
      buffer_size: bufferSize,
      compression: compressionEnabled,
    });
  };

  const handleStopLogging = () => {
    if (sessionId.current) {
      stopLoggingMutation.mutate(sessionId.current);
    }
  };

  const handlePauseLogging = () => {
    setIsLogging(!isLogging);
    // In a real implementation, this would pause the logging on the backend
  };

  const handleExportLogs = () => {
    if (sessionId.current) {
      exportLogsMutation.mutate(sessionId.current);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getDataTypeColor = (type: string): 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info' => {
    switch (type) {
      case 'telemetry': return 'primary';
      case 'sensor': return 'success';
      case 'environmental': return 'warning';
      case 'target': return 'error';
      case 'video': return 'secondary';
      case 'system': return 'info';
      default: return 'primary';
    }
  };

  // Auto-start logging if requested
  useEffect(() => {
    if (autoStart && !isLogging && !sessionId.current) {
      handleStartLogging();
    }
  }, [autoStart]);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        <Storage sx={{ mr: 1, verticalAlign: 'middle' }} />
        Real-time Data Logging System
      </Typography>

      {/* Session Status */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color={isLogging ? 'success' : 'text.secondary'}>
                {isLogging ? <PlayArrow /> : <Pause />}
              </Typography>
              <Typography color="textSecondary">
                {isLogging ? 'Recording' : 'Stopped'}
              </Typography>
              <Chip 
                label={isLogging ? 'LIVE' : 'IDLE'}
                color={isLogging ? 'success' : 'default'}
                size="small"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {currentSession?.total_entries || 0}
              </Typography>
              <Typography color="textSecondary">
                Log Entries
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info">
                {formatBytes(currentSession?.total_size_bytes || 0)}
              </Typography>
              <Typography color="textSecondary">
                Data Size
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning">
                {currentSession ? format(new Date(currentSession.start_time), 'HH:mm:ss') : '--:--:--'}
              </Typography>
              <Typography color="textSecondary">
                Session Start
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Logging Configuration
          </Typography>
          
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                label="Session Name"
                value={sessionName}
                onChange={(e) => setSessionName(e.target.value)}
                size="small"
                disabled={isLogging}
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Log Level</InputLabel>
                <Select
                  value={logLevel}
                  label="Log Level"
                  onChange={(e) => setLogLevel(e.target.value as any)}
                  disabled={isLogging}
                >
                  <MenuItem value="all">All Data</MenuItem>
                  <MenuItem value="critical">Critical Only</MenuItem>
                  <MenuItem value="warnings">Warnings+</MenuItem>
                  <MenuItem value="errors">Errors Only</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Buffer Size</InputLabel>
                <Select
                  value={bufferSize}
                  label="Buffer Size"
                  onChange={(e) => setBufferSize(Number(e.target.value))}
                  disabled={isLogging}
                >
                  <MenuItem value={500}>500 entries</MenuItem>
                  <MenuItem value={1000}>1000 entries</MenuItem>
                  <MenuItem value={5000}>5000 entries</MenuItem>
                  <MenuItem value={10000}>10000 entries</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={compressionEnabled}
                      onChange={(e) => setCompressionEnabled(e.target.checked)}
                      disabled={isLogging}
                    />
                  }
                  label="Compression"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={realTimeDisplay}
                      onChange={(e) => setRealTimeDisplay(e.target.checked)}
                    />
                  }
                  label="Real-time View"
                />
              </Box>
            </Grid>
          </Grid>

          {/* Data Types Selection */}
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Data Types to Log:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {['telemetry', 'sensor', 'environmental', 'target', 'video', 'system'].map((type) => (
                <Chip
                  key={type}
                  label={type}
                  color={dataTypes.includes(type) ? getDataTypeColor(type) : 'default'}
                  onClick={() => {
                    if (isLogging) return;
                    setDataTypes(prev => 
                      prev.includes(type) 
                        ? prev.filter(t => t !== type)
                        : [...prev, type]
                    );
                  }}
                  variant={dataTypes.includes(type) ? 'filled' : 'outlined'}
                  disabled={isLogging}
                />
              ))}
            </Box>
          </Box>

          {/* Control Buttons */}
          <Box sx={{ mt: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {!isLogging ? (
              <Button
                variant="contained"
                startIcon={<PlayArrow />}
                onClick={handleStartLogging}
                disabled={startLoggingMutation.isPending || dataTypes.length === 0}
              >
                Start Logging
              </Button>
            ) : (
              <>
                <Button
                  variant="outlined"
                  startIcon={<Pause />}
                  onClick={handlePauseLogging}
                >
                  {isLogging ? 'Pause' : 'Resume'}
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Stop />}
                  onClick={handleStopLogging}
                  disabled={stopLoggingMutation.isPending}
                >
                  Stop Logging
                </Button>
              </>
            )}
            
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={handleExportLogs}
              disabled={!currentSession || exportLogsMutation.isPending}
            >
              Export Data
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<CloudUpload />}
              disabled={!currentSession}
            >
              Upload to Cloud
            </Button>

            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={() => queryClient.invalidateQueries()}
            >
              Refresh
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Active Logging Indicator */}
      {isLogging && (
        <Alert severity="success" sx={{ mb: 3 }}>
          <strong>Data logging is active:</strong> Recording {dataTypes.join(', ')} data to session "{sessionName}".
          All data is being timestamped and stored with integrity checksums.
        </Alert>
      )}

      {/* Progress Indicators */}
      {(sessionLoading || entriesLoading) && <LinearProgress sx={{ mb: 2 }} />}

      {/* Recent Log Entries */}
      {realTimeDisplay && (
        <Paper>
          <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h6">
                <AccessTime sx={{ mr: 1, verticalAlign: 'middle' }} />
                Recent Log Entries
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Showing last {Math.min(maxDisplayEntries, logEntries.length)} entries
                </Typography>
                <FormControl size="small">
                  <Select
                    value={maxDisplayEntries}
                    onChange={(e) => setMaxDisplayEntries(Number(e.target.value))}
                  >
                    <MenuItem value={50}>50</MenuItem>
                    <MenuItem value={100}>100</MenuItem>
                    <MenuItem value={250}>250</MenuItem>
                    <MenuItem value={500}>500</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </Box>
          </Box>

          <TableContainer sx={{ maxHeight: 400 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Source</TableCell>
                  <TableCell>UAV</TableCell>
                  <TableCell>Data Size</TableCell>
                  <TableCell>Checksum</TableCell>
                  <TableCell>Preview</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {logEntries.map((entry) => (
                  <TableRow key={entry.id} hover>
                    <TableCell>
                      <Tooltip title={format(new Date(entry.timestamp), 'PPpp')}>
                        <span>{format(new Date(entry.timestamp), 'HH:mm:ss.SSS')}</span>
                      </Tooltip>
                    </TableCell>
                    
                    <TableCell>
                      <Chip
                        label={entry.data_type}
                        color={getDataTypeColor(entry.data_type)}
                        size="small"
                      />
                    </TableCell>
                    
                    <TableCell>
                      <Typography variant="body2" noWrap>
                        {entry.data_source}
                      </Typography>
                    </TableCell>
                    
                    <TableCell>
                      UAV-{entry.uav_id}
                    </TableCell>
                    
                    <TableCell>
                      {entry.file_size ? formatBytes(entry.file_size) : 
                       formatBytes(JSON.stringify(entry.data_payload).length)}
                    </TableCell>
                    
                    <TableCell>
                      <Tooltip title={entry.checksum}>
                        <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                          {entry.checksum ? entry.checksum.substring(0, 8) + '...' : 'N/A'}
                        </Typography>
                      </Tooltip>
                    </TableCell>
                    
                    <TableCell>
                      <Tooltip title={JSON.stringify(entry.data_payload, null, 2)}>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                          {Object.keys(entry.data_payload).join(', ')}
                        </Typography>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {logEntries.length === 0 && !entriesLoading && (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="body1" color="text.secondary">
                {isLogging ? 'Waiting for log entries...' : 'No log entries available. Start logging to see data.'}
              </Typography>
            </Box>
          )}
        </Paper>
      )}

      {/* Session Information */}
      {currentSession && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Current Session Details
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="body2"><strong>Session Name:</strong> {currentSession.session_name}</Typography>
                <Typography variant="body2"><strong>Status:</strong> {currentSession.status}</Typography>
                <Typography variant="body2"><strong>Start Time:</strong> {format(new Date(currentSession.start_time), 'PPpp')}</Typography>
                {currentSession.end_time && (
                  <Typography variant="body2"><strong>End Time:</strong> {format(new Date(currentSession.end_time), 'PPpp')}</Typography>
                )}
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="body2"><strong>Total Entries:</strong> {currentSession.total_entries.toLocaleString()}</Typography>
                <Typography variant="body2"><strong>Data Size:</strong> {formatBytes(currentSession.total_size_bytes)}</Typography>
                <Typography variant="body2"><strong>Data Types:</strong> {currentSession.data_types.join(', ')}</Typography>
                <Typography variant="body2"><strong>UAV ID:</strong> {currentSession.uav_id}</Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default RealTimeDataLogger;