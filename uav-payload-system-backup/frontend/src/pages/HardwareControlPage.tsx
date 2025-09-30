import React, { useState } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
  Alert,
  Grid,
  Card,
  CardContent,
  Chip,
} from '@mui/material';
import {
  Build,
  Videocam,
  Sensors,
  Analytics,
  Settings,
  Warning,
  CheckCircle,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import HardwareControlInterface from '../components/HardwareControlInterface';
import VideoStreamingInterface from '../components/VideoStreamingInterface';
import EnvironmentalSensorDashboard from '../components/EnvironmentalSensorDashboard';
import TargetDetectionInterface from '../components/TargetDetectionInterface';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`hardware-tabpanel-${index}`}
      aria-labelledby={`hardware-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 0 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const HardwareControlPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedUAV, setSelectedUAV] = useState<number>(1); // Default to first UAV

  // Get overall hardware status for all UAVs
  const { data: systemStatus } = useQuery({
    queryKey: ['system-hardware-status'],
    queryFn: async () => {
      try {
        const response = await axios.get('/api/hardware/status');
        return response.data.data;
      } catch (error) {
        console.warn('Hardware status not available:', error);
        return null;
      }
    },
    refetchInterval: 10000,
  });

  // Get video streams status
  const { data: videoStreams } = useQuery({
    queryKey: ['video-streams-status'],
    queryFn: async () => {
      try {
        const response = await axios.get('/api/video/streams');
        return response.data.data || [];
      } catch (error) {
        console.warn('Video streams not available:', error);
        return [];
      }
    },
    refetchInterval: 15000,
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const getSystemHealthStatus = () => {
    if (!systemStatus) {
      return {
        status: 'unknown',
        color: 'default' as const,
        message: 'Hardware status unavailable'
      };
    }

    const health = systemStatus.system_health;
    
    if (health === 'healthy') {
      return {
        status: 'healthy',
        color: 'success' as const,
        message: 'All hardware systems operational'
      };
    } else if (health === 'degraded') {
      return {
        status: 'degraded',
        color: 'warning' as const,
        message: 'Some hardware systems may be offline'
      };
    } else {
      return {
        status: 'error',
        color: 'error' as const,
        message: 'Hardware systems not responding'
      };
    }
  };

  const getVideoStreamStatus = () => {
    if (!videoStreams || videoStreams.length === 0) {
      return {
        online: 0,
        total: 0,
        status: 'No streams configured'
      };
    }

    const online = videoStreams.filter((stream: any) => stream.status === 'online').length;
    const total = videoStreams.length;

    return {
      online,
      total,
      status: `${online}/${total} streams online`
    };
  };

  const systemHealth = getSystemHealthStatus();
  const videoStatus = getVideoStreamStatus();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Hardware Control Center
      </Typography>

      {/* System Overview */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          System Overview
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Build sx={{ mr: 1 }} />
                  <Typography variant="body2">Hardware Status</Typography>
                </Box>
                <Chip 
                  label={systemHealth.status.toUpperCase()}
                  color={systemHealth.color}
                  size="small"
                  icon={systemHealth.status === 'healthy' ? <CheckCircle /> : <Warning />}
                />
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  {systemHealth.message}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Videocam sx={{ mr: 1 }} />
                  <Typography variant="body2">Video Streams</Typography>
                </Box>
                <Chip 
                  label={videoStatus.status}
                  color={videoStatus.online === videoStatus.total ? 'success' : videoStatus.online > 0 ? 'warning' : 'error'}
                  size="small"
                />
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Camera feeds available
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Sensors sx={{ mr: 1 }} />
                  <Typography variant="body2">Environmental</Typography>
                </Box>
                <Chip 
                  label={systemStatus?.hardware_status?.sensors_online ? 'ONLINE' : 'OFFLINE'}
                  color={systemStatus?.hardware_status?.sensors_online ? 'success' : 'error'}
                  size="small"
                />
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Air quality & sensors
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Analytics sx={{ mr: 1 }} />
                  <Typography variant="body2">Target Detection</Typography>
                </Box>
                <Chip 
                  label={systemStatus?.hardware_status?.camera_online ? 'ACTIVE' : 'INACTIVE'}
                  color={systemStatus?.hardware_status?.camera_online ? 'success' : 'error'}
                  size="small"
                />
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  AI vision processing
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {!systemStatus && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            Hardware bridge service is not available. Some features may be limited to simulation mode.
          </Alert>
        )}
      </Paper>

      {/* Hardware Control Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="hardware control tabs">
            <Tab 
              label="Drilling Control" 
              icon={<Build />} 
              iconPosition="start"
              id="hardware-tab-0"
              aria-controls="hardware-tabpanel-0"
            />
            <Tab 
              label="Video Streaming" 
              icon={<Videocam />} 
              iconPosition="start"
              id="hardware-tab-1"
              aria-controls="hardware-tabpanel-1"
            />
            <Tab 
              label="Environmental Sensors" 
              icon={<Sensors />} 
              iconPosition="start"
              id="hardware-tab-2"
              aria-controls="hardware-tabpanel-2"
            />
            <Tab 
              label="Target Detection" 
              icon={<Analytics />} 
              iconPosition="start"
              id="hardware-tab-3"
              aria-controls="hardware-tabpanel-3"
            />
            <Tab 
              label="System Settings" 
              icon={<Settings />} 
              iconPosition="start"
              id="hardware-tab-4"
              aria-controls="hardware-tabpanel-4"
            />
          </Tabs>
        </Box>

        <TabPanel value={activeTab} index={0}>
          <HardwareControlInterface uavId={selectedUAV} />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <VideoStreamingInterface uavId={selectedUAV} />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <EnvironmentalSensorDashboard />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <TargetDetectionInterface />
        </TabPanel>

        <TabPanel value={activeTab} index={4}>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Hardware System Settings
            </Typography>
            
            <Alert severity="info">
              System configuration settings will be available here. This includes:
              <ul>
                <li>Hardware communication parameters</li>
                <li>Sensor calibration settings</li>
                <li>Video streaming configuration</li>
                <li>Safety thresholds and limits</li>
                <li>Data logging preferences</li>
              </ul>
            </Alert>

            {/* Future enhancement: Add hardware configuration forms */}
          </Box>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default HardwareControlPage;