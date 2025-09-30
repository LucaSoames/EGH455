import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Tabs,
  Tab,
  Alert,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  VideoCall,
  CameraAlt,
  DeviceThermostat,
  Videocam,
  ViewModule,
  Fullscreen,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import VideoStreamingInterface from '../components/VideoStreamingInterface';

interface UAV {
  id: number;
  serial_number: string;
  model: string;
  status: string;
}

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
      id={`video-tabpanel-${index}`}
      aria-labelledby={`video-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const VideoStreamingPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedUAVs, setSelectedUAVs] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<'single' | 'split' | 'grid'>('single');

  const { data: uavs, isLoading } = useQuery({
    queryKey: ['uavs'],
    queryFn: async () => {
      const response = await axios.get('/api/uavs');
      return response.data.data as UAV[];
    },
  });

  const activeUAVs = uavs?.filter(uav => uav.status === 'active') || [];

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleUAVSelection = (uavId: number) => {
    if (selectedUAVs.includes(uavId)) {
      setSelectedUAVs(selectedUAVs.filter(id => id !== uavId));
    } else {
      if (viewMode === 'single') {
        setSelectedUAVs([uavId]);
      } else {
        setSelectedUAVs([...selectedUAVs, uavId]);
      }
    }
  };

  const renderSingleView = () => {
    const selectedUAV = selectedUAVs[0];
    
    if (!selectedUAV) {
      return (
        <Alert severity="info" sx={{ mt: 2 }}>
          Select a UAV from the sidebar to view its video stream
        </Alert>
      );
    }

    return <VideoStreamingInterface uavId={selectedUAV} />;
  };

  const renderSplitView = () => {
    if (selectedUAVs.length === 0) {
      return (
        <Alert severity="info" sx={{ mt: 2 }}>
          Select up to 2 UAVs to view split screen video streams
        </Alert>
      );
    }

    return (
      <Grid container spacing={2}>
        {selectedUAVs.slice(0, 2).map((uavId) => (
          <Grid item xs={12} md={6} key={uavId}>
            <VideoStreamingInterface uavId={uavId} />
          </Grid>
        ))}
      </Grid>
    );
  };

  const renderGridView = () => {
    if (selectedUAVs.length === 0) {
      return (
        <Alert severity="info" sx={{ mt: 2 }}>
          Select up to 4 UAVs to view grid layout video streams
        </Alert>
      );
    }

    return (
      <Grid container spacing={2}>
        {selectedUAVs.slice(0, 4).map((uavId) => (
          <Grid item xs={12} sm={6} key={uavId}>
            <VideoStreamingInterface uavId={uavId} />
          </Grid>
        ))}
      </Grid>
    );
  };

  const renderViewContent = () => {
    switch (viewMode) {
      case 'single':
        return renderSingleView();
      case 'split':
        return renderSplitView();
      case 'grid':
        return renderGridView();
      default:
        return renderSingleView();
    }
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          <VideoCall sx={{ mr: 1, verticalAlign: 'middle' }} />
          Live Video Streaming
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Monitor live camera feeds from active UAVs with multi-camera support
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Sidebar - UAV Selection */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active UAVs
              </Typography>
              
              {/* View Mode Selection */}
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>View Mode</InputLabel>
                <Select
                  value={viewMode}
                  label="View Mode"
                  onChange={(e) => {
                    setViewMode(e.target.value as any);
                    setSelectedUAVs([]);
                  }}
                >
                  <MenuItem value="single">
                    <CameraAlt sx={{ mr: 1 }} />
                    Single View
                  </MenuItem>
                  <MenuItem value="split">
                    <ViewModule sx={{ mr: 1 }} />
                    Split View (2)
                  </MenuItem>
                  <MenuItem value="grid">
                    <ViewModule sx={{ mr: 1 }} />
                    Grid View (4)
                  </MenuItem>
                </Select>
              </FormControl>

              {isLoading ? (
                <Typography>Loading UAVs...</Typography>
              ) : activeUAVs.length === 0 ? (
                <Alert severity="warning">
                  No active UAVs available for video streaming
                </Alert>
              ) : (
                <Box>
                  {activeUAVs.map((uav) => {
                    const isSelected = selectedUAVs.includes(uav.id);
                    const maxSelections = viewMode === 'single' ? 1 : viewMode === 'split' ? 2 : 4;
                    const canSelect = selectedUAVs.length < maxSelections || isSelected;
                    
                    return (
                      <Button
                        key={uav.id}
                        fullWidth
                        variant={isSelected ? 'contained' : 'outlined'}
                        sx={{ mb: 1, justifyContent: 'flex-start' }}
                        onClick={() => handleUAVSelection(uav.id)}
                        disabled={!canSelect}
                        startIcon={<CameraAlt />}
                      >
                        <Box sx={{ textAlign: 'left', flex: 1 }}>
                          <Typography variant="body2" noWrap>
                            {uav.serial_number}
                          </Typography>
                          <Typography variant="caption" color="textSecondary" noWrap>
                            {uav.model}
                          </Typography>
                        </Box>
                      </Button>
                    );
                  })}
                  
                  {selectedUAVs.length > 0 && (
                    <Button
                      fullWidth
                      variant="outlined"
                      color="secondary"
                      sx={{ mt: 2 }}
                      onClick={() => setSelectedUAVs([])}
                    >
                      Clear Selection
                    </Button>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Camera Type Info */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Camera Types
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <CameraAlt color="primary" sx={{ mr: 1 }} />
                  <Typography variant="body2">Optical Camera</Typography>
                </Box>
                <Typography variant="caption" color="textSecondary">
                  High-resolution visible light imaging for target identification
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <DeviceThermostat color="warning" sx={{ mr: 1 }} />
                  <Typography variant="body2">Thermal Camera</Typography>
                </Box>
                <Typography variant="caption" color="textSecondary">
                  Infrared imaging for heat signature detection and night vision
                </Typography>
              </Box>

              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Videocam color="info" sx={{ mr: 1 }} />
                  <Typography variant="body2">Multispectral</Typography>
                </Box>
                <Typography variant="caption" color="textSecondary">
                  Multi-band imaging for environmental and agricultural analysis
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Main Video Area */}
        <Grid item xs={12} md={9}>
          <Paper sx={{ minHeight: 600 }}>
            {/* Camera Type Tabs */}
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={activeTab} onChange={handleTabChange}>
                <Tab
                  label="Optical"
                  icon={<CameraAlt />}
                  iconPosition="start"
                />
                <Tab
                  label="Thermal"
                  icon={<DeviceThermostat />}
                  iconPosition="start"
                />
                <Tab
                  label="Multispectral"
                  icon={<Videocam />}
                  iconPosition="start"
                />
              </Tabs>
            </Box>

            {/* Video Stream Content */}
            <TabPanel value={activeTab} index={0}>
              {renderViewContent()}
            </TabPanel>
            <TabPanel value={activeTab} index={1}>
              {renderViewContent()}
            </TabPanel>
            <TabPanel value={activeTab} index={2}>
              {renderViewContent()}
            </TabPanel>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default VideoStreamingPage;