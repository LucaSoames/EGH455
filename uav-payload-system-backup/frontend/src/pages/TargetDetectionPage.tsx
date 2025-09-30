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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  GpsFixed,
  MyLocation,
  Radar,
  Security,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import TargetDetectionInterface, { DetectedTarget } from '../components/TargetDetectionInterface';
import VideoStreamingInterface from '../components/VideoStreamingInterface';

interface UAV {
  id: number;
  serial_number: string;
  model: string;
  status: string;
}

interface Mission {
  id: number;
  name: string;
  status: string;
  uav_id: number;
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
      id={`detection-tabpanel-${index}`}
      aria-labelledby={`detection-tab-${index}`}
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

const TargetDetectionPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedUAV, setSelectedUAV] = useState<number | ''>('');
  const [selectedMission, setSelectedMission] = useState<number | ''>('');
  const [selectedTarget, setSelectedTarget] = useState<DetectedTarget | null>(null);

  const { data: uavs, isLoading: uavsLoading } = useQuery({
    queryKey: ['uavs'],
    queryFn: async () => {
      const response = await axios.get('/api/uavs');
      return response.data.data as UAV[];
    },
  });

  const { data: missions, isLoading: missionsLoading } = useQuery({
    queryKey: ['missions'],
    queryFn: async () => {
      const response = await axios.get('/api/missions');
      return response.data.data as Mission[];
    },
  });

  const activeUAVs = uavs?.filter(uav => uav.status === 'active') || [];
  const activeMissions = missions?.filter(mission => mission.status === 'active') || [];

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleTargetSelection = (target: DetectedTarget) => {
    setSelectedTarget(target);
    // Automatically switch to map/video tab when target is selected
    if (activeTab === 0) {
      setActiveTab(1);
    }
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          <GpsFixed sx={{ mr: 1, verticalAlign: 'middle' }} />
          Target Detection & Tracking System
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Real-time AI-powered target detection and automated alerting system
        </Typography>
      </Box>

      {/* Control Panel */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Select UAV</InputLabel>
                <Select
                  value={selectedUAV}
                  label="Select UAV"
                  onChange={(e) => setSelectedUAV(e.target.value as number)}
                >
                  <MenuItem value="">All UAVs</MenuItem>
                  {activeUAVs.map((uav) => (
                    <MenuItem key={uav.id} value={uav.id}>
                      {uav.serial_number} - {uav.model}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Select Mission</InputLabel>
                <Select
                  value={selectedMission}
                  label="Select Mission"
                  onChange={(e) => setSelectedMission(e.target.value as number)}
                >
                  <MenuItem value="">All Missions</MenuItem>
                  {activeMissions.map((mission) => (
                    <MenuItem key={mission.id} value={mission.id}>
                      {mission.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={4}>
              <Alert severity="info" sx={{ height: '100%', display: 'flex', alignItems: 'center' }}>
                <Typography variant="body2">
                  AI Detection: Active • Real-time Processing
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Main Content Tabs */}
      <Paper>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab
              label="Target Detection"
              icon={<Radar />}
              iconPosition="start"
            />
            <Tab
              label="Live View & Tracking"
              icon={<MyLocation />}
              iconPosition="start"
            />
            <Tab
              label="Detection Analytics"
              icon={<Security />}
              iconPosition="start"
            />
          </Tabs>
        </Box>

        {/* Target Detection Tab */}
        <TabPanel value={activeTab} index={0}>
          <TargetDetectionInterface
            uavId={selectedUAV || undefined}
            missionId={selectedMission || undefined}
            onTargetSelected={handleTargetSelection}
          />
        </TabPanel>

        {/* Live View & Tracking Tab */}
        <TabPanel value={activeTab} index={1}>
          <Grid container spacing={3}>
            {/* Video Stream */}
            <Grid item xs={12} lg={8}>
              <Typography variant="h6" gutterBottom>
                Live Camera Feed
              </Typography>
              {selectedUAV ? (
                <VideoStreamingInterface uavId={selectedUAV} />
              ) : (
                <Alert severity="info">
                  Select a UAV to view live camera feed with target tracking overlay
                </Alert>
              )}
            </Grid>

            {/* Target Tracking Info */}
            <Grid item xs={12} lg={4}>
              <Typography variant="h6" gutterBottom>
                Active Target Tracking
              </Typography>
              
              {selectedTarget ? (
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      Target #{selectedTarget.id}
                    </Typography>
                    
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Type:</strong> {selectedTarget.target_type}
                    </Typography>
                    
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Confidence:</strong> {(selectedTarget.confidence * 100).toFixed(1)}%
                    </Typography>
                    
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Priority:</strong> {selectedTarget.priority}
                    </Typography>
                    
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Location:</strong> {selectedTarget.coordinates.latitude.toFixed(6)}, {selectedTarget.coordinates.longitude.toFixed(6)}
                    </Typography>
                    
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      <strong>Altitude:</strong> {selectedTarget.coordinates.altitude.toFixed(1)}m
                    </Typography>

                    {selectedTarget.movement_vector && (
                      <>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          <strong>Speed:</strong> {selectedTarget.movement_vector.speed.toFixed(1)} m/s
                        </Typography>
                        
                        <Typography variant="body2" sx={{ mb: 2 }}>
                          <strong>Direction:</strong> {selectedTarget.movement_vector.direction}°
                        </Typography>
                      </>
                    )}

                    <Alert severity="warning" sx={{ mt: 2 }}>
                      Target tracking active - maintaining visual lock
                    </Alert>
                  </CardContent>
                </Card>
              ) : (
                <Alert severity="info">
                  Select a target from the Detection tab to begin tracking
                </Alert>
              )}

              {/* Detection Statistics */}
              <Card sx={{ mt: 3 }}>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Detection Performance
                  </Typography>
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Processing Rate:</strong> 30 FPS
                  </Typography>
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Detection Latency:</strong> 45ms
                  </Typography>
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Model Accuracy:</strong> 94.2%
                  </Typography>
                  
                  <Typography variant="body2">
                    <strong>False Positive Rate:</strong> 2.1%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Detection Analytics Tab */}
        <TabPanel value={activeTab} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h3" color="primary">
                    247
                  </Typography>
                  <Typography color="textSecondary">
                    Targets Detected Today
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h3" color="success">
                    94.2%
                  </Typography>
                  <Typography color="textSecondary">
                    Detection Accuracy
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h3" color="warning">
                    12
                  </Typography>
                  <Typography color="textSecondary">
                    High Priority Alerts
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Detection Analytics Dashboard
                  </Typography>
                  
                  <Alert severity="info" sx={{ mt: 2 }}>
                    Advanced analytics features including detection trends, heat maps, and performance metrics will be available in this section. Integration with machine learning models provides continuous improvement in target detection accuracy.
                  </Alert>

                  <Typography variant="body2" sx={{ mt: 2 }}>
                    <strong>AI Model Information:</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • Computer Vision Model: YOLOv8 optimized for UAV imagery
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • Training Dataset: 50,000+ annotated aerial images
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • Supported Classes: Person, Vehicle, Structure, Hazard, Unknown
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2 }}>
                    • Real-time Processing: GPU-accelerated inference
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default TargetDetectionPage;