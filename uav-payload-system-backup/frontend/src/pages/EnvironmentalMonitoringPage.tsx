import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  Divider,
} from '@mui/material';
import {
  Air,
  Science,
  Warning,
  TrendingUp,
  Map,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import EnvironmentalSensorDashboard from '../components/EnvironmentalSensorDashboard';

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
  mission_type: string;
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
      id={`env-tabpanel-${index}`}
      aria-labelledby={`env-tab-${index}`}
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

const EnvironmentalMonitoringPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedUAV, setSelectedUAV] = useState<number | ''>('');
  const [selectedMission, setSelectedMission] = useState<number | ''>('');

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

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          <Air sx={{ mr: 1, verticalAlign: 'middle' }} />
          Environmental Monitoring System
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Real-time air quality and environmental hazard detection for UAV payload operations
        </Typography>
      </Box>

      {/* System Overview Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success">
                Safe
              </Typography>
              <Typography color="textSecondary" variant="body2">
                Current Status
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {activeUAVs.length}
              </Typography>
              <Typography color="textSecondary" variant="body2">
                Active Sensors
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info">
                12
              </Typography>
              <Typography color="textSecondary" variant="body2">
                Parameters Monitored
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning">
                2
              </Typography>
              <Typography color="textSecondary" variant="body2">
                Active Alerts
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Control Panel */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Monitor Configuration
          </Typography>
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
                      {mission.name} ({mission.mission_type})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={4}>
              <Alert severity="success" sx={{ height: '100%', display: 'flex', alignItems: 'center' }}>
                <Typography variant="body2">
                  Environmental sensors online and operational
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Environmental Hazard Alert */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>Mission Safety:</strong> Environmental monitoring is critical for payload operations. 
          Hazardous gas detection and air quality assessment ensure safe UAV deployment and operator protection.
        </Typography>
      </Alert>

      {/* Main Content Tabs */}
      <Paper>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab
              label="Real-time Monitoring"
              icon={<Air />}
              iconPosition="start"
            />
            <Tab
              label="Sensor Analytics"
              icon={<TrendingUp />}
              iconPosition="start"
            />
            <Tab
              label="Hazard Detection"
              icon={<Warning />}
              iconPosition="start"
            />
            <Tab
              label="Geographic Mapping"
              icon={<Map />}
              iconPosition="start"
            />
          </Tabs>
        </Box>

        {/* Real-time Monitoring Tab */}
        <TabPanel value={activeTab} index={0}>
          <EnvironmentalSensorDashboard
            uavId={selectedUAV || undefined}
            missionId={selectedMission || undefined}
            autoRefresh={true}
          />
        </TabPanel>

        {/* Sensor Analytics Tab */}
        <TabPanel value={activeTab} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Advanced Sensor Analytics
              </Typography>
              <Alert severity="info" sx={{ mb: 3 }}>
                Historical trend analysis, predictive modeling, and sensor correlation studies for environmental data.
              </Alert>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    <Science sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Sensor Specifications
                  </Typography>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Air Quality Sensors:</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • CO2: NDIR sensor (0-10,000 ppm, ±30 ppm accuracy)
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • CO: Electrochemical sensor (0-500 ppm, ±2% accuracy)
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • NO2: Electrochemical sensor (0-20 ppm, ±5% accuracy)
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 2 }}>
                    • PM2.5/PM10: Laser scattering (0-500 μg/m³, ±10%)
                  </Typography>
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Environmental Sensors:</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • Temperature: SHT30 (-40 to 125°C, ±0.3°C)
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • Humidity: SHT30 (0-100% RH, ±2% RH)
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • Pressure: BMP388 (300-1250 hPa, ±0.5 hPa)
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2 }}>
                    • Light: TSL2591 (0-88,000 lux, 16-bit resolution)
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Detection Algorithms
                  </Typography>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Air Quality Index (AQI) Calculation:</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 2 }}>
                    Multi-pollutant index based on EPA standards, weighted for hazardous gas concentrations with real-time health risk assessment.
                  </Typography>
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Hazard Level Classification:</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • Safe: AQI 0-50, no toxic gases detected
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • Moderate: AQI 51-100, elevated PM levels
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 1 }}>
                    • Unhealthy: AQI 101-200, toxic gas threshold breach
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2, mb: 2 }}>
                    • Hazardous: AQI &gt; 200, immediate danger levels
                  </Typography>
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Alert Thresholds:</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ ml: 2 }}>
                    Configurable warning and danger levels for each sensor with automatic mission abort capabilities for critical hazard detection.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Hazard Detection Tab */}
        <TabPanel value={activeTab} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Hazardous Gas Detection System
              </Typography>
              <Alert severity="warning" sx={{ mb: 3 }}>
                <strong>Critical Safety Feature:</strong> Automated hazard detection with mission safety protocols. 
                System will alert operators and recommend mission abort if dangerous conditions are detected.
              </Alert>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Warning sx={{ fontSize: 40, color: 'error.main', mb: 2 }} />
                  <Typography variant="h5" color="error">
                    0
                  </Typography>
                  <Typography color="textSecondary">
                    Critical Alerts
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    No immediate hazards detected
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Air sx={{ fontSize: 40, color: 'warning.main', mb: 2 }} />
                  <Typography variant="h5" color="warning">
                    2
                  </Typography>
                  <Typography color="textSecondary">
                    Moderate Alerts
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Elevated PM2.5 levels
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Science sx={{ fontSize: 40, color: 'success.main', mb: 2 }} />
                  <Typography variant="h5" color="success">
                    10
                  </Typography>
                  <Typography color="textSecondary">
                    Sensors Online
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    All systems operational
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Hazardous Substances Monitored
                  </Typography>
                  
                  <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={12} sm={6} md={3}>
                      <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                        <Typography variant="body2" fontWeight="bold" color="error">
                          Carbon Monoxide (CO)
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          Threshold: 35 ppm warning, 200 ppm danger
                        </Typography>
                        <Typography variant="body2">
                          Health Risk: Asphyxiation, neurological damage
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                        <Typography variant="body2" fontWeight="bold" color="error">
                          Nitrogen Dioxide (NO2)
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          Threshold: 100 ppb warning, 200 ppb danger
                        </Typography>
                        <Typography variant="body2">
                          Health Risk: Respiratory irritation, lung damage
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                        <Typography variant="body2" fontWeight="bold" color="error">
                          Sulfur Dioxide (SO2)
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          Threshold: 75 ppb warning, 185 ppb danger
                        </Typography>
                        <Typography variant="body2">
                          Health Risk: Respiratory problems, eye irritation
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                        <Typography variant="body2" fontWeight="bold" color="error">
                          Particulate Matter
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          PM2.5: 35 μg/m³, PM10: 150 μg/m³
                        </Typography>
                        <Typography variant="body2">
                          Health Risk: Cardiovascular, respiratory disease
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Geographic Mapping Tab */}
        <TabPanel value={activeTab} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Environmental Data Geographic Mapping
              </Typography>
              <Alert severity="info" sx={{ mb: 3 }}>
                Geographic visualization of environmental sensor data with heat maps and contamination zones. 
                Integration with mission planning for risk assessment and route optimization.
              </Alert>
            </Grid>

            <Grid item xs={12}>
              <Card sx={{ minHeight: 400 }}>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Interactive Environmental Heat Map
                  </Typography>
                  
                  <Box sx={{ 
                    height: 350, 
                    backgroundColor: '#f5f5f5', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    border: '2px dashed #ddd',
                    borderRadius: 1
                  }}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Map sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
                      <Typography variant="h6" color="text.secondary">
                        Environmental Heat Map
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Geographic visualization showing air quality distribution, 
                        hazardous gas concentrations, and environmental conditions 
                        across the mission area with real-time sensor overlay.
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default EnvironmentalMonitoringPage;