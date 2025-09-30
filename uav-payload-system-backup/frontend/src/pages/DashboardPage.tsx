import React, { useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  Flight,
  Assignment,
  Inventory,
  Warning,
  CheckCircle,
  Error,
  Info,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { useSocket } from '../contexts/SocketContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';

interface DashboardStats {
  total_uavs: number;
  active_uavs: number;
  total_missions: number;
  active_missions: number;
  completed_missions_today: number;
  total_payloads: number;
  available_payloads: number;
  system_alerts: number;
}

interface UAVStatus {
  uav_id: number;
  serial_number: string;
  model: string;
  status: string;
  current_mission_id?: number;
  battery_level?: number;
  last_telemetry?: string;
  location?: {
    latitude: number;
    longitude: number;
    altitude: number;
  };
}

interface MissionSummary {
  mission_id: number;
  name: string;
  status: string;
  priority: string;
  uav_serial: string;
  progress_percentage: number;
  estimated_completion?: string;
}

const DashboardPage: React.FC = () => {
  const { joinTelemetryUpdates, latestTelemetry } = useSocket();

  useEffect(() => {
    joinTelemetryUpdates();
  }, [joinTelemetryUpdates]);

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const response = await axios.get('/api/dashboard/stats');
      return response.data.data as DashboardStats;
    },
    refetchInterval: 30000
  });

  const { data: uavStatus, isLoading: uavLoading } = useQuery({
    queryKey: ['uav-status'],
    queryFn: async () => {
      const response = await axios.get('/api/dashboard/uav-status');
      return response.data.data as UAVStatus[];
    },
    refetchInterval: 10000
  });

  const { data: missionSummary, isLoading: missionLoading } = useQuery({
    queryKey: ['mission-summary'],
    queryFn: async () => {
      const response = await axios.get('/api/dashboard/mission-summary');
      return response.data.data as MissionSummary[];
    },
    refetchInterval: 15000
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'default';
      case 'maintenance':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical':
        return 'error';
      case 'high':
        return 'warning';
      case 'medium':
        return 'info';
      case 'low':
        return 'default';
      default:
        return 'default';
    }
  };

  const formatTime = (dateString?: string) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleTimeString();
  };

  // Create chart data for mission types
  const missionTypeData = missionSummary?.reduce((acc, mission) => {
    const existingType = acc.find(item => item.type === mission.status);
    if (existingType) {
      existingType.count += 1;
    } else {
      acc.push({ type: mission.status, count: 1 });
    }
    return acc;
  }, [] as { type: string; count: number }[]) || [];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Dashboard
      </Typography>

      {/* System Status - Simplified */}
      <Alert severity="info" sx={{ mb: 3 }}>
        System operating normally. All UAVs connected and reporting telemetry.
      </Alert>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Flight sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    UAVs Active
                  </Typography>
                  <Typography variant="h4">
                    {statsLoading ? '-' : `${stats?.active_uavs}/${stats?.total_uavs}`}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Assignment sx={{ fontSize: 40, color: 'secondary.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Active Missions
                  </Typography>
                  <Typography variant="h4">
                    {statsLoading ? '-' : stats?.active_missions}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Inventory sx={{ fontSize: 40, color: 'warning.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Available Payloads
                  </Typography>
                  <Typography variant="h4">
                    {statsLoading ? '-' : `${stats?.available_payloads}/${stats?.total_payloads}`}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <CheckCircle sx={{ fontSize: 40, color: 'success.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Completed Today
                  </Typography>
                  <Typography variant="h4">
                    {statsLoading ? '-' : stats?.completed_missions_today}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts and Lists */}
      <Grid container spacing={3}>
        {/* Mission Status Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Mission Status Overview
              </Typography>
              {missionLoading ? (
                <LinearProgress />
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={missionTypeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="count" fill="#1976d2" />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* UAV Status List */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                UAV Fleet Status
              </Typography>
              {uavLoading ? (
                <LinearProgress />
              ) : (
                <List>
                  {uavStatus?.map((uav) => (
                    <ListItem key={uav.uav_id}>
                      <ListItemIcon>
                        <Flight color={getStatusColor(uav.status) as any} />
                      </ListItemIcon>
                      <ListItemText
                        primary={`${uav.serial_number} - ${uav.model}`}
                        secondary={
                          <Box>
                            <Chip 
                              label={uav.status} 
                              color={getStatusColor(uav.status) as any}
                              size="small" 
                              sx={{ mr: 1 }}
                            />
                            {uav.battery_level && (
                              <Typography variant="body2" component="span">
                                Battery: {uav.battery_level}%
                              </Typography>
                            )}
                            {uav.last_telemetry && (
                              <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                                Last Update: {formatTime(uav.last_telemetry)}
                              </Typography>
                            )}
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Active Missions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Missions
              </Typography>
              {missionLoading ? (
                <LinearProgress />
              ) : (
                <List>
                  {missionSummary?.filter(mission => mission.status === 'active').map((mission) => (
                    <ListItem key={mission.mission_id}>
                      <ListItemIcon>
                        <Assignment color="primary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={mission.name}
                        secondary={
                          <Box>
                            <Chip 
                              label={mission.priority} 
                              color={getPriorityColor(mission.priority) as any}
                              size="small" 
                              sx={{ mr: 1 }}
                            />
                            <Typography variant="body2" component="span">
                              UAV: {mission.uav_serial} | Progress: {mission.progress_percentage.toFixed(1)}%
                            </Typography>
                            {mission.estimated_completion && (
                              <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                                ETA: {formatTime(mission.estimated_completion)}
                              </Typography>
                            )}
                          </Box>
                        }
                      />
                      <Box sx={{ width: 100 }}>
                        <LinearProgress 
                          variant="determinate" 
                          value={mission.progress_percentage} 
                        />
                      </Box>
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
