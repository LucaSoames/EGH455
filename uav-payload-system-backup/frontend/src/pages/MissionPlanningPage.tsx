import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Divider,
  Tooltip,
  Backdrop,
  CircularProgress,
} from '@mui/material';
import {
  Add,
  Edit,
  Delete,
  PlayArrow,
  Save,
  FileCopy,
  Map,
  Assignment,
  Flight,
  Refresh,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { canCreate, canEdit, canDelete } from '../utils/permissions';
import { formatDistanceToNow } from 'date-fns';

interface MissionPlan {
  id?: number;
  name: string;
  description: string;
  waypoints: Waypoint[];
  uav_id?: number;
  payload_id?: number;
  max_altitude: number;
  max_speed: number;
  estimated_duration: number;
  estimated_distance: number;
  status: 'draft' | 'planned' | 'active' | 'completed' | 'aborted';
  priority: 'low' | 'medium' | 'high' | 'critical';
  created_at?: string;
  updated_at?: string;
  created_by?: string;
  geofence?: {
    center: [number, number];
    radius: number;
  };
  no_fly_zones?: Array<{
    center: [number, number];
    radius: number;
  }>;
}

interface Waypoint {
  id: string;
  latitude: number;
  longitude: number;
  altitude: number;
  speed?: number;
  action?: 'hover' | 'photo' | 'video' | 'scan' | 'deploy' | 'pickup';
  duration?: number;
  description?: string;
}

interface UAV {
  id: number;
  model: string;
  serial_number: string;
  status: string;
  max_altitude: number;
  max_speed: number;
  max_payload_weight: number;
}

interface Payload {
  id: number;
  name: string;
  type: string;
  weight: number;
  status: string;
  specifications?: any;
}

const MissionPlanningPage: React.FC = () => {
  const [selectedMission, setSelectedMission] = useState<MissionPlan | null>(null);
  const [showMapDialog, setShowMapDialog] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [editingMission, setEditingMission] = useState<MissionPlan | null>(null);
  const [formData, setFormData] = useState<Partial<MissionPlan>>({
    name: '',
    description: '',
    max_altitude: 120,
    max_speed: 15,
    priority: 'medium',
    waypoints: [],
  });

  const { user } = useAuth();
  const queryClient = useQueryClient();

  // Fetch mission plans
  const { data: missionPlans = [], isLoading: plansLoading, error: plansError } = useQuery({
    queryKey: ['mission-plans'],
    queryFn: async () => {
      const response = await axios.get('/api/missions');
      return response.data.data as MissionPlan[];
    },
    refetchInterval: 30000,
  });

  // Fetch available UAVs
  const { data: uavs = [], isLoading: uavsLoading } = useQuery({
    queryKey: ['available-uavs'],
    queryFn: async () => {
      const response = await axios.get('/api/uavs?status=available');
      return response.data.data as UAV[];
    },
  });

  // Fetch available payloads
  const { data: payloads = [], isLoading: payloadsLoading } = useQuery({
    queryKey: ['available-payloads'],
    queryFn: async () => {
      const response = await axios.get('/api/payloads?status=available');
      return response.data.data as Payload[];
    },
  });

  // Create mission mutation
  const createMissionMutation = useMutation({
    mutationFn: async (missionData: Partial<MissionPlan>) => {
      const response = await axios.post('/api/missions', missionData);
      return response.data;
    },
    onSuccess: () => {
      alert('Mission plan created successfully');
      queryClient.invalidateQueries({ queryKey: ['mission-plans'] });
      setShowCreateDialog(false);
      resetForm();
    },
    onError: (error: any) => {
      alert(error.response?.data?.message || 'Failed to create mission plan');
    },
  });

  // Update mission mutation
  const updateMissionMutation = useMutation({
    mutationFn: async ({ id, data }: { id: number; data: Partial<MissionPlan> }) => {
      const response = await axios.put(`/api/missions/${id}`, data);
      return response.data;
    },
    onSuccess: () => {
      alert('Mission plan updated successfully');
      queryClient.invalidateQueries({ queryKey: ['mission-plans'] });
      setEditingMission(null);
      resetForm();
    },
    onError: (error: any) => {
      alert(error.response?.data?.message || 'Failed to update mission plan');
    },
  });

  // Delete mission mutation
  const deleteMissionMutation = useMutation({
    mutationFn: async (missionId: number) => {
      await axios.delete(`/api/missions/${missionId}`);
    },
    onSuccess: () => {
      alert('Mission plan deleted successfully');
      queryClient.invalidateQueries({ queryKey: ['mission-plans'] });
    },
    onError: (error: any) => {
      alert(error.response?.data?.message || 'Failed to delete mission plan');
    },
  });

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      max_altitude: 120,
      max_speed: 15,
      priority: 'medium',
      waypoints: [],
    });
  };

  const handleCreateMission = () => {
    if (!user || !canCreate(user.role)) {
      alert('You do not have permission to create missions');
      return;
    }
    setEditingMission(null);
    resetForm();
    setShowCreateDialog(true);
  };

  const handleEditMission = (mission: MissionPlan) => {
    if (!user || !canEdit(user.role)) {
      alert('You do not have permission to edit missions');
      return;
    }
    setEditingMission(mission);
    setFormData(mission);
    setShowCreateDialog(true);
  };

  const handleDeleteMission = (missionId: number) => {
    if (!user || !canDelete(user.role)) {
      alert('You do not have permission to delete missions');
      return;
    }
    if (window.confirm('Are you sure you want to delete this mission plan?')) {
      deleteMissionMutation.mutate(missionId);
    }
  };

  const handleSaveMission = () => {
    if (!formData.name || !formData.description) {
      alert('Please fill in all required fields');
      return;
    }

    if (formData.waypoints && formData.waypoints.length < 2) {
      alert('Mission must have at least 2 waypoints');
      return;
    }

    if (editingMission) {
      updateMissionMutation.mutate({
        id: editingMission.id!,
        data: formData as MissionPlan,
      });
    } else {
      createMissionMutation.mutate(formData);
    }
  };

  const handlePlanWithMap = (mission?: MissionPlan) => {
    setSelectedMission(mission || null);
    setShowMapDialog(true);
  };

  const handleMapSave = useCallback((missionData: MissionPlan) => {
    if (selectedMission?.id) {
      updateMissionMutation.mutate({
        id: selectedMission.id,
        data: missionData,
      });
    } else {
      createMissionMutation.mutate(missionData);
    }
    setShowMapDialog(false);
  }, [selectedMission, updateMissionMutation, createMissionMutation]);

  const getMissionStatusColor = (status: string) => {
    switch (status) {
      case 'draft': return 'default';
      case 'planned': return 'info';
      case 'active': return 'success';
      case 'completed': return 'success';
      case 'aborted': return 'error';
      default: return 'default';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'default';
      default: return 'default';
    }
  };

  if (plansLoading || uavsLoading || payloadsLoading) {
    return (
      <Backdrop open={true} sx={{ color: '#fff', zIndex: 1 }}>
        <CircularProgress color="inherit" />
      </Backdrop>
    );
  }

  if (plansError) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Failed to load mission plans. Please refresh the page or check your connection.
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Mission Planning
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Create and manage UAV mission plans with interactive mapping
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<Map />}
            onClick={() => handlePlanWithMap()}
            disabled={!user || !canCreate(user.role)}
          >
            Plan with Map
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={handleCreateMission}
            disabled={!user || !canCreate(user.role)}
          >
            Create Mission
          </Button>
        </Box>
      </Box>

      {/* Statistics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Assignment sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" variant="body2">
                    Total Plans
                  </Typography>
                  <Typography variant="h4">
                    {missionPlans.length}
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
                <PlayArrow sx={{ fontSize: 40, color: 'success.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" variant="body2">
                    Active Missions
                  </Typography>
                  <Typography variant="h4">
                    {missionPlans.filter(m => m.status === 'active').length}
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
                <Flight sx={{ fontSize: 40, color: 'info.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" variant="body2">
                    Available UAVs
                  </Typography>
                  <Typography variant="h4">
                    {uavs.length}
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
                <Save sx={{ fontSize: 40, color: 'warning.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" variant="body2">
                    Draft Plans
                  </Typography>
                  <Typography variant="h4">
                    {missionPlans.filter(m => m.status === 'draft').length}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Mission Plans List */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Mission Plans</Typography>
            <IconButton onClick={() => queryClient.invalidateQueries({ queryKey: ['mission-plans'] })}>
              <Refresh />
            </IconButton>
          </Box>
          
          {missionPlans.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Assignment sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
              <Typography variant="h6" color="textSecondary" gutterBottom>
                No Mission Plans
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                Create your first mission plan to get started
              </Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={handleCreateMission}
                disabled={!user || !canCreate(user.role)}
              >
                Create Mission Plan
              </Button>
            </Box>
          ) : (
            <List>
              {missionPlans.map((mission, index) => (
                <React.Fragment key={mission.id}>
                  <ListItem>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <Typography variant="subtitle1" component="div">
                            {mission.name}
                          </Typography>
                          <Chip
                            label={mission.status}
                            size="small"
                            color={getMissionStatusColor(mission.status) as any}
                            variant="outlined"
                          />
                          <Chip
                            label={mission.priority}
                            size="small"
                            color={getPriorityColor(mission.priority) as any}
                            variant="filled"
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            {mission.description}
                          </Typography>
                          <Typography variant="caption" display="block" color="textSecondary">
                            {mission.waypoints?.length || 0} waypoints • 
                            {Math.round(mission.estimated_distance / 1000)}km • 
                            {Math.round(mission.estimated_duration / 60)}min
                          </Typography>
                          <Typography variant="caption" display="block" color="textSecondary">
                            Created {mission.created_at ? formatDistanceToNow(new Date(mission.created_at)) : 'unknown'} ago
                          </Typography>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <Tooltip title="Plan with Map">
                          <IconButton
                            size="small"
                            onClick={() => handlePlanWithMap(mission)}
                          >
                            <Map />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit Mission">
                          <IconButton
                            size="small"
                            onClick={() => handleEditMission(mission)}
                            disabled={!user || !canEdit(user.role)}
                          >
                            <Edit />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Duplicate Mission">
                          <IconButton
                            size="small"
                            onClick={() => {
                              const duplicated = {
                                ...mission,
                                name: `${mission.name} (Copy)`,
                                id: undefined,
                                status: 'draft' as const,
                              };
                              createMissionMutation.mutate(duplicated);
                            }}
                            disabled={!user || !canCreate(user.role)}
                          >
                            <FileCopy />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete Mission">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDeleteMission(mission.id!)}
                            disabled={!user || !canDelete(user.role) || mission.status === 'active'}
                          >
                            <Delete />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </ListItemSecondaryAction>
                  </ListItem>
                  {index < missionPlans.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      {/* Create/Edit Mission Dialog */}
      <Dialog
        open={showCreateDialog}
        onClose={() => setShowCreateDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingMission ? 'Edit Mission Plan' : 'Create New Mission Plan'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Mission Name"
              value={formData.name || ''}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
              fullWidth
            />
            <TextField
              label="Description"
              value={formData.description || ''}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              multiline
              rows={3}
              required
              fullWidth
            />
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Priority</InputLabel>
                  <Select
                    value={formData.priority || 'medium'}
                    label="Priority"
                    onChange={(e) => setFormData({ ...formData, priority: e.target.value as any })}
                  >
                    <MenuItem value="low">Low</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                    <MenuItem value="critical">Critical</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>UAV</InputLabel>
                  <Select
                    value={formData.uav_id || ''}
                    label="UAV"
                    onChange={(e) => setFormData({ ...formData, uav_id: e.target.value as number })}
                  >
                    <MenuItem value="">None (assign later)</MenuItem>
                    {uavs.map((uav) => (
                      <MenuItem key={uav.id} value={uav.id}>
                        {uav.model} - {uav.serial_number}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Max Altitude (m)"
                  type="number"
                  value={formData.max_altitude || 120}
                  onChange={(e) => setFormData({ ...formData, max_altitude: parseInt(e.target.value) })}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Max Speed (m/s)"
                  type="number"
                  value={formData.max_speed || 15}
                  onChange={(e) => setFormData({ ...formData, max_speed: parseInt(e.target.value) })}
                  fullWidth
                />
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowCreateDialog(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSaveMission}
            variant="contained"
            disabled={createMissionMutation.isPending || updateMissionMutation.isPending}
          >
            {createMissionMutation.isPending || updateMissionMutation.isPending ? 'Saving...' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Map Planning Dialog */}
      <Dialog
        open={showMapDialog}
        onClose={() => setShowMapDialog(false)}
        maxWidth="xl"
        fullWidth
        PaperProps={{ sx: { height: '90vh' } }}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            Mission Planning Map
            <IconButton onClick={() => setShowMapDialog(false)}>
              <Delete />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body1" color="text.secondary" align="center" sx={{ py: 4 }}>
            Advanced mapping features have been simplified. 
            Use the basic waypoint form above to plan missions.
          </Typography>
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default MissionPlanningPage;