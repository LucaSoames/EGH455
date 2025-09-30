import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
} from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import { Add, Edit, Delete, PlayArrow, LocationOn } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';

interface Mission {
  id: number;
  name: string;
  description: string;
  status: 'planned' | 'active' | 'completed' | 'cancelled';
  created_at: string;
  start_time?: string;
  end_time?: string;
  uav_id?: number;
  uav?: {
    id: number;
    model: string;
    serial_number: string;
  };
  waypoints: Array<{
    latitude: number;
    longitude: number;
    altitude: number;
    order: number;
  }>;
}

interface MissionFormData {
  name: string;
  description: string;
  uav_id: number;
  waypoints: Array<{
    latitude: number;
    longitude: number;
    altitude: number;
    order: number;
  }>;
}

const MissionManagementPage: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [editingMission, setEditingMission] = useState<Mission | null>(null);
  const [formData, setFormData] = useState<MissionFormData>({
    name: '',
    description: '',
    uav_id: 0,
    waypoints: [],
  });
  const [error, setError] = useState<string | null>(null);

  const queryClient = useQueryClient();

  // Fetch missions
  const { data: missions = [], isLoading } = useQuery({
    queryKey: ['missions'],
    queryFn: async () => {
      const response = await fetch('http://localhost:5000/api/missions', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (!response.ok) throw new Error('Failed to fetch missions');
      const result = await response.json();
      return result.data || [];
    },
  });

  // Fetch UAVs for assignment
  const { data: uavs = [] } = useQuery({
    queryKey: ['uavs'],
    queryFn: async () => {
      const response = await fetch('http://localhost:5000/api/uavs', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (!response.ok) throw new Error('Failed to fetch UAVs');
      const result = await response.json();
      return result.data || [];
    },
  });

  // Create mission mutation
  const createMissionMutation = useMutation({
    mutationFn: async (data: MissionFormData) => {
      const response = await fetch('http://localhost:5000/api/missions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Failed to create mission');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['missions'] });
      handleCloseDialog();
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  // Update mission mutation
  const updateMissionMutation = useMutation({
    mutationFn: async (data: { id: number } & Partial<MissionFormData>) => {
      const { id, ...updateData } = data;
      const response = await fetch(`http://localhost:5000/api/missions/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(updateData),
      });
      if (!response.ok) throw new Error('Failed to update mission');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['missions'] });
      handleCloseDialog();
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  // Delete mission mutation
  const deleteMissionMutation = useMutation({
    mutationFn: async (id: number) => {
      const response = await fetch(`http://localhost:5000/api/missions/${id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (!response.ok) throw new Error('Failed to delete mission');
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['missions'] });
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  // Start mission mutation
  const startMissionMutation = useMutation({
    mutationFn: async (id: number) => {
      const response = await fetch(`http://localhost:5000/api/missions/${id}/start`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (!response.ok) throw new Error('Failed to start mission');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['missions'] });
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  const columns: GridColDef[] = [
    { field: 'id', headerName: 'ID', width: 70 },
    { field: 'name', headerName: 'Mission Name', width: 200 },
    { field: 'description', headerName: 'Description', width: 300 },
    {
      field: 'status',
      headerName: 'Status',
      width: 130,
      renderCell: (params) => {
        const getStatusColor = (status: string) => {
          switch (status) {
            case 'planned': return 'default';
            case 'active': return 'primary';
            case 'completed': return 'success';
            case 'cancelled': return 'error';
            default: return 'default';
          }
        };
        return (
          <Chip
            label={params.value}
            color={getStatusColor(params.value) as any}
            size="small"
          />
        );
      },
    },
    {
      field: 'uav',
      headerName: 'Assigned UAV',
      width: 150,
      valueGetter: (params) => params.row.uav?.model || 'Unassigned',
    },
    {
      field: 'created_at',
      headerName: 'Created',
      width: 180,
      valueFormatter: (params) => format(new Date(params.value), 'MMM dd, yyyy HH:mm'),
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 200,
      sortable: false,
      renderCell: (params) => (
        <Box>
          <Button
            size="small"
            startIcon={<Edit />}
            onClick={() => handleEdit(params.row)}
            sx={{ mr: 1 }}
          >
            Edit
          </Button>
          {params.row.status === 'planned' && (
            <Button
              size="small"
              startIcon={<PlayArrow />}
              onClick={() => startMissionMutation.mutate(params.row.id)}
              color="success"
              sx={{ mr: 1 }}
            >
              Start
            </Button>
          )}
          <Button
            size="small"
            startIcon={<Delete />}
            onClick={() => deleteMissionMutation.mutate(params.row.id)}
            color="error"
          >
            Delete
          </Button>
        </Box>
      ),
    },
  ];

  const handleAdd = () => {
    setEditingMission(null);
    setFormData({
      name: '',
      description: '',
      uav_id: 0,
      waypoints: [],
    });
    setOpen(true);
  };

  const handleEdit = (mission: Mission) => {
    setEditingMission(mission);
    setFormData({
      name: mission.name,
      description: mission.description,
      uav_id: mission.uav_id || 0,
      waypoints: mission.waypoints,
    });
    setOpen(true);
  };

  const handleCloseDialog = () => {
    setOpen(false);
    setEditingMission(null);
    setError(null);
  };

  const handleSubmit = () => {
    if (editingMission) {
      updateMissionMutation.mutate({ id: editingMission.id, ...formData });
    } else {
      createMissionMutation.mutate(formData);
    }
  };

  const addWaypoint = () => {
    setFormData({
      ...formData,
      waypoints: [
        ...formData.waypoints,
        {
          latitude: 0,
          longitude: 0,
          altitude: 100,
          order: formData.waypoints.length + 1,
        },
      ],
    });
  };

  const updateWaypoint = (index: number, field: string, value: number) => {
    const updatedWaypoints = [...formData.waypoints];
    updatedWaypoints[index] = { ...updatedWaypoints[index], [field]: value };
    setFormData({ ...formData, waypoints: updatedWaypoints });
  };

  const removeWaypoint = (index: number) => {
    const updatedWaypoints = formData.waypoints.filter((_, i) => i !== index);
    setFormData({ ...formData, waypoints: updatedWaypoints });
  };

  return (
    <Box p={3}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Mission Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={handleAdd}
        >
          New Mission
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Mission Statistics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Missions
              </Typography>
              <Typography variant="h5">
                {missions.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Missions
              </Typography>
              <Typography variant="h5" color="primary">
                {missions.filter((m: Mission) => m.status === 'active').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Planned Missions
              </Typography>
              <Typography variant="h5" color="warning.main">
                {missions.filter((m: Mission) => m.status === 'planned').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Completed Missions
              </Typography>
              <Typography variant="h5" color="success.main">
                {missions.filter((m: Mission) => m.status === 'completed').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Missions Data Grid */}
      <Card>
        <CardContent>
          <DataGrid
            rows={missions}
            columns={columns}
            loading={isLoading}
            autoHeight
            checkboxSelection
            disableRowSelectionOnClick
            initialState={{
              pagination: { paginationModel: { pageSize: 10 } },
            }}
            pageSizeOptions={[5, 10, 25]}
          />
        </CardContent>
      </Card>

      {/* Add/Edit Mission Dialog */}
      <Dialog open={open} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingMission ? 'Edit Mission' : 'Add New Mission'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Mission Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Assign UAV</InputLabel>
                <Select
                  value={formData.uav_id}
                  onChange={(e) => setFormData({ ...formData, uav_id: Number(e.target.value) })}
                  label="Assign UAV"
                >
                  <MenuItem value={0}>No UAV Assigned</MenuItem>
                  {uavs.map((uav: any) => (
                    <MenuItem key={uav.id} value={uav.id}>
                      {uav.model} - {uav.serial_number}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            {/* Waypoints Section */}
            <Grid item xs={12}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">Waypoints</Typography>
                <Button startIcon={<LocationOn />} onClick={addWaypoint}>
                  Add Waypoint
                </Button>
              </Box>
              {formData.waypoints.map((waypoint, index) => (
                <Box key={index} mb={2} p={2} border={1} borderRadius={1} borderColor="grey.300">
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={3}>
                      <TextField
                        fullWidth
                        label="Latitude"
                        type="number"
                        value={waypoint.latitude}
                        onChange={(e) => updateWaypoint(index, 'latitude', parseFloat(e.target.value))}
                      />
                    </Grid>
                    <Grid item xs={3}>
                      <TextField
                        fullWidth
                        label="Longitude"
                        type="number"
                        value={waypoint.longitude}
                        onChange={(e) => updateWaypoint(index, 'longitude', parseFloat(e.target.value))}
                      />
                    </Grid>
                    <Grid item xs={3}>
                      <TextField
                        fullWidth
                        label="Altitude (m)"
                        type="number"
                        value={waypoint.altitude}
                        onChange={(e) => updateWaypoint(index, 'altitude', parseFloat(e.target.value))}
                      />
                    </Grid>
                    <Grid item xs={2}>
                      <Typography variant="body2" color="textSecondary">
                        Order: {waypoint.order}
                      </Typography>
                    </Grid>
                    <Grid item xs={1}>
                      <Button
                        size="small"
                        color="error"
                        onClick={() => removeWaypoint(index)}
                      >
                        Remove
                      </Button>
                    </Grid>
                  </Grid>
                </Box>
              ))}
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button
            onClick={handleSubmit}
            variant="contained"
            disabled={createMissionMutation.isPending || updateMissionMutation.isPending}
          >
            {editingMission ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MissionManagementPage;
