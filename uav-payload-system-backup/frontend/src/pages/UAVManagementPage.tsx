import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
} from '@mui/material';
import { DataGrid, GridColDef, GridActionsCellItem } from '@mui/x-data-grid';
import { Add, Edit, Delete, Flight } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { canCreate, canEdit, canDelete } from '../utils/permissions';

interface UAV {
  id: number;
  serial_number: string;
  model: string;
  max_payload_weight: number;
  max_altitude: number;
  max_speed: number;
  battery_capacity: number;
  communication_range: number;
  status: 'active' | 'inactive' | 'maintenance';
  created_at: string;
  updated_at: string;
}

const UAVManagementPage: React.FC = () => {
  const [openDialog, setOpenDialog] = useState(false);
  const [editingUAV, setEditingUAV] = useState<UAV | null>(null);
  const { user } = useAuth();
  const [formData, setFormData] = useState({
    serial_number: '',
    model: '',
    max_payload_weight: 0,
    max_altitude: 0,
    max_speed: 0,
    battery_capacity: 0,
    communication_range: 0,
    status: 'inactive' as 'active' | 'inactive' | 'maintenance',
  });

  const queryClient = useQueryClient();

  const { data: uavs, isLoading } = useQuery({
    queryKey: ['uavs'],
    queryFn: async () => {
      const response = await axios.get('/api/uavs');
      return response.data.data as UAV[];
    },
    refetchInterval: 30000
  });

  const createUAVMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await axios.post('/api/uavs', data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['uavs'] });
      alert('UAV created successfully');
      handleCloseDialog();
    },
    onError: (error: any) => {
      alert(error.response?.data?.error || 'Failed to create UAV');
    },
  });

  const updateUAVMutation = useMutation({
    mutationFn: async ({ id, data }: { id: number; data: any }) => {
      const response = await axios.put(`/api/uavs/${id}`, data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['uavs'] });
      alert('UAV updated successfully');
      handleCloseDialog();
    },
    onError: (error: any) => {
      alert(error.response?.data?.error || 'Failed to update UAV');
    },
  });

  const deleteUAVMutation = useMutation({
    mutationFn: async (id: number) => {
      const response = await axios.delete(`/api/uavs/${id}`);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['uavs'] });
      alert('UAV deleted successfully');
    },
    onError: (error: any) => {
      alert(error.response?.data?.error || 'Failed to delete UAV');
    },
  });

  const handleOpenDialog = (uav?: UAV) => {
    if (uav) {
      setEditingUAV(uav);
      setFormData({
        serial_number: uav.serial_number,
        model: uav.model,
        max_payload_weight: uav.max_payload_weight,
        max_altitude: uav.max_altitude,
        max_speed: uav.max_speed,
        battery_capacity: uav.battery_capacity,
        communication_range: uav.communication_range,
        status: uav.status,
      });
    } else {
      setEditingUAV(null);
      setFormData({
        serial_number: '',
        model: '',
        max_payload_weight: 0,
        max_altitude: 0,
        max_speed: 0,
        battery_capacity: 0,
        communication_range: 0,
        status: 'inactive',
      });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingUAV(null);
  };

  const handleSubmit = () => {
    if (editingUAV) {
      updateUAVMutation.mutate({ id: editingUAV.id, data: formData });
    } else {
      createUAVMutation.mutate(formData);
    }
  };

  const handleDelete = (id: number) => {
    if (window.confirm('Are you sure you want to delete this UAV?')) {
      deleteUAVMutation.mutate(id);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'default';
      case 'maintenance':
        return 'warning';
      default:
        return 'default';
    }
  };

  const columns: GridColDef[] = [
    { field: 'serial_number', headerName: 'Serial Number', width: 150 },
    { field: 'model', headerName: 'Model', width: 200 },
    {
      field: 'status',
      headerName: 'Status',
      width: 120,
      renderCell: (params) => (
        <Chip 
          label={params.value} 
          color={getStatusColor(params.value) as any}
          size="small"
        />
      ),
    },
    { field: 'max_payload_weight', headerName: 'Max Payload (kg)', width: 150, type: 'number' },
    { field: 'max_altitude', headerName: 'Max Altitude (m)', width: 150, type: 'number' },
    { field: 'max_speed', headerName: 'Max Speed (m/s)', width: 150, type: 'number' },
    { field: 'battery_capacity', headerName: 'Battery (mAh)', width: 150, type: 'number' },
    { field: 'communication_range', headerName: 'Range (m)', width: 130, type: 'number' },
    {
      field: 'actions',
      type: 'actions',
      headerName: 'Actions',
      width: 120,
      getActions: (params) => {
        const actions = [];
        
        if (user && canEdit(user.role)) {
          actions.push(
            <GridActionsCellItem
              key="edit"
              icon={<Edit />}
              label="Edit"
              onClick={() => handleOpenDialog(params.row)}
            />
          );
        }
        
        if (user && canDelete(user.role)) {
          actions.push(
            <GridActionsCellItem
              key="delete"
              icon={<Delete />}
              label="Delete"
              onClick={() => handleDelete(params.row.id)}
            />
          );
        }
        
        return actions;
      },
    },
  ];

  return (
    <Container maxWidth="lg">
      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" gutterBottom>
            UAV Fleet Management
          </Typography>
          {user && canCreate(user.role) && (
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => handleOpenDialog()}
            >
              Add UAV
            </Button>
          )}
        </Box>

      {/* Quick Stats */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Flight sx={{ fontSize: 40, color: 'success.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Active UAVs
                  </Typography>
                  <Typography variant="h5">
                    {uavs?.filter((uav: UAV) => uav.status === 'active').length || 0}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Flight sx={{ fontSize: 40, color: 'warning.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    In Maintenance
                  </Typography>
                  <Typography variant="h5">
                    {uavs?.filter((uav: UAV) => uav.status === 'maintenance').length || 0}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Flight sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Total Fleet
                  </Typography>
                  <Typography variant="h5">
                    {uavs?.length || 0}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* UAV Data Grid */}
      <Card>
        <CardContent>
          <DataGrid
            rows={uavs || []}
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

      {/* Add/Edit Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingUAV ? 'Edit UAV' : 'Add New UAV'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Serial Number"
                value={formData.serial_number}
                onChange={(e) => setFormData({ ...formData, serial_number: e.target.value })}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Model"
                value={formData.model}
                onChange={(e) => setFormData({ ...formData, model: e.target.value })}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Max Payload Weight (kg)"
                type="number"
                value={formData.max_payload_weight}
                onChange={(e) => setFormData({ ...formData, max_payload_weight: parseFloat(e.target.value) })}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Max Altitude (m)"
                type="number"
                value={formData.max_altitude}
                onChange={(e) => setFormData({ ...formData, max_altitude: parseFloat(e.target.value) })}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Max Speed (m/s)"
                type="number"
                value={formData.max_speed}
                onChange={(e) => setFormData({ ...formData, max_speed: parseFloat(e.target.value) })}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Battery Capacity (mAh)"
                type="number"
                value={formData.battery_capacity}
                onChange={(e) => setFormData({ ...formData, battery_capacity: parseFloat(e.target.value) })}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Communication Range (m)"
                type="number"
                value={formData.communication_range}
                onChange={(e) => setFormData({ ...formData, communication_range: parseFloat(e.target.value) })}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Status"
                value={formData.status}
                onChange={(e) => setFormData({ ...formData, status: e.target.value as any })}
                required
              >
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="inactive">Inactive</MenuItem>
                <MenuItem value="maintenance">Maintenance</MenuItem>
              </TextField>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button 
            onClick={handleSubmit} 
            variant="contained"
            disabled={createUAVMutation.isPending || updateUAVMutation.isPending}
          >
            {editingUAV ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      </Paper>
    </Container>
  );
};

export default UAVManagementPage;