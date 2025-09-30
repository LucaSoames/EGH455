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
  IconButton,
  Tooltip,
} from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import { Add, Edit, Delete, Info } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';

interface Payload {
  id: number;
  name: string;
  type: string;
  weight: number;
  dimensions: string;
  status: 'available' | 'deployed' | 'maintenance';
  description?: string;
  created_at: string;
  mission_id?: number;
  mission?: {
    id: number;
    name: string;
  };
}

interface PayloadFormData {
  name: string;
  type: string;
  weight: number;
  dimensions: string;
  description: string;
}

const PayloadManagementPage: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [editingPayload, setEditingPayload] = useState<Payload | null>(null);
  const [formData, setFormData] = useState<PayloadFormData>({
    name: '',
    type: '',
    weight: 0,
    dimensions: '',
    description: '',
  });
  const [error, setError] = useState<string | null>(null);

  const queryClient = useQueryClient();

  // Fetch payloads
  const { data: payloads = [], isLoading } = useQuery({
    queryKey: ['payloads'],
    queryFn: async () => {
      const response = await fetch('http://localhost:5000/api/payloads', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (!response.ok) throw new Error('Failed to fetch payloads');
      const result = await response.json();
      return result.data || [];
    },
  });

  // Create payload mutation
  const createPayloadMutation = useMutation({
    mutationFn: async (data: PayloadFormData) => {
      const response = await fetch('http://localhost:5000/api/payloads', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Failed to create payload');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['payloads'] });
      handleCloseDialog();
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  // Update payload mutation
  const updatePayloadMutation = useMutation({
    mutationFn: async (data: { id: number } & Partial<PayloadFormData>) => {
      const { id, ...updateData } = data;
      const response = await fetch(`http://localhost:5000/api/payloads/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(updateData),
      });
      if (!response.ok) throw new Error('Failed to update payload');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['payloads'] });
      handleCloseDialog();
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  // Delete payload mutation
  const deletePayloadMutation = useMutation({
    mutationFn: async (id: number) => {
      const response = await fetch(`http://localhost:5000/api/payloads/${id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (!response.ok) throw new Error('Failed to delete payload');
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['payloads'] });
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  const columns: GridColDef[] = [
    { field: 'id', headerName: 'ID', width: 70 },
    { field: 'name', headerName: 'Payload Name', width: 200 },
    { field: 'type', headerName: 'Type', width: 150 },
    { field: 'weight', headerName: 'Weight (kg)', width: 120, type: 'number' },
    { field: 'dimensions', headerName: 'Dimensions', width: 150 },
    {
      field: 'status',
      headerName: 'Status',
      width: 130,
      renderCell: (params) => {
        const getStatusColor = (status: string) => {
          switch (status) {
            case 'available': return 'success';
            case 'deployed': return 'primary';
            case 'maintenance': return 'warning';
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
      field: 'mission',
      headerName: 'Assigned Mission',
      width: 150,
      valueGetter: (params) => params.row.mission?.name || 'Not Assigned',
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
      width: 150,
      sortable: false,
      renderCell: (params) => (
        <Box>
          <Tooltip title="View Details">
            <IconButton size="small" onClick={() => handleEdit(params.row)}>
              <Info />
            </IconButton>
          </Tooltip>
          <Tooltip title="Edit">
            <IconButton size="small" onClick={() => handleEdit(params.row)}>
              <Edit />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton
              size="small"
              onClick={() => deletePayloadMutation.mutate(params.row.id)}
              color="error"
            >
              <Delete />
            </IconButton>
          </Tooltip>
        </Box>
      ),
    },
  ];

  const handleAdd = () => {
    setEditingPayload(null);
    setFormData({
      name: '',
      type: '',
      weight: 0,
      dimensions: '',
      description: '',
    });
    setOpen(true);
  };

  const handleEdit = (payload: Payload) => {
    setEditingPayload(payload);
    setFormData({
      name: payload.name,
      type: payload.type,
      weight: payload.weight,
      dimensions: payload.dimensions,
      description: payload.description || '',
    });
    setOpen(true);
  };

  const handleCloseDialog = () => {
    setOpen(false);
    setEditingPayload(null);
    setError(null);
  };

  const handleSubmit = () => {
    if (editingPayload) {
      updatePayloadMutation.mutate({ id: editingPayload.id, ...formData });
    } else {
      createPayloadMutation.mutate(formData);
    }
  };

  const payloadTypes = [
    'Camera',
    'Sensor',
    'Communication Equipment',
    'Research Instrument',
    'Cargo',
    'Other',
  ];

  return (
    <Box p={3}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Payload Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={handleAdd}
        >
          New Payload
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Payload Statistics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Payloads
              </Typography>
              <Typography variant="h5">
                {payloads.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Available
              </Typography>
              <Typography variant="h5" color="success.main">
                {payloads.filter((p: Payload) => p.status === 'available').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Deployed
              </Typography>
              <Typography variant="h5" color="primary">
                {payloads.filter((p: Payload) => p.status === 'deployed').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Maintenance
              </Typography>
              <Typography variant="h5" color="warning.main">
                {payloads.filter((p: Payload) => p.status === 'maintenance').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Payloads Data Grid */}
      <Card>
        <CardContent>
          <DataGrid
            rows={payloads}
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

      {/* Add/Edit Payload Dialog */}
      <Dialog open={open} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingPayload ? 'Edit Payload' : 'Add New Payload'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Payload Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Type</InputLabel>
                <Select
                  value={formData.type}
                  onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                  label="Type"
                >
                  {payloadTypes.map((type) => (
                    <MenuItem key={type} value={type}>
                      {type}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Weight (kg)"
                type="number"
                value={formData.weight}
                onChange={(e) => setFormData({ ...formData, weight: parseFloat(e.target.value) })}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Dimensions (LxWxH)"
                value={formData.dimensions}
                onChange={(e) => setFormData({ ...formData, dimensions: e.target.value })}
                placeholder="e.g., 10x5x3 cm"
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
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button
            onClick={handleSubmit}
            variant="contained"
            disabled={createPayloadMutation.isPending || updatePayloadMutation.isPending}
          >
            {editingPayload ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PayloadManagementPage;
