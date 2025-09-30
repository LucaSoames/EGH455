import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Chip,
  IconButton,
  Alert,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { DataGrid, GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import {
  Security,
  FilterList,
  Clear,
  Download,
  Refresh,
  Search,
  Warning,
  Error as ErrorIcon,
  Info,
  CheckCircle,
} from '@mui/icons-material';
import { format } from 'date-fns';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { useAudit, AuditEvent, AuditEventType, AuditSeverity, AuditFilters } from '../contexts/AuditContext';

interface User {
  id: number;
  username: string;
  role: string;
}

const AuditLogPage: React.FC = () => {
  const { user: currentUser } = useAuth();
  const { events, isLoading, filters, setFilters, exportAuditLog, clearFilters, refreshEvents } = useAudit();
  const [exportLoading, setExportLoading] = useState(false);

  const { data: users } = useQuery({
    queryKey: ['users'],
    queryFn: async () => {
      const response = await axios.get('/api/users');
      return response.data.data as User[];
    },
    enabled: currentUser?.role === 'admin',
  });

  const eventTypeOptions: { value: AuditEventType; label: string }[] = [
    { value: 'user_login', label: 'User Login' },
    { value: 'user_logout', label: 'User Logout' },
    { value: 'uav_created', label: 'UAV Created' },
    { value: 'uav_updated', label: 'UAV Updated' },
    { value: 'uav_deleted', label: 'UAV Deleted' },
    { value: 'mission_created', label: 'Mission Created' },
    { value: 'mission_updated', label: 'Mission Updated' },
    { value: 'mission_deleted', label: 'Mission Deleted' },
    { value: 'mission_started', label: 'Mission Started' },
    { value: 'mission_completed', label: 'Mission Completed' },
    { value: 'payload_assigned', label: 'Payload Assigned' },
    { value: 'payload_unassigned', label: 'Payload Unassigned' },
    { value: 'telemetry_data_received', label: 'Telemetry Data Received' },
    { value: 'system_alert', label: 'System Alert' },
    { value: 'data_export', label: 'Data Export' },
    { value: 'settings_changed', label: 'Settings Changed' },
    { value: 'video_stream_started', label: 'Video Stream Started' },
    { value: 'video_stream_stopped', label: 'Video Stream Stopped' },
    { value: 'emergency_stop', label: 'Emergency Stop' },
    { value: 'target_detected', label: 'Target Detected' },
    { value: 'sensor_data_logged', label: 'Sensor Data Logged' },
  ];

  const severityOptions: { value: AuditSeverity; label: string; color: string }[] = [
    { value: 'low', label: 'Low', color: '#4caf50' },
    { value: 'medium', label: 'Medium', color: '#ff9800' },
    { value: 'high', label: 'High', color: '#f44336' },
    { value: 'critical', label: 'Critical', color: '#d32f2f' },
  ];

  const handleFilterChange = (field: keyof AuditFilters, value: any) => {
    setFilters({
      ...filters,
      [field]: value,
    });
  };

  const handleExport = async () => {
    setExportLoading(true);
    try {
      await exportAuditLog();
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExportLoading(false);
    }
  };

  const getSeverityIcon = (severity: AuditSeverity) => {
    switch (severity) {
      case 'low':
        return <CheckCircle sx={{ color: '#4caf50', fontSize: 20 }} />;
      case 'medium':
        return <Info sx={{ color: '#2196f3', fontSize: 20 }} />;
      case 'high':
        return <Warning sx={{ color: '#ff9800', fontSize: 20 }} />;
      case 'critical':
        return <ErrorIcon sx={{ color: '#f44336', fontSize: 20 }} />;
      default:
        return <Info sx={{ color: '#2196f3', fontSize: 20 }} />;
    }
  };

  const getSeverityColor = (severity: AuditSeverity): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (severity) {
      case 'low': return 'success';
      case 'medium': return 'info';
      case 'high': return 'warning';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  const columns: GridColDef[] = [
    {
      field: 'timestamp',
      headerName: 'Timestamp',
      width: 180,
      renderCell: (params: GridRenderCellParams) => (
        <Tooltip title={format(new Date(params.value), 'PPpp')}>
          <span>{format(new Date(params.value), 'MMM dd, HH:mm:ss')}</span>
        </Tooltip>
      ),
    },
    {
      field: 'severity',
      headerName: 'Severity',
      width: 120,
      renderCell: (params: GridRenderCellParams) => (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {getSeverityIcon(params.value)}
          <Chip
            label={params.value.toUpperCase()}
            color={getSeverityColor(params.value)}
            size="small"
          />
        </Box>
      ),
    },
    {
      field: 'event_type',
      headerName: 'Event Type',
      width: 200,
      renderCell: (params: GridRenderCellParams) => {
        const eventTypeOption = eventTypeOptions.find(option => option.value === params.value);
        return <span>{eventTypeOption?.label || params.value}</span>;
      },
    },
    {
      field: 'username',
      headerName: 'User',
      width: 150,
      renderCell: (params: GridRenderCellParams) => (
        <span>{params.value || 'System'}</span>
      ),
    },
    {
      field: 'description',
      headerName: 'Description',
      flex: 1,
      minWidth: 300,
    },
    {
      field: 'ip_address',
      headerName: 'IP Address',
      width: 130,
      renderCell: (params: GridRenderCellParams) => (
        <span>{params.value || 'N/A'}</span>
      ),
    },
    {
      field: 'details',
      headerName: 'Details',
      width: 100,
      renderCell: (params: GridRenderCellParams) => (
        <Tooltip title={JSON.stringify(params.value, null, 2)}>
          <IconButton size="small">
            <Info fontSize="small" />
          </IconButton>
        </Tooltip>
      ),
    },
  ];

  const stats = {
    total: events.length,
    critical: events.filter(e => e.severity === 'critical').length,
    high: events.filter(e => e.severity === 'high').length,
    today: events.filter(e => {
      const today = new Date();
      const eventDate = new Date(e.timestamp);
      return eventDate.toDateString() === today.toDateString();
    }).length,
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Container maxWidth="xl">
        <Paper sx={{ p: 3 }}>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h4" gutterBottom>
              <Security sx={{ mr: 1, verticalAlign: 'middle' }} />
              Audit Log Management
            </Typography>
            <Typography variant="body1" color="textSecondary">
              Comprehensive system activity logging and security audit trail
            </Typography>
          </Box>

          {/* Statistics Cards */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="primary">
                    {stats.total}
                  </Typography>
                  <Typography color="textSecondary">
                    Total Events
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="error">
                    {stats.critical}
                  </Typography>
                  <Typography color="textSecondary">
                    Critical Events
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" sx={{ color: '#ff9800' }}>
                    {stats.high}
                  </Typography>
                  <Typography color="textSecondary">
                    High Priority
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="secondary">
                    {stats.today}
                  </Typography>
                  <Typography color="textSecondary">
                    Today's Events
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Filters */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              <FilterList sx={{ mr: 1, verticalAlign: 'middle' }} />
              Filters
            </Typography>
            
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6} md={2}>
                <FormControl fullWidth size="small">
                  <InputLabel>Event Type</InputLabel>
                  <Select
                    value={filters.event_type || ''}
                    label="Event Type"
                    onChange={(e) => handleFilterChange('event_type', e.target.value || '')}
                  >
                    <MenuItem value="">All Types</MenuItem>
                    {eventTypeOptions.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <FormControl fullWidth size="small">
                  <InputLabel>Severity</InputLabel>
                  <Select
                    value={filters.severity || ''}
                    label="Severity"
                    onChange={(e) => handleFilterChange('severity', e.target.value || '')}
                  >
                    <MenuItem value="">All Severities</MenuItem>
                    {severityOptions.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getSeverityIcon(option.value)}
                          {option.label}
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              {currentUser?.role === 'admin' && (
                <Grid item xs={12} sm={6} md={2}>
                  <FormControl fullWidth size="small">
                    <InputLabel>User</InputLabel>
                    <Select
                      value={filters.user_id || ''}
                      label="User"
                      onChange={(e) => handleFilterChange('user_id', e.target.value || '')}
                    >
                      <MenuItem value="">All Users</MenuItem>
                      {users?.map((user) => (
                        <MenuItem key={user.id} value={user.id}>
                          {user.username} ({user.role})
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              )}

              <Grid item xs={12} sm={6} md={2}>
                <DatePicker
                  label="From Date"
                  value={filters.date_from}
                  onChange={(date) => handleFilterChange('date_from', date)}
                  slotProps={{ 
                    textField: { 
                      size: 'small',
                      fullWidth: true
                    } 
                  }}
                />
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <DatePicker
                  label="To Date"
                  value={filters.date_to}
                  onChange={(date) => handleFilterChange('date_to', date)}
                  slotProps={{ 
                    textField: { 
                      size: 'small',
                      fullWidth: true
                    } 
                  }}
                />
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <TextField
                  fullWidth
                  size="small"
                  label="Search"
                  value={filters.search || ''}
                  onChange={(e) => handleFilterChange('search', e.target.value)}
                  InputProps={{
                    startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
                  }}
                />
              </Grid>
            </Grid>

            <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                startIcon={<Clear />}
                onClick={clearFilters}
                size="small"
              >
                Clear Filters
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={refreshEvents}
                size="small"
              >
                Refresh
              </Button>
              
              <Button
                variant="contained"
                startIcon={<Download />}
                onClick={handleExport}
                disabled={exportLoading}
                size="small"
              >
                Export CSV
              </Button>
            </Box>
          </Paper>

          {/* Loading Indicator */}
          {isLoading && <LinearProgress sx={{ mb: 2 }} />}

          {/* Alert for Security Events */}
          {stats.critical > 0 && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {stats.critical} critical security events detected. Immediate attention required.
            </Alert>
          )}

          {stats.high > 0 && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              {stats.high} high-priority events require review.
            </Alert>
          )}

          {/* Data Grid */}
          <Paper>
            <DataGrid
              rows={events}
              columns={columns}
              loading={isLoading}
              autoHeight
              checkboxSelection
              disableRowSelectionOnClick
              initialState={{
                pagination: { paginationModel: { pageSize: 25 } },
                sorting: {
                  sortModel: [{ field: 'timestamp', sort: 'desc' }],
                },
              }}
              pageSizeOptions={[10, 25, 50, 100]}
              sx={{
                '& .MuiDataGrid-row': {
                  '&:hover': {
                    backgroundColor: 'rgba(0, 0, 0, 0.04)',
                  },
                },
              }}
            />
          </Paper>
        </Paper>
      </Container>
    </LocalizationProvider>
  );
};

export default AuditLogPage;