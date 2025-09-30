import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Divider,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Chip,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Notifications,
  Security,
  Storage,
  Wifi,
  Save,
  RestartAlt,
} from '@mui/icons-material';

interface SystemSettings {
  notifications: {
    emailAlerts: boolean;
    pushNotifications: boolean;
    systemAlerts: boolean;
    missionAlerts: boolean;
  };
  system: {
    autoBackup: boolean;
    dataRetention: number; // days
    logLevel: 'debug' | 'info' | 'warning' | 'error';
    maxConnections: number;
  };
  security: {
    sessionTimeout: number; // minutes
    requireTwoFactor: boolean;
    passwordExpiry: number; // days
    allowRemoteAccess: boolean;
  };
  network: {
    apiPort: number;
    websocketPort: number;
    maxRetries: number;
    timeoutSeconds: number;
  };
}

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<SystemSettings>({
    notifications: {
      emailAlerts: true,
      pushNotifications: true,
      systemAlerts: true,
      missionAlerts: true,
    },
    system: {
      autoBackup: true,
      dataRetention: 90,
      logLevel: 'info',
      maxConnections: 100,
    },
    security: {
      sessionTimeout: 30,
      requireTwoFactor: false,
      passwordExpiry: 90,
      allowRemoteAccess: true,
    },
    network: {
      apiPort: 5000,
      websocketPort: 5001,
      maxRetries: 3,
      timeoutSeconds: 30,
    },
  });

  const [isDirty, setIsDirty] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleSettingChange = (section: keyof SystemSettings, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value,
      },
    }));
    setIsDirty(true);
  };

  const handleSave = async () => {
    try {
      // In a real app, this would save to the backend
      console.log('Saving settings:', settings);
      setMessage({ type: 'success', text: 'Settings saved successfully!' });
      setIsDirty(false);
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to save settings. Please try again.' });
    }
  };

  const handleReset = () => {
    // Reset to default values
    setSettings({
      notifications: {
        emailAlerts: true,
        pushNotifications: true,
        systemAlerts: true,
        missionAlerts: true,
      },
      system: {
        autoBackup: true,
        dataRetention: 90,
        logLevel: 'info',
        maxConnections: 100,
      },
      security: {
        sessionTimeout: 30,
        requireTwoFactor: false,
        passwordExpiry: 90,
        allowRemoteAccess: true,
      },
      network: {
        apiPort: 5000,
        websocketPort: 5001,
        maxRetries: 3,
        timeoutSeconds: 30,
      },
    });
    setIsDirty(true);
  };

  return (
    <Box p={3}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          System Settings
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<RestartAlt />}
            onClick={handleReset}
            sx={{ mr: 2 }}
          >
            Reset to Defaults
          </Button>
          <Button
            variant="contained"
            startIcon={<Save />}
            onClick={handleSave}
            disabled={!isDirty}
          >
            Save Changes
          </Button>
        </Box>
      </Box>

      {message && (
        <Alert
          severity={message.type}
          sx={{ mb: 3 }}
          onClose={() => setMessage(null)}
        >
          {message.text}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Notifications Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Notifications sx={{ mr: 1 }} />
                <Typography variant="h6">Notifications</Typography>
              </Box>
              <List>
                <ListItem>
                  <ListItemText
                    primary="Email Alerts"
                    secondary="Receive important alerts via email"
                  />
                  <ListItemSecondaryAction>
                    <Switch
                      checked={settings.notifications.emailAlerts}
                      onChange={(e) =>
                        handleSettingChange('notifications', 'emailAlerts', e.target.checked)
                      }
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Push Notifications"
                    secondary="Browser push notifications"
                  />
                  <ListItemSecondaryAction>
                    <Switch
                      checked={settings.notifications.pushNotifications}
                      onChange={(e) =>
                        handleSettingChange('notifications', 'pushNotifications', e.target.checked)
                      }
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="System Alerts"
                    secondary="Hardware and software alerts"
                  />
                  <ListItemSecondaryAction>
                    <Switch
                      checked={settings.notifications.systemAlerts}
                      onChange={(e) =>
                        handleSettingChange('notifications', 'systemAlerts', e.target.checked)
                      }
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Mission Alerts"
                    secondary="Mission status updates"
                  />
                  <ListItemSecondaryAction>
                    <Switch
                      checked={settings.notifications.missionAlerts}
                      onChange={(e) =>
                        handleSettingChange('notifications', 'missionAlerts', e.target.checked)
                      }
                    />
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Security Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Security sx={{ mr: 1 }} />
                <Typography variant="h6">Security</Typography>
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Session Timeout (minutes)"
                    type="number"
                    value={settings.security.sessionTimeout}
                    onChange={(e) =>
                      handleSettingChange('security', 'sessionTimeout', parseInt(e.target.value))
                    }
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Password Expiry (days)"
                    type="number"
                    value={settings.security.passwordExpiry}
                    onChange={(e) =>
                      handleSettingChange('security', 'passwordExpiry', parseInt(e.target.value))
                    }
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.security.requireTwoFactor}
                        onChange={(e) =>
                          handleSettingChange('security', 'requireTwoFactor', e.target.checked)
                        }
                      />
                    }
                    label="Require Two-Factor Authentication"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.security.allowRemoteAccess}
                        onChange={(e) =>
                          handleSettingChange('security', 'allowRemoteAccess', e.target.checked)
                        }
                      />
                    }
                    label="Allow Remote Access"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* System Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Storage sx={{ mr: 1 }} />
                <Typography variant="h6">System</Typography>
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Data Retention (days)"
                    type="number"
                    value={settings.system.dataRetention}
                    onChange={(e) =>
                      handleSettingChange('system', 'dataRetention', parseInt(e.target.value))
                    }
                    helperText="How long to keep telemetry data"
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Maximum Connections"
                    type="number"
                    value={settings.system.maxConnections}
                    onChange={(e) =>
                      handleSettingChange('system', 'maxConnections', parseInt(e.target.value))
                    }
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.system.autoBackup}
                        onChange={(e) =>
                          handleSettingChange('system', 'autoBackup', e.target.checked)
                        }
                      />
                    }
                    label="Automatic Backup"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Network Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Wifi sx={{ mr: 1 }} />
                <Typography variant="h6">Network</Typography>
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="API Port"
                    type="number"
                    value={settings.network.apiPort}
                    onChange={(e) =>
                      handleSettingChange('network', 'apiPort', parseInt(e.target.value))
                    }
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="WebSocket Port"
                    type="number"
                    value={settings.network.websocketPort}
                    onChange={(e) =>
                      handleSettingChange('network', 'websocketPort', parseInt(e.target.value))
                    }
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Max Retries"
                    type="number"
                    value={settings.network.maxRetries}
                    onChange={(e) =>
                      handleSettingChange('network', 'maxRetries', parseInt(e.target.value))
                    }
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Timeout (seconds)"
                    type="number"
                    value={settings.network.timeoutSeconds}
                    onChange={(e) =>
                      handleSettingChange('network', 'timeoutSeconds', parseInt(e.target.value))
                    }
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* System Information */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="subtitle2" color="textSecondary">
                      Version
                    </Typography>
                    <Typography variant="h6">
                      v1.0.0
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="subtitle2" color="textSecondary">
                      Status
                    </Typography>
                    <Chip label="Running" color="success" />
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="subtitle2" color="textSecondary">
                      Uptime
                    </Typography>
                    <Typography variant="h6">
                      2d 14h 32m
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="subtitle2" color="textSecondary">
                      Active Users
                    </Typography>
                    <Typography variant="h6">
                      3
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SettingsPage;
