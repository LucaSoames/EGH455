import React, { useState } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { Box, AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemIcon, ListItemText, IconButton, Divider } from '@mui/material';
import { Menu as MenuIcon, Dashboard, Flight, Assignment, Inventory, Settings, ExitToApp, Analytics, VideoCall, Security, GpsFixed, Science, Storage, Build } from '@mui/icons-material';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { SocketProvider } from './contexts/SocketContext';
import { AuditProvider } from './contexts/AuditContext';
import ErrorBoundary from './components/ErrorBoundary';
import { canView, canCreate, isAdmin } from './utils/permissions';
import { RoleChip } from './components/RoleDisplay';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import UAVManagementPage from './pages/UAVManagementPage';
import MissionManagementPage from './pages/MissionManagementPage';
import MissionPlanningPage from './pages/MissionPlanningPage';
import PayloadManagementPage from './pages/PayloadManagementPage';
import TelemetryVisualizationPage from './pages/TelemetryVisualizationPage';
import VideoStreamingPage from './pages/VideoStreamingPage';
import TargetDetectionPage from './pages/TargetDetectionPage';
import EnvironmentalMonitoringPage from './pages/EnvironmentalMonitoringPage';
import AuditLogPage from './pages/AuditLogPage';
import DataLoggingPage from './pages/DataLoggingPage';
import HardwareControlPage from './pages/HardwareControlPage';
import SettingsPage from './pages/SettingsPage';
import './App.css';

const drawerWidth = 240;

interface MenuItem {
  text: string;
  icon: React.ReactNode;
  path: string;
  requireCreate?: boolean;
  adminOnly?: boolean;
}

const menuItems: MenuItem[] = [
  { text: 'Dashboard', icon: <Dashboard />, path: '/dashboard' },
  { text: 'UAV Management', icon: <Flight />, path: '/uavs' },
  { text: 'Mission Control', icon: <Assignment />, path: '/missions' },
  { text: 'Mission Planning', icon: <Assignment />, path: '/mission-planning', requireCreate: true },
  { text: 'Payload Management', icon: <Inventory />, path: '/payloads' },
  { text: 'Live Video Stream', icon: <VideoCall />, path: '/video-stream' },
  { text: 'Target Detection', icon: <GpsFixed />, path: '/target-detection' },
  { text: 'Environmental Monitor', icon: <Science />, path: '/environmental' },
  { text: 'Telemetry Visualization', icon: <Analytics />, path: '/telemetry' },
  { text: 'Data Logging', icon: <Storage />, path: '/data-logging' },
  { text: 'Hardware Control', icon: <Build />, path: '/hardware-control' },
  { text: 'Audit Log', icon: <Security />, path: '/audit-log', adminOnly: true },
  { text: 'Settings', icon: <Settings />, path: '/settings', adminOnly: true },
];

const AppContent: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleMenuClick = (path: string) => {
    navigate(path);
    setMobileOpen(false);
  };

  // Filter menu items based on user role
  const visibleMenuItems = menuItems.filter(item => {
    if (!user) return false;
    
    if (item.adminOnly && !isAdmin(user.role)) {
      return false;
    }
    
    if (item.requireCreate && !canCreate(user.role)) {
      return false;
    }
    
    // All users can view basic pages
    return canView(user.role);
  });

  if (!user && location.pathname !== '/login') {
    return <LoginPage />;
  }

  if (location.pathname === '/login' && user) {
    navigate('/dashboard');
    return null;
  }

  const drawer = (
    <div>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          UAV TAQ-25
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {visibleMenuItems.map((item) => (
          <ListItem 
            component="button"
            key={item.text}
            onClick={() => handleMenuClick(item.path)}
            sx={{
              backgroundColor: location.pathname === item.path ? 'action.selected' : 'transparent',
              cursor: 'pointer',
              '&:hover': {
                backgroundColor: 'action.hover',
              },
            }}
          >
            <ListItemIcon>{item.icon}</ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
      <Divider />
      <List>
        <ListItem 
          component="button"
          onClick={handleLogout}
          sx={{
            cursor: 'pointer',
            '&:hover': {
              backgroundColor: 'action.hover',
            },
          }}
        >
          <ListItemIcon><ExitToApp /></ListItemIcon>
          <ListItemText primary="Logout" />
        </ListItem>
      </List>
    </div>
  );

  if (location.pathname === '/login') {
    return <LoginPage />;
  }

  return (
    <ErrorBoundary>
      <AuditProvider>
        <SocketProvider>
          <Box sx={{ display: 'flex' }}>
            <AppBar
            position="fixed"
            sx={{
              width: { sm: `calc(100% - ${drawerWidth}px)` },
              ml: { sm: `${drawerWidth}px` },
            }}
          >
            <Toolbar>
              <IconButton
                color="inherit"
                aria-label="open drawer"
                edge="start"
                onClick={handleDrawerToggle}
                sx={{ mr: 2, display: { sm: 'none' } }}
              >
                <MenuIcon />
              </IconButton>
              <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                UAV Payload Tracking & Acquisition System
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography variant="body2" color="inherit">
                  Welcome, {user?.username}
                </Typography>
                {user && <RoleChip role={user.role} variant="outlined" />}
              </Box>
            </Toolbar>
          </AppBar>
          
          <Box
            component="nav"
            sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
          >
            <Drawer
              variant="temporary"
              open={mobileOpen}
              onClose={handleDrawerToggle}
              ModalProps={{
                keepMounted: true,
              }}
              sx={{
                display: { xs: 'block', sm: 'none' },
                '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
              }}
            >
              {drawer}
            </Drawer>
            <Drawer
              variant="permanent"
              sx={{
                display: { xs: 'none', sm: 'block' },
                '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
              }}
              open
            >
              {drawer}
            </Drawer>
          </Box>
          
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              p: 3,
              width: { sm: `calc(100% - ${drawerWidth}px)` },
              mt: 8,
            }}
          >
            <ErrorBoundary>
              <Routes>
                <Route path="/dashboard" element={<DashboardPage />} />
                <Route path="/uavs" element={<UAVManagementPage />} />
                <Route path="/missions" element={<MissionManagementPage />} />
                <Route path="/mission-planning" element={<MissionPlanningPage />} />
                <Route path="/payloads" element={<PayloadManagementPage />} />
                <Route path="/video-stream" element={<VideoStreamingPage />} />
                <Route path="/target-detection" element={<TargetDetectionPage />} />
                <Route path="/environmental" element={<EnvironmentalMonitoringPage />} />
                <Route path="/telemetry" element={<TelemetryVisualizationPage />} />
                <Route path="/data-logging" element={<DataLoggingPage />} />
                <Route path="/hardware-control" element={<HardwareControlPage />} />
                <Route path="/audit-log" element={<AuditLogPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="/" element={<DashboardPage />} />
              </Routes>
            </ErrorBoundary>
            </Box>
          </Box>
        </SocketProvider>
      </AuditProvider>
    </ErrorBoundary>
  );
};

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </ErrorBoundary>
  );
};

export default App;