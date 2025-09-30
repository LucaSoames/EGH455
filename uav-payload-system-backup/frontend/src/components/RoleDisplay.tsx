import React from 'react';
import {
  Chip,
  Tooltip,
  Box,
  Typography,
  Avatar,
} from '@mui/material';
import {
  AdminPanelSettings,
  Engineering,
  Visibility,
} from '@mui/icons-material';
import { UserRole, getRoleDisplayName, getRoleColor, getRoleDescription } from '../utils/permissions';
import { useAuth } from '../contexts/AuthContext';

interface RoleChipProps {
  role: UserRole;
  size?: 'small' | 'medium';
  variant?: 'filled' | 'outlined';
}

export const RoleChip: React.FC<RoleChipProps> = ({
  role,
  size = 'small',
  variant = 'filled',
}) => {
  const getRoleIcon = (role: UserRole) => {
    switch (role) {
      case 'admin':
        return <AdminPanelSettings fontSize="small" />;
      case 'operator':
        return <Engineering fontSize="small" />;
      case 'viewer':
        return <Visibility fontSize="small" />;
      default:
        return undefined;
    }
  };

  return (
    <Tooltip title={getRoleDescription(role)}>
      <Chip
        icon={getRoleIcon(role)}
        label={getRoleDisplayName(role)}
        color={getRoleColor(role)}
        size={size}
        variant={variant}
      />
    </Tooltip>
  );
};

interface UserRoleDisplayProps {
  role: UserRole;
  username?: string;
  showDescription?: boolean;
  compact?: boolean;
}

export const UserRoleDisplay: React.FC<UserRoleDisplayProps> = ({
  role,
  username,
  showDescription = false,
  compact = false,
}) => {
  const getRoleIcon = (role: UserRole) => {
    const iconProps = { fontSize: compact ? 'small' : 'medium' } as const;
    switch (role) {
      case 'admin':
        return <AdminPanelSettings color="error" {...iconProps} />;
      case 'operator':
        return <Engineering color="warning" {...iconProps} />;
      case 'viewer':
        return <Visibility color="info" {...iconProps} />;
      default:
        return null;
    }
  };

  if (compact) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        {getRoleIcon(role)}
        <Typography variant="body2" color="textSecondary">
          {username ? `${username} (${getRoleDisplayName(role)})` : getRoleDisplayName(role)}
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
      <Avatar sx={{ bgcolor: `${getRoleColor(role)}.main`, width: 32, height: 32 }}>
        {getRoleIcon(role)}
      </Avatar>
      <Box>
        <Typography variant="subtitle2">
          {username || getRoleDisplayName(role)}
        </Typography>
        {username && (
          <RoleChip role={role} size="small" />
        )}
        {showDescription && (
          <Typography variant="caption" color="textSecondary" display="block">
            {getRoleDescription(role)}
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export const CurrentUserRoleDisplay: React.FC = () => {
  const { user } = useAuth();

  if (!user) {
    return null;
  }

  return (
    <UserRoleDisplay
      role={user.role}
      username={user.username}
      showDescription
      compact
    />
  );
};