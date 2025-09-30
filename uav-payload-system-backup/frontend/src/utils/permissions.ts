// Simplified role-based access control
export type UserRole = 'admin' | 'operator' | 'viewer';

// Simple role-based permissions - no granular permissions needed
export const canAccess = (userRole: UserRole, action: 'view' | 'create' | 'edit' | 'delete'): boolean => {
  switch (userRole) {
    case 'admin':
      return true; // Admins can do everything
    case 'operator':
      return action === 'view' || action === 'create' || action === 'edit'; // Can't delete
    case 'viewer':
      return action === 'view'; // Can only view
    default:
      return false;
  }
};

// Simple permission checks
export const canView = (userRole: UserRole): boolean => canAccess(userRole, 'view');
export const canCreate = (userRole: UserRole): boolean => canAccess(userRole, 'create');
export const canEdit = (userRole: UserRole): boolean => canAccess(userRole, 'edit');
export const canDelete = (userRole: UserRole): boolean => canAccess(userRole, 'delete');

// Admin-only actions
export const isAdmin = (userRole: UserRole): boolean => userRole === 'admin';

// Role display utilities
export const getRoleDisplayName = (role: UserRole): string => {
  switch (role) {
    case 'admin':
      return 'Administrator';
    case 'operator':
      return 'Operator';
    case 'viewer':
      return 'Viewer';
    default:
      return role;
  }
};

export const getRoleColor = (role: UserRole): 'error' | 'warning' | 'info' | 'success' => {
  switch (role) {
    case 'admin':
      return 'error';
    case 'operator':
      return 'warning';
    case 'viewer':
      return 'info';
    default:
      return 'info';
  }
};

export const getRoleDescription = (role: UserRole): string => {
  switch (role) {
    case 'admin':
      return 'Full system access with all privileges';
    case 'operator':
      return 'Can create, edit and operate UAVs, missions, and payloads';
    case 'viewer':
      return 'Read-only access to view system status and data';
    default:
      return 'Unknown role';
  }
};