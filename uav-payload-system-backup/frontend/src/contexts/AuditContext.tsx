import React, { createContext, useContext, useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { useAuth } from './AuthContext';

export type AuditEventType = 
  | 'user_login'
  | 'user_logout' 
  | 'uav_created'
  | 'uav_updated'
  | 'uav_deleted'
  | 'mission_created'
  | 'mission_updated'
  | 'mission_deleted'
  | 'mission_started'
  | 'mission_completed'
  | 'payload_assigned'
  | 'payload_unassigned'
  | 'telemetry_data_received'
  | 'system_alert'
  | 'data_export'
  | 'settings_changed'
  | 'video_stream_started'
  | 'video_stream_stopped'
  | 'emergency_stop'
  | 'target_detected'
  | 'sensor_data_logged';

export type AuditSeverity = 'low' | 'medium' | 'high' | 'critical';

export interface AuditEvent {
  id: number;
  event_type: AuditEventType;
  severity: AuditSeverity;
  user_id?: number;
  username?: string;
  description: string;
  details: Record<string, any>;
  ip_address?: string;
  user_agent?: string;
  timestamp: string;
  created_at: string;
}

export interface AuditFilters {
  event_type?: AuditEventType | '';
  severity?: AuditSeverity | '';
  user_id?: number | '';
  date_from?: Date | null;
  date_to?: Date | null;
  search?: string;
}

interface AuditContextType {
  events: AuditEvent[];
  isLoading: boolean;
  filters: AuditFilters;
  setFilters: (filters: AuditFilters) => void;
  logEvent: (event: {
    event_type: AuditEventType;
    severity: AuditSeverity;
    description: string;
    details?: Record<string, any>;
  }) => void;
  exportAuditLog: (filters?: AuditFilters) => Promise<void>;
  clearFilters: () => void;
  refreshEvents: () => void;
}

const AuditContext = createContext<AuditContextType | undefined>(undefined);

export const useAudit = () => {
  const context = useContext(AuditContext);
  if (!context) {
    throw new Error('useAudit must be used within an AuditProvider');
  }
  return context;
};

interface AuditProviderProps {
  children: React.ReactNode;
}

export const AuditProvider: React.FC<AuditProviderProps> = ({ children }) => {
  const [filters, setFilters] = useState<AuditFilters>({});
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const { data: events = [], isLoading, refetch } = useQuery({
    queryKey: ['audit-events', filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      
      if (filters.event_type) params.append('event_type', filters.event_type);
      if (filters.severity) params.append('severity', filters.severity);
      if (filters.user_id) params.append('user_id', filters.user_id.toString());
      if (filters.date_from) params.append('date_from', filters.date_from.toISOString());
      if (filters.date_to) params.append('date_to', filters.date_to.toISOString());
      if (filters.search) params.append('search', filters.search);
      
      const response = await axios.get(`/api/audit/events?${params}`);
      return response.data.data as AuditEvent[];
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const logEventMutation = useMutation({
    mutationFn: async (eventData: {
      event_type: AuditEventType;
      severity: AuditSeverity;
      description: string;
      details?: Record<string, any>;
    }) => {
      const response = await axios.post('/api/audit/events', {
        ...eventData,
        details: eventData.details || {},
        ip_address: await getClientIP(),
        user_agent: navigator.userAgent,
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['audit-events'] });
    },
    onError: (error) => {
      console.error('Failed to log audit event:', error);
    },
  });

  const logEvent = (eventData: {
    event_type: AuditEventType;
    severity: AuditSeverity;
    description: string;
    details?: Record<string, any>;
  }) => {
    logEventMutation.mutate(eventData);
  };

  const exportAuditLog = async (exportFilters?: AuditFilters) => {
    try {
      const params = new URLSearchParams();
      const filtersToUse = exportFilters || filters;
      
      if (filtersToUse.event_type) params.append('event_type', filtersToUse.event_type);
      if (filtersToUse.severity) params.append('severity', filtersToUse.severity);
      if (filtersToUse.user_id) params.append('user_id', filtersToUse.user_id.toString());
      if (filtersToUse.date_from) params.append('date_from', filtersToUse.date_from.toISOString());
      if (filtersToUse.date_to) params.append('date_to', filtersToUse.date_to.toISOString());
      if (filtersToUse.search) params.append('search', filtersToUse.search);
      
      const response = await axios.get(`/api/audit/export?${params}`, {
        responseType: 'blob',
      });
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `audit_log_${new Date().toISOString().split('T')[0]}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      // Log the export action
      logEvent({
        event_type: 'data_export',
        severity: 'medium',
        description: 'Audit log exported',
        details: { filters: filtersToUse, exported_at: new Date().toISOString() },
      });
      
    } catch (error) {
      console.error('Failed to export audit log:', error);
      throw error;
    }
  };

  const clearFilters = () => {
    setFilters({});
  };

  const refreshEvents = () => {
    refetch();
  };

  // Helper function to get client IP (simplified)
  const getClientIP = async (): Promise<string> => {
    try {
      const response = await fetch('https://api.ipify.org?format=json');
      const data = await response.json();
      return data.ip;
    } catch {
      return 'unknown';
    }
  };

  // Auto-log user actions
  useEffect(() => {
    if (user) {
      // Log successful login (this would typically be done server-side)
      // Only log once per session
      const hasLoggedLogin = sessionStorage.getItem('audit_login_logged');
      if (!hasLoggedLogin) {
        logEvent({
          event_type: 'user_login',
          severity: 'low',
          description: `User ${user.username} logged in successfully`,
          details: { 
            role: user.role,
            login_timestamp: new Date().toISOString()
          },
        });
        sessionStorage.setItem('audit_login_logged', 'true');
      }
    }
  }, [user]);

  // Auto-log logout on window beforeunload
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (user) {
        // Use sendBeacon for reliable logging on page unload
        navigator.sendBeacon('/api/audit/events', JSON.stringify({
          event_type: 'user_logout',
          severity: 'low',
          description: `User ${user.username} logged out`,
          details: { 
            logout_timestamp: new Date().toISOString()
          },
        }));
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [user]);

  const value: AuditContextType = {
    events,
    isLoading,
    filters,
    setFilters,
    logEvent,
    exportAuditLog,
    clearFilters,
    refreshEvents,
  };

  return <AuditContext.Provider value={value}>{children}</AuditContext.Provider>;
};