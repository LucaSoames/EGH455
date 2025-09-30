import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import axios from 'axios';

interface TelemetryData {
  id: number;
  uav_id: number;
  mission_id?: number;
  latitude: number;
  longitude: number;
  altitude: number;
  heading: number;
  speed: number;
  vertical_speed: number;
  battery_level: number;
  signal_strength: number;
  gps_satellites: number;
  system_status: 'normal' | 'warning' | 'error';
  error_messages?: string;
  temperature?: number;
  wind_speed?: number;
  wind_direction?: number;
  timestamp: string;
}

interface SocketContextType {
  telemetryData: TelemetryData[];
  latestTelemetry: TelemetryData | null;
  isConnected: boolean;
  refreshTelemetry: () => void;
  joinTelemetryUpdates: () => void; // For compatibility with existing components
}

const SocketContext = createContext<SocketContextType | undefined>(undefined);

export const useSocket = (): SocketContextType => {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
};

export const SocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [telemetryData, setTelemetryData] = useState<TelemetryData[]>([]);
  const [latestTelemetry, setLatestTelemetry] = useState<TelemetryData | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const fetchTelemetryData = useCallback(async () => {
    try {
      const response = await axios.get('/api/telemetry/latest');
      if (response.data.success && response.data.data) {
        setTelemetryData(response.data.data);
        setLatestTelemetry(response.data.data[0] || null);
        setIsConnected(true);
      }
    } catch (error) {
      console.error('Failed to fetch telemetry data:', error);
      setIsConnected(false);
    }
  }, []);

  const refreshTelemetry = useCallback(() => {
    fetchTelemetryData();
  }, [fetchTelemetryData]);

  const joinTelemetryUpdates = useCallback(() => {
    // Compatibility function - in polling mode, this just refreshes data
    fetchTelemetryData();
  }, [fetchTelemetryData]);

  // Simple polling every 5 seconds instead of WebSocket
  useEffect(() => {
    // Initial fetch
    fetchTelemetryData();

    // Set up polling
    const interval = setInterval(fetchTelemetryData, 5000);

    return () => {
      clearInterval(interval);
    };
  }, [fetchTelemetryData]);

  const value: SocketContextType = {
    telemetryData,
    latestTelemetry,
    isConnected,
    refreshTelemetry,
    joinTelemetryUpdates,
  };

  return <SocketContext.Provider value={value}>{children}</SocketContext.Provider>;
};