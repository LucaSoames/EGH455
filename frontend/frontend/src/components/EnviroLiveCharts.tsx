import React, { useEffect, useMemo, useState } from 'react';
import io from 'socket.io-client';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';

type TelemetryPacket = {
  temperature?: number;       // °C
  humidity?: number;          // %
  gauge_pressure_bar?: number; // bar
  // ...other fields as needed
};

type Point = {
  t: number;                // epoch ms
  temperature?: number;
  humidity?: number;
  pressure?: number;
};

const SOCKET_URL = typeof window !== 'undefined' ? `${window.location.protocol}//${window.location.hostname}:3000` : 'http://localhost:3000';
const MAX_POINTS = 300; // keep last N points

function timeTick(t: number) {
  const d = new Date(t);
  return d.toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' });
}

function Stat({ label, value, unit, color }: { label: string; value: string; unit?: string; color?: string }) {
  return (
    <div className="telemetry-item" style={{ minWidth: 160 }}>
      <span>{label}</span>
      <strong style={{ color }}>{value}{unit ? ` ${unit}` : ''}</strong>
    </div>
  );
}

export default function EnviroLiveCharts() {
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState<Point[]>([]);

  useEffect(() => {
    const socket = io(SOCKET_URL);

    socket.on('connect', () => setConnected(true));
    socket.on('disconnect', () => setConnected(false));

    socket.on('telemetry_update', (pkt: TelemetryPacket) => {
      const t = Date.now();
      const next: Point = {
        t,
        temperature: pkt.temperature,
        humidity: pkt.humidity,
        pressure: pkt.gauge_pressure_bar,
      };
      setData(prev => {
        const merged = [...prev, next];
        return merged.length > MAX_POINTS ? merged.slice(merged.length - MAX_POINTS) : merged;
      });
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const latest = data.length ? data[data.length - 1] : undefined;

  const tempColor = '#e67e22';
  const humColor = '#3498db';
  const pressColor = '#27ae60';

  const hasAnyData = useMemo(
    () => data.some(d => d.temperature != null || d.humidity != null || d.pressure != null),
    [data]
  );

  return (
    <div className="card" style={{ marginTop: '2rem' }}>
      <h3>Enviro Sensors (Live)</h3>

      {!connected && (
        <p style={{ color: '#666', marginTop: 0 }}>Connecting to TAIP system...</p>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
        <Stat
          label="Temperature"
          value={latest?.temperature != null ? latest.temperature.toFixed(1) : '—'}
          unit="°C"
          color={tempColor}
        />
        <Stat
          label="Humidity"
          value={latest?.humidity != null ? latest.humidity.toFixed(1) : '—'}
          unit="%"
          color={humColor}
        />
        <Stat
          label="Pressure"
          value={latest?.pressure != null ? latest.pressure.toFixed(2) : '—'}
          unit="bar"
          color={pressColor}
        />
      </div>

      {!hasAnyData ? (
        <p style={{ color: '#666', textAlign: 'center', padding: '1rem' }}>
          Waiting for enviro telemetry…
        </p>
      ) : (
        <div style={{ display: 'grid', gap: '1rem' }}>
          {/* Temperature */}
          <div style={{ background: '#f8f9fa', borderRadius: 4, padding: '0.5rem' }}>
            <div style={{ margin: '0.25rem 0 0.5rem 0', fontWeight: 600, color: tempColor }}>Temperature (°C)</div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="t" tickFormatter={timeTick} minTickGap={24} />
                <YAxis domain={[-10, 60]} />
                <Tooltip labelFormatter={(v) => timeTick(v as number)} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="temperature"
                  stroke={tempColor}
                  isAnimationActive={false}
                  dot={false}
                  name="°C"
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Humidity */}
          <div style={{ background: '#f8f9fa', borderRadius: 4, padding: '0.5rem' }}>
            <div style={{ margin: '0.25rem 0 0.5rem 0', fontWeight: 600, color: humColor }}>Humidity (%)</div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="t" tickFormatter={timeTick} minTickGap={24} />
                <YAxis domain={[0, 100]} />
                <Tooltip labelFormatter={(v) => timeTick(v as number)} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="humidity"
                  stroke={humColor}
                  isAnimationActive={false}
                  dot={false}
                  name="%"
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Pressure */}
          <div style={{ background: '#f8f9fa', borderRadius: 4, padding: '0.5rem' }}>
            <div style={{ margin: '0.25rem 0 0.5rem 0', fontWeight: 600, color: pressColor }}>Gauge Pressure (bar)</div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="t" tickFormatter={timeTick} minTickGap={24} />
                <YAxis domain={[0, 10]} />
                <Tooltip labelFormatter={(v) => timeTick(v as number)} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="pressure"
                  stroke={pressColor}
                  isAnimationActive={false}
                  dot={false}
                  name="bar"
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
