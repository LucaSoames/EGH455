import React, { useState } from 'react';
import TelemetryDisplay from './TelemetryDisplay';
import VideoStream from './VideoStream';
import AuditLogs from './AuditLogs';
import LCDControl from './LCDControl';

function Dashboard() {
  const [activeTab, setActiveTab] = useState<'overview' | 'audit'>('overview');

  return (
    <div className="container">
      <div className="header">
        <div>
          <h1>UAV Payload System</h1>
          <p>Real-time telemetry, video streaming, and audit logging</p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', gap: '1rem', borderBottom: '1px solid #dee2e6' }}>
          <button
            onClick={() => setActiveTab('overview')}
            style={{
              padding: '1rem 2rem',
              border: 'none',
              backgroundColor: 'transparent',
              borderBottom: activeTab === 'overview' ? '2px solid #3498db' : '2px solid transparent',
              color: activeTab === 'overview' ? '#3498db' : '#666',
              fontWeight: activeTab === 'overview' ? 'bold' : 'normal',
              cursor: 'pointer'
            }}
          >
            📊 Overview
          </button>
          <button
            onClick={() => setActiveTab('audit')}
            style={{
              padding: '1rem 2rem',
              border: 'none',
              backgroundColor: 'transparent',
              borderBottom: activeTab === 'audit' ? '2px solid #3498db' : '2px solid transparent',
              color: activeTab === 'audit' ? '#3498db' : '#666',
              fontWeight: activeTab === 'audit' ? 'bold' : 'normal',
              cursor: 'pointer'
            }}
          >
            📋 Audit Logs
          </button>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
            <div>
              <h2>Live Telemetry</h2>
              <TelemetryDisplay />
            </div>

            <div>
              <h2>Video Stream</h2>
              <VideoStream />
            </div>
          </div>

          {/* LCD Control Section - Full Width Below */}
          <div style={{ marginTop: '2rem' }}>
            <h2>LCD Display Control</h2>
            <LCDControl />
          </div>
        </>
      )}

      {activeTab === 'audit' && (
        <div>
          <h2>System Audit Logs</h2>
          <AuditLogs />
        </div>
      )}
    </div>
  );
}

export default Dashboard;