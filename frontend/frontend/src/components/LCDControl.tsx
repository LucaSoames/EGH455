import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

interface LCDControlProps {
  // Optional props for styling
}

function LCDControl() {
  const [currentTab, setCurrentTab] = useState<number>(0);
  const [connected, setConnected] = useState<boolean>(false);
  const [sending, setSending] = useState<boolean>(false);

  const tabs = [
    { index: 0, name: 'IP', icon: 'IP', description: 'IP Address' },
    { index: 1, name: 'CAM', icon: 'CAM', description: 'Camera Feed' },
    { index: 2, name: 'TEMP', icon: 'TEMP', description: 'Temperature' }
  ];

  useEffect(() => {
    // Connect to the same host that served the page
    const socket = io(window.location.origin, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    socket.on('connect', () => {
      console.log('LCDControl: Connected to server');
      setConnected(true);
    });

    socket.on('disconnect', () => {
      console.log('LCDControl: Disconnected from server');
      setConnected(false);
    });

    // Listen for LCD tab updates from server (if we want to sync state)
    socket.on('lcd_tab_update', (data: { tab_index: number }) => {
      setCurrentTab(data.tab_index);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const setLCDTab = async (tabIndex: number) => {
    setSending(true);
    try {
      const response = await fetch(`${window.location.origin}/api/lcd/tab`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tab_index: tabIndex }),
      });

      if (response.ok) {
        setCurrentTab(tabIndex);
        console.log(`LCD tab set to: ${tabIndex}`);
      } else {
        const error = await response.json();
        console.error('Failed to set LCD tab:', error);
      }
    } catch (error) {
      console.error('Error setting LCD tab:', error);
    } finally {
      setSending(false);
    }
  };

  const cycleTab = () => {
    const nextTab = (currentTab + 1) % tabs.length;
    setLCDTab(nextTab);
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3>LCD Display Control</h3>
        <div style={{ 
          padding: '0.25rem 0.5rem',
          borderRadius: '4px',
          backgroundColor: connected ? '#27ae60' : '#e74c3c',
          color: 'white',
          fontSize: '0.8rem',
          fontWeight: 'bold'
        }}>
          {connected ? 'CONNECTED' : 'DISCONNECTED'}
        </div>
      </div>

      <p style={{ color: '#666', marginBottom: '1rem', fontSize: '0.9rem' }}>
        Control the LCD display on the Raspberry Pi remotely
      </p>

      {/* Current Tab Display */}
      <div style={{
        padding: '1rem',
        backgroundColor: '#f8f9fa',
        borderRadius: '4px',
        marginBottom: '1rem',
        textAlign: 'center'
      }}>
        <div style={{ fontSize: '0.85rem', color: '#666', marginBottom: '0.5rem' }}>
          Current LCD Tab
        </div>
        <div style={{ fontSize: '2rem', marginBottom: '0.25rem' }}>
          {tabs[currentTab].icon}
        </div>
        <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#3498db' }}>
          {tabs[currentTab].name}
        </div>
        <div style={{ fontSize: '0.85rem', color: '#666' }}>
          {tabs[currentTab].description}
        </div>
      </div>

      {/* Tab Selection Buttons */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(3, 1fr)', 
        gap: '0.5rem',
        marginBottom: '1rem'
      }}>
        {tabs.map((tab) => (
          <button
            key={tab.index}
            onClick={() => setLCDTab(tab.index)}
            disabled={!connected || sending}
            style={{
              padding: '1rem',
              border: currentTab === tab.index ? '2px solid #3498db' : '1px solid #ddd',
              borderRadius: '4px',
              backgroundColor: currentTab === tab.index ? '#e3f2fd' : 'white',
              cursor: connected && !sending ? 'pointer' : 'not-allowed',
              opacity: connected && !sending ? 1 : 0.6,
              transition: 'all 0.2s',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '0.25rem'
            }}
          >
            <span style={{ fontSize: '1.5rem' }}>{tab.icon}</span>
            <span style={{ 
              fontSize: '0.9rem', 
              fontWeight: currentTab === tab.index ? 'bold' : 'normal',
              color: currentTab === tab.index ? '#3498db' : '#333'
            }}>
              {tab.name}
            </span>
            <span style={{ fontSize: '0.75rem', color: '#666' }}>
              {tab.description}
            </span>
          </button>
        ))}
      </div>

      {/* Cycle Button */}
      <button
        onClick={cycleTab}
        disabled={!connected || sending}
        className="btn btn-primary"
        style={{
          width: '100%',
          padding: '0.75rem',
          opacity: connected && !sending ? 1 : 0.6,
          cursor: connected && !sending ? 'pointer' : 'not-allowed'
        }}
      >
        {sending ? 'Sending...' : 'Cycle to Next Tab'}
      </button>

      {!connected && (
        <div style={{
          marginTop: '1rem',
          padding: '0.75rem',
          backgroundColor: '#fff3cd',
          border: '1px solid #ffc107',
          borderRadius: '4px',
          color: '#856404',
          fontSize: '0.85rem',
          textAlign: 'center'
        }}>
          Not connected to Pi - LCD control unavailable
        </div>
      )}
    </div>
  );
}

export default LCDControl;