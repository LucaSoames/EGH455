import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

function VideoStream() {
  const [streamActive, setStreamActive] = useState<boolean>(false);
  const [currentFrame, setCurrentFrame] = useState<string>('');
  const [connected, setConnected] = useState<boolean>(false);

  useEffect(() => {
    const socket = io('http://localhost:5000');

    socket.on('connect', () => {
      console.log('VideoStream: Connected to server');
      setConnected(true);
    });

    socket.on('disconnect', () => {
      console.log('VideoStream: Disconnected from server');
      setConnected(false);
      setCurrentFrame('');
    });

    socket.on('video_frame', (data: { frame: string }) => {
      if (streamActive) {
        setCurrentFrame(`data:image/jpeg;base64,${data.frame}`);
      }
    });

    socket.on('error', (error) => {
      console.error('VideoStream socket error:', error);
    });

    return () => {
      socket.disconnect();
    };
  }, [streamActive]);

  const toggleStream = () => {
    setStreamActive(!streamActive);
    if (!streamActive) {
      // Request initial frame when starting stream
      const socket = io('http://localhost:5000');
      socket.emit('request_video_frame', {});
      socket.disconnect();
    } else {
      // Clear frame when stopping stream
      setCurrentFrame('');
    }
  };

  return (
    <div className="card">
      <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <span style={{ color: '#666', fontSize: '0.9rem' }}>THE UAV Live Feed</span>
        </div>
        
        <button 
          onClick={toggleStream} 
          className={`btn ${streamActive ? 'btn-danger' : 'btn-primary'}`}
        >
          {streamActive ? '⏹️ Stop Stream' : '▶️ Start Stream'}
        </button>
      </div>

      <div className="video-container">
        {streamActive && connected ? (
          currentFrame ? (
            <img 
              src={currentFrame} 
              alt="Live video stream"
              className="video-stream"
              style={{ width: '100%', height: 'auto', maxHeight: '400px', objectFit: 'contain' }}
              onError={() => {
                console.error('Video frame display error');
              }}
            />
          ) : (
            <div 
              style={{
                width: '100%',
                height: '300px',
                backgroundColor: '#f8f9fa',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '2px dashed #dee2e6',
                borderRadius: '8px',
                color: '#6c757d'
              }}
            >
              📡 Waiting for video frames...
            </div>
          )
        ) : !connected ? (
          <div 
            style={{
              width: '100%',
              height: '300px',
              backgroundColor: '#fff5f5',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px dashed #fed7d7',
              borderRadius: '8px',
              color: '#e53e3e'
            }}
          >
            ⚠️ Not connected to server
          </div>
        ) : (
          <div 
            style={{
              width: '100%',
              height: '300px',
              backgroundColor: '#f8f9fa',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px dashed #dee2e6',
              borderRadius: '8px',
              color: '#6c757d'
            }}
          >
            📹 Click "Start Stream" to begin video feed
          </div>
        )}
      </div>
      
      <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#666' }}>
        Status: {connected ? (streamActive ? 'Streaming' : 'Ready') : 'Disconnected'}
      </div>
    </div>
  );
}

export default VideoStream;