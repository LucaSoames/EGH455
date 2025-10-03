import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

const VideoStream = () => {
  const canvasRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    // Initialize Socket.IO connection
    const newSocket = io('http://localhost:5000', {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
      timeout: 20000,
    });

    setSocket(newSocket);

    newSocket.on('connect', () => {
      setConnected(true);
      console.log('Connected to video stream');
    });

    newSocket.on('disconnect', () => {
      setConnected(false);
      console.log('Disconnected from video stream');
    });

    // Listen for video frames
    newSocket.on('video_frame', (data) => {
      if (canvasRef.current && data.frame) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // Create image from base64 data
        const img = new Image();
        img.onload = () => {
          // Set canvas size to match image
          canvas.width = img.width;
          canvas.height = img.height;
          
          // Draw image on canvas
          ctx.drawImage(img, 0, 0);
          
          // Draw detection boxes if present
          if (data.detections && data.detections.length > 0) {
            drawDetections(ctx, data.detections, img.width, img.height);
          }
          
          // Draw ArUco markers if present
          if (data.aruco_markers && data.aruco_markers.length > 0) {
            drawArucoMarkers(ctx, data.aruco_markers);
          }
          
          setFrameCount(prev => prev + 1);
        };
        
        // Handle both data URL and raw base64
        if (data.frame.startsWith('data:image/')) {
          img.src = data.frame;
        } else {
          img.src = `data:image/jpeg;base64,${data.frame}`;
        }
      }
    });

    // FPS calculation
    const fpsInterval = setInterval(() => {
      setFps(prev => {
        const currentFps = frameCount;
        setFrameCount(0);
        return currentFps;
      });
    }, 1000);

    return () => {
      newSocket.close();
      clearInterval(fpsInterval);
    };
  }, [frameCount]);

  const drawDetections = (ctx, detections, width, height) => {
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = '#ff0000';

    detections.forEach(detection => {
      const { bbox, label, confidence } = detection;
      
      if (bbox && bbox.length === 4) {
        const [x, y, w, h] = bbox;
        
        // Draw bounding box
        ctx.strokeRect(x, y, w, h);
        
        // Draw label background
        const labelText = `${label} (${(confidence * 100).toFixed(1)}%)`;
        const textMetrics = ctx.measureText(labelText);
        ctx.fillStyle = '#ff0000';
        ctx.fillRect(x, y - 25, textMetrics.width + 10, 25);
        
        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x + 5, y - 5);
        ctx.fillStyle = '#ff0000';
      }
    });
  };

  const drawArucoMarkers = (ctx, markers) => {
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.font = '14px Arial';
    ctx.fillStyle = '#00ff00';

    markers.forEach(marker => {
      const { corners, id } = marker;
      
      if (corners && corners.length === 4) {
        // Draw marker outline
        ctx.beginPath();
        ctx.moveTo(corners[0][0], corners[0][1]);
        for (let i = 1; i < corners.length; i++) {
          ctx.lineTo(corners[i][0], corners[i][1]);
        }
        ctx.closePath();
        ctx.stroke();
        
        // Draw marker ID
        const centerX = corners.reduce((sum, corner) => sum + corner[0], 0) / 4;
        const centerY = corners.reduce((sum, corner) => sum + corner[1], 0) / 4;
        
        ctx.fillStyle = '#00ff00';
        ctx.fillText(`ID: ${id}`, centerX - 15, centerY + 5);
      }
    });
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3>Live Camera Feed</h3>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div style={{ 
            padding: '0.25rem 0.5rem',
            borderRadius: '4px',
            backgroundColor: connected ? '#27ae60' : '#e74c3c',
            color: 'white',
            fontSize: '0.8em',
            fontWeight: 'bold'
          }}>
            {connected ? '🟢 CONNECTED' : '🔴 DISCONNECTED'}
          </div>
          <div style={{ fontSize: '0.9em', color: '#666' }}>
            FPS: {fps} | Frames: {frameCount}
          </div>
        </div>
      </div>
      
      <div style={{ 
        border: '2px dashed #ddd',
        borderRadius: '8px',
        padding: '1rem',
        textAlign: 'center',
        backgroundColor: '#f8f9fa'
      }}>
        {connected ? (
          <canvas
            ref={canvasRef}
            style={{
              maxWidth: '100%',
              height: 'auto',
              border: '1px solid #ddd',
              borderRadius: '4px'
            }}
          />
        ) : (
          <div style={{ padding: '2rem', color: '#666' }}>
            <div style={{ fontSize: '3em', marginBottom: '1rem' }}>📷</div>
            <p>Camera feed not available</p>
            <p style={{ fontSize: '0.9em' }}>
              Waiting for connection to TAIP system...
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoStream;