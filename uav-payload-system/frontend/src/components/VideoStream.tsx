import React, { useState } from 'react';

function VideoStream() {
  const [streamActive, setStreamActive] = useState<boolean>(false);

  const toggleStream = () => {
    setStreamActive(!streamActive);
  };

  const getStreamUrl = () => {
    return `/api/video/stream`;
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
        {streamActive ? (
          <img 
            src={getStreamUrl()} 
            alt="Live video stream"
            className="video-stream"
            onError={(e) => {
              console.error('Video stream error');
              (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTgiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5WaWRlbyBTdHJlYW0gVW5hdmFpbGFibGU8L3RleHQ+PC9zdmc+';
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
            📹 Click "Start Stream" to begin video feed
          </div>
        )}
      </div>
      
      <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#666' }}>
        Stream URL: {getStreamUrl()}
      </div>
    </div>
  );
}

export default VideoStream;