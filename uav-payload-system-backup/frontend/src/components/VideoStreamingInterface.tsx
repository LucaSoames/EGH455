import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  IconButton,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  Switch,
  FormControlLabel,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Fullscreen,
  FullscreenExit,
  VolumeUp,
  VolumeOff,
  VideoSettings,
  CameraAlt,
  Videocam,
  HighQuality,
  Hd,
  PhotoCamera,
  VideoCall,
  RecordVoiceOver,
  ZoomIn,
  ZoomOut,
  CenterFocusStrong,
  Brightness6,
  Contrast,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface VideoStreamConfig {
  uav_id: number;
  stream_url: string;
  resolution: string;
  fps: number;
  bitrate: number;
  codec: string;
  status: 'active' | 'inactive' | 'error';
  camera_type: 'optical' | 'thermal' | 'multispectral';
}

interface UAV {
  id: number;
  serial_number: string;
  model: string;
  status: string;
}

interface VideoStreamingInterfaceProps {
  uavId?: number;
  fullscreen?: boolean;
  onFullscreenChange?: (fullscreen: boolean) => void;
}

const VideoStreamingInterface: React.FC<VideoStreamingInterfaceProps> = ({
  uavId,
  fullscreen = false,
  onFullscreenChange
}) => {
  const [selectedUAV, setSelectedUAV] = useState<number | null>(uavId || null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [volume, setVolume] = useState(80);
  const [isMuted, setIsMuted] = useState(false);
  const [quality, setQuality] = useState<'4k' | 'hd' | 'sd' | 'auto'>('hd');
  const [cameraType, setCameraType] = useState<'optical' | 'thermal' | 'multispectral'>('optical');
  const [brightness, setBrightness] = useState(50);
  const [contrast, setContrast] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(fullscreen);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error' | 'disconnected'>('disconnected');
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const { data: uavs } = useQuery({
    queryKey: ['uavs'],
    queryFn: async () => {
      const response = await axios.get('/api/uavs');
      return response.data.data as UAV[];
    },
  });

  const { data: videoStreams } = useQuery({
    queryKey: ['video-streams'],
    queryFn: async () => {
      const response = await axios.get('/api/video/streams');
      return response.data.data;
    },
    refetchInterval: 10000,
  });

  const { data: streamConfig, refetch: refetchStreamConfig } = useQuery({
    queryKey: ['video-stream', selectedUAV],
    queryFn: async () => {
      if (!selectedUAV) return null;
      
      try {
        const response = await axios.get(`/api/video/settings/${selectedUAV}`);
        return {
          uav_id: selectedUAV,
          stream_url: `/api/video/stream/${selectedUAV}`,
          resolution: response.data.data?.settings?.resolution || '640x480',
          fps: response.data.data?.settings?.fps || 30,
          bitrate: response.data.data?.settings?.bitrate || 1000,
          codec: response.data.data?.settings?.format || 'mjpeg',
          status: 'active' as const,
          camera_type: cameraType
        } as VideoStreamConfig;
      } catch (error) {
        // Fallback configuration if hardware API is not available
        return {
          uav_id: selectedUAV,
          stream_url: `/api/video/stream/${selectedUAV}`,
          resolution: quality === 'hd' ? '1920x1080' : quality === 'sd' ? '640x480' : '3840x2160',
          fps: 30,
          bitrate: quality === 'hd' ? 5000 : quality === 'sd' ? 1000 : 10000,
          codec: 'mjpeg',
          status: 'active' as const,
          camera_type: cameraType
        } as VideoStreamConfig;
      }
    },
    enabled: !!selectedUAV,
    refetchInterval: 10000,
  });

  useEffect(() => {
    if (streamConfig && videoRef.current) {
      const video = videoRef.current;
      
      // Simulate WebRTC or WebSocket video stream connection
      video.src = streamConfig.stream_url || '';
      
      video.addEventListener('loadstart', () => setConnectionStatus('connecting'));
      video.addEventListener('canplay', () => setConnectionStatus('connected'));
      video.addEventListener('error', () => setConnectionStatus('error'));
      
      return () => {
        video.removeEventListener('loadstart', () => setConnectionStatus('connecting'));
        video.removeEventListener('canplay', () => setConnectionStatus('connected'));
        video.removeEventListener('error', () => setConnectionStatus('error'));
      };
    }
  }, [streamConfig]);

  const handlePlay = () => {
    if (videoRef.current && streamConfig) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleStop = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  };

  const handleVolumeChange = (event: Event, newValue: number | number[]) => {
    const volumeValue = newValue as number;
    setVolume(volumeValue);
    if (videoRef.current) {
      videoRef.current.volume = volumeValue / 100;
    }
  };

  const handleMute = () => {
    setIsMuted(!isMuted);
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
    }
  };

  const handleFullscreen = () => {
    if (!isFullscreen && containerRef.current) {
      containerRef.current.requestFullscreen();
    } else if (document.fullscreenElement) {
      document.exitFullscreen();
    }
    setIsFullscreen(!isFullscreen);
    onFullscreenChange?.(!isFullscreen);
  };

  const handleScreenshot = async () => {
    if (!selectedUAV) return;
    
    try {
      const response = await axios.post(`/api/video/snapshot/${selectedUAV}`);
      
      if (response.data.success) {
        // Also create a local screenshot from the video element
        if (videoRef.current) {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          canvas.width = videoRef.current.videoWidth;
          canvas.height = videoRef.current.videoHeight;
          
          if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob((blob) => {
              if (blob) {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `uav_${selectedUAV}_${new Date().toISOString()}.png`;
                a.click();
                URL.revokeObjectURL(url);
              }
            }, 'image/png');
          }
        }
        
        alert(`Screenshot captured: ${response.data.data?.filename || 'Unknown filename'}`);
      }
    } catch (error) {
      console.error('Screenshot failed:', error);
      alert('Failed to capture screenshot');
    }
  };

  const handleRecord = async () => {
    if (!selectedUAV) return;
    
    try {
      const action = isRecording ? 'stop' : 'start';
      const response = await axios.post(`/api/video/recording/${selectedUAV}`, {
        action: action
      });
      
      if (response.data.success) {
        setIsRecording(!isRecording);
        alert(`Recording ${action}ed successfully`);
      }
    } catch (error) {
      console.error('Recording control failed:', error);
      alert(`Failed to ${isRecording ? 'stop' : 'start'} recording`);
    }
  };

  const handleQualityChange = (newQuality: string) => {
    setQuality(newQuality as any);
    // In a real implementation, this would change the stream quality
    refetchStreamConfig();
  };

  const getStatusChip = () => {
    const statusConfig = {
      connecting: { color: 'info', label: 'Connecting' },
      connected: { color: 'success', label: 'Live' },
      error: { color: 'error', label: 'Connection Error' },
      disconnected: { color: 'default', label: 'Disconnected' },
    };
    
    const config = statusConfig[connectionStatus];
    return (
      <Chip
        label={config.label}
        color={config.color as any}
        size="small"
        variant="filled"
      />
    );
  };

  const getQualityIcon = () => {
    switch (quality) {
      case '4k': return <HighQuality />;
      case 'hd': return <Hd />;
      case 'sd': return <CameraAlt />;
      default: return <HighQuality />;
    }
  };

  return (
    <Paper sx={{ p: 2 }} ref={containerRef}>
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            <VideoCall sx={{ mr: 1, verticalAlign: 'middle' }} />
            Live Video Stream
          </Typography>
          {getStatusChip()}
        </Box>

        {/* UAV Selection */}
        <Grid container spacing={2} alignItems="center" sx={{ mb: 2 }}>
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Select UAV</InputLabel>
              <Select
                value={selectedUAV || ''}
                label="Select UAV"
                onChange={(e) => setSelectedUAV(Number(e.target.value))}
              >
                {uavs?.filter(uav => uav.status === 'active').map((uav) => (
                  <MenuItem key={uav.id} value={uav.id}>
                    {uav.serial_number} - {uav.model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Camera Type</InputLabel>
              <Select
                value={cameraType}
                label="Camera Type"
                onChange={(e) => setCameraType(e.target.value as any)}
                disabled={!streamConfig}
              >
                <MenuItem value="optical">
                  <CameraAlt sx={{ mr: 1 }} />
                  Optical Camera
                </MenuItem>
                <MenuItem value="thermal">
                  <CameraAlt sx={{ mr: 1 }} />
                  Thermal Camera
                </MenuItem>
                <MenuItem value="multispectral">
                  <Videocam sx={{ mr: 1 }} />
                  Multispectral
                </MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Quality</InputLabel>
              <Select
                value={quality}
                label="Quality"
                onChange={(e) => handleQualityChange(e.target.value)}
                disabled={!streamConfig}
              >
                <MenuItem value="4k">
                  <HighQuality sx={{ mr: 1 }} />
                  4K Ultra HD
                </MenuItem>
                <MenuItem value="hd">
                  <Hd sx={{ mr: 1 }} />
                  HD 1080p
                </MenuItem>
                <MenuItem value="sd">
                  <CameraAlt sx={{ mr: 1 }} />
                  SD 720p
                </MenuItem>
                <MenuItem value="auto">
                  <HighQuality sx={{ mr: 1 }} />
                  Auto
                </MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Box>

      {/* Video Player */}
      <Box sx={{ position: 'relative', mb: 2 }}>
        {!streamConfig ? (
          <Box
            sx={{
              width: '100%',
              height: 400,
              backgroundColor: '#000',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              borderRadius: 1,
            }}
          >
            <Typography variant="h6">
              {selectedUAV ? 'No video stream available' : 'Select a UAV to view video stream'}
            </Typography>
          </Box>
        ) : (
          <video
            ref={videoRef}
            style={{
              width: '100%',
              height: isFullscreen ? '100vh' : 400,
              backgroundColor: '#000',
              borderRadius: 4,
              filter: `brightness(${brightness}%) contrast(${contrast}%)`,
              transform: `scale(${zoom})`,
            }}
            controls={false}
            autoPlay={false}
            muted={isMuted}
          >
            <source src={streamConfig.stream_url} />
            Your browser does not support video streaming.
          </video>
        )}

        {/* Video Overlay Controls */}
        {streamConfig && (
          <Box
            sx={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              background: 'linear-gradient(transparent, rgba(0,0,0,0.8))',
              color: 'white',
              p: 1,
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              {/* Playback Controls */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <IconButton
                  onClick={handlePlay}
                  sx={{ color: 'white' }}
                  disabled={connectionStatus !== 'connected'}
                >
                  {isPlaying ? <Pause /> : <PlayArrow />}
                </IconButton>
                <IconButton onClick={handleStop} sx={{ color: 'white' }}>
                  <Stop />
                </IconButton>
              </Box>

              {/* Status Info */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {isRecording && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box
                      sx={{
                        width: 8,
                        height: 8,
                        backgroundColor: 'red',
                        borderRadius: '50%',
                        animation: 'blink 1s infinite',
                      }}
                    />
                    <Typography variant="caption">REC</Typography>
                  </Box>
                )}
                {connectionStatus === 'connected' && streamConfig && (
                  <Typography variant="caption">
                    {streamConfig.resolution} • {streamConfig.fps}fps • {cameraType.toUpperCase()}
                  </Typography>
                )}
              </Box>

              {/* Action Controls */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Tooltip title="Take Screenshot">
                  <IconButton
                    onClick={handleScreenshot}
                    sx={{ color: 'white' }}
                    disabled={!isPlaying}
                  >
                    <PhotoCamera />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title={isRecording ? "Stop Recording" : "Start Recording"}>
                  <IconButton
                    onClick={handleRecord}
                    sx={{ color: isRecording ? 'red' : 'white' }}
                    disabled={!isPlaying}
                  >
                    <RecordVoiceOver />
                  </IconButton>
                </Tooltip>

                <Tooltip title="Volume">
                  <IconButton onClick={handleMute} sx={{ color: 'white' }}>
                    {isMuted ? <VolumeOff /> : <VolumeUp />}
                  </IconButton>
                </Tooltip>

                <Tooltip title="Settings">
                  <IconButton
                    onClick={() => setSettingsOpen(true)}
                    sx={{ color: 'white' }}
                  >
                    <VideoSettings />
                  </IconButton>
                </Tooltip>

                <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
                  <IconButton onClick={handleFullscreen} sx={{ color: 'white' }}>
                    {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          </Box>
        )}
      </Box>

      {/* Volume Control */}
      {!isMuted && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" gutterBottom>Volume</Typography>
          <Slider
            value={volume}
            onChange={handleVolumeChange}
            size="small"
            sx={{ ml: 2, width: 100 }}
          />
        </Box>
      )}

      {/* Connection Status */}
      {connectionStatus === 'connecting' && <LinearProgress sx={{ mb: 2 }} />}
      
      {connectionStatus === 'error' && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to connect to video stream. Please check UAV connection and try again.
        </Alert>
      )}

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Video Stream Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                <Brightness6 sx={{ mr: 1, verticalAlign: 'middle' }} />
                Image Adjustments
              </Typography>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography variant="caption">Brightness</Typography>
              <Slider
                value={brightness}
                onChange={(e, val) => setBrightness(val as number)}
                min={25}
                max={200}
                valueLabelDisplay="auto"
                valueLabelFormat={(val) => `${val}%`}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography variant="caption">Contrast</Typography>
              <Slider
                value={contrast}
                onChange={(e, val) => setContrast(val as number)}
                min={25}
                max={200}
                valueLabelDisplay="auto"
                valueLabelFormat={(val) => `${val}%`}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="caption">Zoom</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <IconButton onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}>
                  <ZoomOut />
                </IconButton>
                <Slider
                  value={zoom}
                  onChange={(e, val) => setZoom(val as number)}
                  min={0.5}
                  max={3}
                  step={0.1}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(val) => `${val}x`}
                  sx={{ flex: 1 }}
                />
                <IconButton onClick={() => setZoom(Math.min(3, zoom + 0.1))}>
                  <ZoomIn />
                </IconButton>
              </Box>
            </Grid>

            <Grid item xs={12}>
              <Button
                variant="outlined"
                onClick={() => {
                  setBrightness(50);
                  setContrast(50);
                  setZoom(1);
                }}
                startIcon={<CenterFocusStrong />}
              >
                Reset to Default
              </Button>
            </Grid>
          </Grid>
        </DialogContent>
      </Dialog>

      {/* CSS for blinking animation */}
      <style>
        {`
          @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
          }
        `}
      </style>
    </Paper>
  );
};

export default VideoStreamingInterface;