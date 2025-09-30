import React, { Component, ReactNode } from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Card,
  CardContent,
  Typography,
} from '@mui/material';
import { Refresh, BugReport } from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: any;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    this.setState({
      error,
      errorInfo,
    });

    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    // In production, you might want to send this to an error reporting service
    if (process.env.NODE_ENV === 'production') {
      // Example: Sentry, LogRocket, etc.
      // errorReportingService.captureException(error, { extra: errorInfo });
    }
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '400px',
            p: 3,
          }}
        >
          <Card sx={{ maxWidth: 600, width: '100%' }}>
            <CardContent>
              <Alert severity="error" sx={{ mb: 2 }}>
                <AlertTitle>Something went wrong</AlertTitle>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  We're sorry, but something unexpected happened. Please try refreshing
                  the page or contact support if the problem persists.
                </Typography>
              </Alert>

              {process.env.NODE_ENV === 'development' && this.state.error && (
                <Box sx={{ mt: 2, mb: 2 }}>
                  <Typography variant="h6" color="error" gutterBottom>
                    <BugReport sx={{ verticalAlign: 'middle', mr: 1 }} />
                    Error Details (Development Mode)
                  </Typography>
                  <Card variant="outlined" sx={{ bgcolor: 'grey.50' }}>
                    <CardContent>
                      <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', fontSize: '0.75rem' }}>
                        {this.state.error.toString()}
                        {this.state.errorInfo?.componentStack}
                      </Typography>
                    </CardContent>
                  </Card>
                </Box>
              )}

              <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<Refresh />}
                  onClick={this.handleReload}
                >
                  Refresh Page
                </Button>
                <Button
                  variant="outlined"
                  onClick={this.handleReset}
                >
                  Try Again
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;