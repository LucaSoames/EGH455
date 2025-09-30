import React, { createContext, useContext, useReducer, useEffect } from 'react';
import axios from 'axios';

// Configure API base URL for axios: use env var if provided, otherwise default to backend port 5000
const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';
axios.defaults.baseURL = API_BASE;

interface User {
  id: number;
  username: string;
  email: string;
  role: 'admin' | 'operator' | 'viewer';
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;
}

type AuthAction =
  | { type: 'LOGIN_START' }
  | { type: 'LOGIN_SUCCESS'; payload: { user: User; token: string } }
  | { type: 'LOGIN_ERROR'; payload: string }
  | { type: 'LOGOUT' }
  | { type: 'CLEAR_ERROR' };

const initialState: AuthState = {
  user: null,
  token: localStorage.getItem('token'),
  isLoading: false,
  error: null,
};

const authReducer = (state: AuthState, action: AuthAction): AuthState => {
  switch (action.type) {
    case 'LOGIN_START':
      return { ...state, isLoading: true, error: null };
    case 'LOGIN_SUCCESS':
      return {
        ...state,
        isLoading: false,
        error: null,
        user: action.payload.user,
        token: action.payload.token,
      };
    case 'LOGIN_ERROR':
      return { ...state, isLoading: false, error: action.payload };
    case 'LOGOUT':
      return { ...state, user: null, token: null, error: null };
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    default:
      return state;
  }
};

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Set up axios interceptor for token
  useEffect(() => {
    if (state.token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${state.token}`;
      localStorage.setItem('token', state.token);
    } else {
      delete axios.defaults.headers.common['Authorization'];
      localStorage.removeItem('token');
    }
  }, [state.token]);

  // Check if user is already logged in on mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      // Verify token with backend
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      axios.get('/api/auth/profile')
        .then(response => {
          if (response.data.success) {
            dispatch({
              type: 'LOGIN_SUCCESS',
              payload: {
                user: response.data.data,
                token: token,
              },
            });
          } else {
            // Invalid token
            localStorage.removeItem('token');
            dispatch({ type: 'LOGOUT' });
          }
        })
        .catch(() => {
          // Invalid token
          localStorage.removeItem('token');
          dispatch({ type: 'LOGOUT' });
        });
    }
  }, []);

  const login = async (username: string, password: string): Promise<void> => {
    dispatch({ type: 'LOGIN_START' });
    
    try {
      const response = await axios.post('/api/auth/login', {
        username,
        password,
      });

      if (response.data.success) {
        dispatch({
          type: 'LOGIN_SUCCESS',
          payload: {
            user: response.data.user,
            token: response.data.access_token,
          },
        });
      } else {
        dispatch({
          type: 'LOGIN_ERROR',
          payload: response.data.error || 'Login failed',
        });
      }
    } catch (error: any) {
      dispatch({
        type: 'LOGIN_ERROR',
        payload: error.response?.data?.error || 'Login failed',
      });
    }
  };

  const logout = (): void => {
    dispatch({ type: 'LOGOUT' });
  };

  const clearError = (): void => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  const value: AuthContextType = {
    user: state.user,
    token: state.token,
    isLoading: state.isLoading,
    error: state.error,
    login,
    logout,
    clearError,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
