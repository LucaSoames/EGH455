import React, { useState } from 'react';
import { useAuth } from './AuthContext';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const success = await login(username, password);
    if (!success) {
      setError('Invalid username or password');
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <div className="header">
        <h1>UAV Payload System</h1>
        <p>Please log in to access the system</p>
      </div>
      
      <div className="card" style={{ maxWidth: '400px', margin: '2rem auto' }}>
        <h2>Login</h2>
        
        {error && (
          <div style={{ color: '#e74c3c', marginBottom: '1rem', padding: '0.5rem', backgroundColor: '#fadbd8', borderRadius: '4px' }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              placeholder="Enter your username"
            />
          </div>
          
          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="Enter your password"
            />
          </div>
          
          <button type="submit" className="btn btn-primary" disabled={loading} style={{ width: '100%' }}>
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
        
        <div style={{ marginTop: '2rem', fontSize: '0.9rem', color: '#666' }}>
          <strong>Demo Credentials:</strong><br />
          Admin: admin / admin123<br />
          Operator: operator / operator123
        </div>
      </div>
    </div>
  );
}

export default Login;