# Frontend Quick Start Guide

## 🚀 Getting Started

This guide will help you quickly integrate with the Redis-Driven LLM Ops Pipeline API.

## Prerequisites

1. **Backend Running**: Ensure the backend is running on `http://localhost:8000`
2. **Redis Connection**: Backend should be connected to Redis
3. **Environment Variables**: Backend should have proper API keys configured

## Quick Setup

### 1. Install Dependencies

```bash
# If using TypeScript
npm install --save-dev typescript @types/node

# If using React
npm install react react-dom @types/react @types/react-dom

# If using Next.js
npx create-next-app@latest my-frontend --typescript
```

### 2. Copy Type Definitions

Copy the following files to your frontend project:
- `frontend-types.ts` - Complete TypeScript interfaces
- `sample-api-client.ts` - Reference API client implementation

### 3. Basic Integration

```typescript
// src/lib/api-client.ts
import { createAPIClient, createWebSocketClient } from './sample-api-client';

export const apiClient = createAPIClient('http://localhost:8000');
export const wsClient = createWebSocketClient('http://localhost:8000');
```

### 4. Basic Usage Example

```typescript
// src/components/AttackGenerator.tsx
import React, { useState } from 'react';
import { apiClient } from '../lib/api-client';
import { AttackRequest, AttackResponse } from '../types';

export const AttackGenerator: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [attacks, setAttacks] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateAttacks = async () => {
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const request: AttackRequest = {
        prompt,
        attack_types: ['jailbreak', 'hallucination', 'advanced']
      };

      const response: AttackResponse = await apiClient.generateAttacks(request);
      setAttacks(response.attacks);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Attack Generator</h2>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter your prompt here..."
        rows={4}
        cols={50}
      />
      <br />
      <button onClick={generateAttacks} disabled={loading}>
        {loading ? 'Generating...' : 'Generate Attacks'}
      </button>
      
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}
      
      {attacks.length > 0 && (
        <div>
          <h3>Generated Attacks ({attacks.length})</h3>
          <ul>
            {attacks.map((attack, index) => (
              <li key={index}>{attack}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
```

### 5. Real-time Updates

```typescript
// src/components/StatsMonitor.tsx
import React, { useEffect, useState } from 'react';
import { apiClient, wsClient } from '../lib/api-client';
import { ComprehensiveStatsResponse, Verdict } from '../types';

export const StatsMonitor: React.FC = () => {
  const [stats, setStats] = useState<ComprehensiveStatsResponse | null>(null);
  const [verdicts, setVerdicts] = useState<Verdict[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    // Load initial stats
    loadStats();
    
    // Connect to WebSocket for real-time updates
    connectWebSocket();
    
    // Poll for stats updates every 30 seconds
    const interval = setInterval(loadStats, 30000);
    
    return () => {
      clearInterval(interval);
      wsClient.disconnect();
    };
  }, []);

  const loadStats = async () => {
    try {
      const statsData = await apiClient.getStats();
      setStats(statsData);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const connectWebSocket = async () => {
    try {
      await wsClient.connect();
      setConnected(true);
      
      wsClient.on('verdict', (verdict: Verdict) => {
        setVerdicts(prev => [verdict, ...prev.slice(0, 9)]); // Keep last 10
      });
    } catch (error) {
      console.error('WebSocket connection failed:', error);
    }
  };

  if (!stats) return <div>Loading stats...</div>;

  return (
    <div>
      <h2>System Statistics</h2>
      <div style={{ color: connected ? 'green' : 'red' }}>
        Status: {connected ? 'Connected' : 'Disconnected'}
      </div>
      
      <div>
        <h3>Key Metrics</h3>
        <p>Total Prompts: {stats.comprehensive_metrics.total_prompts}</p>
        <p>Total Attacks: {stats.comprehensive_metrics.total_attacks}</p>
        <p>Average Robustness: {stats.comprehensive_metrics.average_robustness_score.toFixed(1)}%</p>
        <p>Jailbreaks Caught: {stats.comprehensive_metrics.jailbreaks_caught}</p>
      </div>
      
      <div>
        <h3>Recent Verdicts ({verdicts.length})</h3>
        {verdicts.map((verdict, index) => (
          <div key={index} style={{ 
            border: '1px solid #ccc', 
            margin: '5px', 
            padding: '10px',
            backgroundColor: verdict.risk_level === 'high' ? '#ffebee' : 
                           verdict.risk_level === 'medium' ? '#fff3e0' : '#e8f5e8'
          }}>
            <strong>{verdict.attack_type}</strong> - {verdict.risk_level} risk
            <br />
            Alerts: {verdict.alerts.join(', ')}
          </div>
        ))}
      </div>
    </div>
  );
};
```

## Key Endpoints to Use

### 1. Generate Attacks
```typescript
const response = await apiClient.generateAttacks({
  prompt: "Your prompt here",
  attack_types: ["jailbreak", "hallucination", "advanced"]
});
```

### 2. Get Comprehensive Stats
```typescript
const stats = await apiClient.getStats();
// Contains all metrics, rate limiting info, and Redis features
```

### 3. Get Recent Verdicts
```typescript
const verdicts = await apiClient.getRecentVerdicts(10);
// Get last 10 verdicts with detailed evaluations
```

### 4. Real-time Updates
```typescript
await wsClient.connect();
wsClient.on('verdict', (verdict) => {
  // Handle new verdict in real-time
  console.log('New verdict:', verdict);
});
```

## Rate Limiting

The API implements comprehensive rate limiting:

- **Attack Generation**: 50 requests/hour per user
- **API Calls**: 200 requests/hour per user
- **Global Limits**: 1000 requests/hour for attack generation

Handle rate limiting in your UI:

```typescript
try {
  const response = await apiClient.generateAttacks(request);
  // Success
} catch (error) {
  if (error.message.includes('Rate limit exceeded')) {
    // Show user-friendly rate limit message
    setError('You have exceeded the rate limit. Please try again later.');
  } else {
    setError('An error occurred. Please try again.');
  }
}
```

## Error Handling

Always implement proper error handling:

```typescript
const handleAPIError = (error: any) => {
  if (error.message.includes('Rate limit exceeded')) {
    return 'Rate limit exceeded. Please wait before trying again.';
  }
  
  if (error.message.includes('Network error')) {
    return 'Network error. Please check your connection.';
  }
  
  return 'An unexpected error occurred. Please try again.';
};
```

## Performance Tips

1. **Cache Stats**: Don't poll `/stats` too frequently (30s intervals are good)
2. **Debounce Input**: Debounce user input for attack generation
3. **Show Loading States**: Always show loading states during API calls
4. **Handle Disconnections**: Implement WebSocket reconnection logic
5. **Pagination**: Use limit parameters for large result sets

## Testing

Test your integration:

```bash
# Start the backend
cd backend
python main.py

# In another terminal, start the worker
cd backend
python worker.py

# Test the system
cd backend
python test_final_system.py
```

## Common Issues

1. **CORS Errors**: Ensure backend has CORS configured
2. **WebSocket Connection**: WebSocket implementation is placeholder - implement proper WebSocket server
3. **Rate Limiting**: Monitor 429 responses and implement backoff
4. **Redis Connection**: Ensure Redis is running and accessible

## Next Steps

1. Implement proper WebSocket server for real-time updates
2. Add authentication if needed
3. Implement proper error boundaries
4. Add comprehensive logging
5. Implement retry logic for failed requests

## Support

- Check the `API_DOCUMENTATION.md` for complete endpoint details
- Use the TypeScript interfaces for type safety
- Monitor the `/stats` endpoint for system health
- Check the test files for usage examples 