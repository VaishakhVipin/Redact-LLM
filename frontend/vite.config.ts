import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { componentTagger } from 'lovable-tagger';

// Load environment variables
const env = loadEnv('development', process.cwd(), '');

// Ensure required environment variables are present
if (!env.VITE_SUPABASE_URL || !env.VITE_SUPABASE_KEY) {
  throw new Error('Missing required environment variables. Please check your .env file');
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: '127.0.0.1',
    port: 8080,
    strictPort: true,
    cors: true,
    proxy: {
      // Proxy Supabase auth requests
      '/auth/v1': {
        target: env.VITE_SUPABASE_URL,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/auth\/v1/, '/auth/v1'),
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader('apikey', env.VITE_SUPABASE_KEY);
          });
        },
      },
      // Proxy Supabase REST requests
      '/rest/v1': {
        target: env.VITE_SUPABASE_URL,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/rest\/v1/, '/rest/v1'),
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader('apikey', env.VITE_SUPABASE_KEY);
          });
        },
      },
    },
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, PATCH, OPTIONS',
      'Access-Control-Allow-Headers': 'X-Requested-With, content-type, Authorization, apikey',
    },
  },
  plugins: [
    react(),
    mode === 'development' && componentTagger(),
  ].filter(Boolean) as any[],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-dom/client',
      'react-router-dom',
      '@tanstack/react-query',
    ],
    exclude: ['js-big-decimal'],
    force: true, // Force dependency optimization
  },
  define: {
    'import.meta.env.VITE_SUPABASE_URL': JSON.stringify(env.VITE_SUPABASE_URL),
    'import.meta.env.VITE_SUPABASE_KEY': JSON.stringify(env.VITE_SUPABASE_KEY),
    'import.meta.env.VITE_SITE_URL': JSON.stringify(env.VITE_SITE_URL || 'http://localhost:8080'),
  },
}));
