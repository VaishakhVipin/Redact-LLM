import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Toaster } from '@/components/ui/toaster';
import { AuthProvider, ProtectedRoute } from '@/contexts/AuthContext';
import { AppLayout } from './components/AppLayout';
import { Dashboard } from './components/Dashboard';
import AnalysisPage from './pages/AnalysisPage';
import LoginPage from './pages/auth/LoginPage';
import { AuthCallback } from './pages/auth/AuthCallback';
import { LandingPage } from './pages/LandingPage';
import { NotFoundPage } from './pages/NotFoundPage';
import './index.css';

const queryClient = new QueryClient();

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Router>
          <AuthProvider>
            <Routes>
              {/* Public routes */}
              {/* Public routes */}
              <Route path="/login" element={<LoginPage />} />
              <Route path="/auth/callback" element={<AuthCallback />} />
              
              {/* Protected routes */}
              <Route element={
                <ProtectedRoute>
                  <AppLayout />
                </ProtectedRoute>
              }>
                <Route path="/analysis/:testId" element={<AnalysisPage />} />
                <Route path="/" element={<Dashboard />} />
              </Route>
              
              {/* Public landing page */}
              <Route path="/welcome" element={<LandingPage />} />
              
              {/* Catch all other routes */}
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
            <Toaster />
          </AuthProvider>
        </Router>
      </TooltipProvider>
    </QueryClientProvider>
  </React.StrictMode>
);
