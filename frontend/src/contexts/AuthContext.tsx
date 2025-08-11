import { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { toast } from 'sonner';
import { supabase, signInWithEmail, verifyOtp, getSession, signOut } from '@/lib/supabase';
import { Session, User } from '@supabase/supabase-js';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signInWithMagicLink: (email: string) => Promise<{ success: boolean; error?: string }>;
  verifyEmailToken: (email: string, token: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

// Create a wrapper component that provides the auth context
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  // Handle auth state changes
  const handleAuthStateChange = useCallback(
    async (event: string, session: Session | null) => {
      setUser(session?.user ?? null);
      
      if (event === 'SIGNED_IN') {
        toast.success('Successfully signed in!');
        // Only redirect if we're on auth-related pages
        if (['/login', '/auth/callback'].includes(location.pathname)) {
          const redirectTo = location.state?.from?.pathname || '/';
          navigate(redirectTo, { replace: true });
        }
      } else if (event === 'SIGNED_OUT') {
        // Only redirect to login if we're not already there
        if (location.pathname !== '/login') {
          navigate('/login', { replace: true });
        }
      }
    },
    [navigate, location.pathname, location.state?.from]
  );

  // Initialize auth state
  useEffect(() => {
    const checkAuth = async () => {
      try {
        setLoading(true);
        const { data: { session }, error } = await getSession();
        
        if (error) throw error;
        
        setUser(session?.user ?? null);
        
        // Only redirect to login if there's no session and we're not on a public route
        if (!session && !['/login', '/', '/auth/callback'].includes(location.pathname)) {
          navigate('/login', { 
            replace: true,
            state: { from: location.pathname }
          });
        }
        // If user is already logged in and on login page, redirect to home or intended path
        else if (session && ['/login', '/auth/callback'].includes(location.pathname)) {
          const redirectTo = location.state?.from?.pathname || '/';
          navigate(redirectTo, { replace: true });
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        toast.error('Failed to check authentication status');
      } finally {
        setLoading(false);
      }
    };

    checkAuth();

    // Set up auth state listener
    const { data: { subscription } } = supabase.auth.onAuthStateChange(handleAuthStateChange);

    return () => {
      subscription?.unsubscribe();
    };
  }, [navigate, location.pathname, handleAuthStateChange]);

  // Sign in with email (Magic Link)
  const signInWithMagicLink = useCallback(async (email: string) => {
    try {
      setLoading(true);
      const { error } = await signInWithEmail(email);

      if (error) throw error;
      toast.success('Check your email for the login link!');
      return { success: true };
    } catch (error) {
      console.error('Error signing in:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to send magic link';
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  }, []);

  // Verify OTP
  const verifyEmailToken = useCallback(async (email: string, token: string) => {
    try {
      setLoading(true);
      const { data, error } = await verifyOtp(email, token);
      
      if (error) throw error;
      if (data.session) {
        setUser(data.session.user);
      }
      return { success: true };
    } catch (error) {
      console.error('Error verifying token:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to verify token';
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  }, []);

  // Logout user
  const logout = useCallback(async () => {
    try {
      setLoading(true);
      const { error } = await signOut();
      if (error) throw error;
      
      setUser(null);
      toast.success('Successfully signed out');
      navigate('/login', { replace: true });
    } catch (error) {
      console.error('Error signing out:', error);
      toast.error('Failed to sign out');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  const value = {
    user,
    loading,
    signInWithMagicLink,
    verifyEmailToken,
    logout,
    isAuthenticated: !!user,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Custom hook to use the auth context
export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// Protected route component
export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (!loading && !isAuthenticated) {
      navigate('/login', { 
        replace: true,
        state: { from: location.pathname }
      });
    }
  }, [isAuthenticated, loading, navigate, location.pathname]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
      </div>
    );
  }

  return isAuthenticated ? <>{children}</> : null;
}
