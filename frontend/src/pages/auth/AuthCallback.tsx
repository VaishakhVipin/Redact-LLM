import { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { verifyOtp } from '@/lib/supabase';
import { useAuth } from '@/contexts/AuthContext';
import { Loader2, CheckCircle2, XCircle, Mail } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

export function AuthCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState<'loading' | 'success' | 'error' | 'verifying'>('loading');
  const [error, setError] = useState<string | null>(null);
  const { verifyEmailToken } = useAuth();
  const email = searchParams.get('email');
  const token = searchParams.get('token_hash');
  const type = searchParams.get('type');

  const handleVerification = async () => {
    if (!token || !type || !email) {
      setStatus('error');
      setError('Invalid verification link');
      return;
    }

    try {
      setStatus('verifying');
      
      // First verify the OTP with Supabase
      const { error: otpError } = await verifyOtp(email, token);
      if (otpError) throw new Error(otpError.message);
      
      // Then verify with our backend
      const { error: tokenError } = await verifyEmailToken(email, token);
      if (tokenError) throw new Error(tokenError);
      
      toast.success('Email verified successfully!');
      setStatus('success');
      
      // Redirect after a short delay
      setTimeout(() => {
        const redirectTo = localStorage.getItem('redirectTo') || '/';
        localStorage.removeItem('redirectTo');
        navigate(redirectTo, { replace: true });
      }, 1500);
      
    } catch (error) {
      console.error('Verification error:', error);
      setStatus('error');
      setError(error instanceof Error ? error.message : 'Failed to verify email');
      toast.error('Verification failed. Please try again.');
    }
  };

  useEffect(() => {
    if (token && type === 'email' && email) {
      handleVerification();
    } else {
      setStatus('error');
      setError('Missing required parameters in the verification link');
    }
  }, [token, type, email]);

  const handleResendEmail = async () => {
    try {
      // TODO: Implement resend verification email logic
      toast.success('Verification email resent!');
    } catch (error) {
      console.error('Resend error:', error);
      toast.error('Failed to resend verification email');
    }
  };

  if (status === 'loading' || status === 'verifying') {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="max-w-md w-full space-y-6 text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
          <h1 className="text-2xl font-bold">
            {status === 'verifying' ? 'Verifying your email...' : 'Loading...'}
          </h1>
          <p className="text-muted-foreground">
            Please wait while we verify your email address.
          </p>
          {email && (
            <Button
              variant="outline"
              className="w-full"
              onClick={handleResendEmail}
            >
              <Mail className="mr-2 h-4 w-4" />
              Resend Verification Email
            </Button>
          )}
        </div>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="max-w-md w-full space-y-6 text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
            <XCircle className="h-8 w-8 text-red-600" />
          </div>
          <h1 className="text-2xl font-bold">Verification Failed</h1>
          <p className="text-muted-foreground">
            {error || 'An unexpected error occurred during verification.'}
          </p>
          <div className="pt-4 space-y-3">
            <Button 
              className="w-full" 
              onClick={() => window.location.href = '/login'}
            >
              Return to Login
            </Button>
          </div>
        </div>
      </div>
    );
  }
}
