import { createClient } from '@supabase/supabase-js';

// In Vite, environment variables need to be prefixed with VITE_
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseKey = import.meta.env.VITE_SUPABASE_KEY;
const siteUrl = import.meta.env.VITE_SITE_URL || 'https://redact-llm.vercel.app';

if (!supabaseUrl || !supabaseKey) {
  throw new Error('Missing Supabase environment variables. Please check your .env file');
}

// Create the Supabase client with production settings
export const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true,
    flowType: 'pkce',
    debug: false, // Disable debug in production
    storageKey: 'redact-llm-session',
    storage: window.localStorage
  }
});

// Set secure cookies for production in the browser
if (import.meta.env.PROD && typeof window !== 'undefined') {
  const cookieOptions = {
    domain: 'redact-llm.vercel.app',
    path: '/',
    secure: true,
    sameSite: 'lax' as const,
    maxAge: 60 * 60 * 24 * 7 // 7 days
  };
  
  // Set auth cookies
  document.cookie = `sb:token=; domain=${cookieOptions.domain}; path=${cookieOptions.path}; secure; samesite=${cookieOptions.sameSite}; max-age=${cookieOptions.maxAge}`;
  document.cookie = `sb:refresh_token=; domain=${cookieOptions.domain}; path=${cookieOptions.path}; secure; samesite=${cookieOptions.sameSite}; max-age=${cookieOptions.maxAge}`;
}

// Helper function for email sign in with OTP (Magic Link)
export const signInWithEmail = async (email: string) => {
  try {
    const redirectTo = `${siteUrl}/auth/callback`;
    
    const { data, error } = await supabase.auth.signInWithOtp({
      email,
      options: {
        emailRedirectTo: redirectTo,
        shouldCreateUser: true,
      },
    });
    
    if (error) {
      console.error('Sign in error:');
      throw error;
    }

    return { data, error: null };
  } catch (error) {
    console.error('Error in signInWithEmail:');
    return { data: null, error };
  }
};

// Verify OTP for email/passwordless login
export const verifyOtp = async (email: string, token: string) => {
  return await supabase.auth.verifyOtp({
    email,
    token,
    type: 'email',
  });
};

// Get the current session
export const getSession = async () => {
  return await supabase.auth.getSession();
};

// Sign out
export const signOut = async () => {
  return await supabase.auth.signOut();
};
