import { useEffect } from 'react';
import { supabase } from '@/lib/supabase';

export function TestSupabase() {
  useEffect(() => {
    const testConnection = async () => {
      try {
        const { data, error } = await supabase.auth.getSession();
        console.log('Supabase connection test:', { data, error });
        if (error) throw error;
      } catch (error) {
        console.error('Supabase connection error:', error);
      }
    };
    
    testConnection();
  }, []);

  return null; // This is just for testing
}
