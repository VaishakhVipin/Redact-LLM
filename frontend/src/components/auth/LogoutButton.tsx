import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';
import { LogOut } from 'lucide-react';

export function LogoutButton() {
  const { logout } = useAuth();
  
  return (
    <Button 
      variant="ghost" 
      onClick={logout}
      className="flex items-center gap-2"
    >
      <LogOut className="h-4 w-4" />
      Sign out
    </Button>
  );
}
