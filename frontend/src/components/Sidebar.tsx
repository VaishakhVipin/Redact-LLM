import { Shield, LogOut } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";

export const Sidebar = () => {
  const { user, logout } = useAuth();

  return (
    <div className="fixed left-0 top-0 h-full w-64 border-r bg-[var(--sidebar)] flex flex-col z-10">
      {/* Branding */}
      <div className="p-4">
        <div className="flex items-center gap-2">
          <Shield className="h-6 w-6 text-sidebar-primary" />
          <span className="font-serif text-xl font-semibold text-sidebar-foreground">Redact</span>
        </div>
      </div>

      <Separator className="bg-sidebar-border" />

      {/* Spacer to push content to center */}
      <div className="flex-1" />

      {/* Profile Section */}
      <div className="p-4 border-t border-sidebar-border mt-auto">
        {user ? (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Avatar className="h-10 w-10">
                <AvatarImage src={user.user_metadata?.avatar_url} alt={user.email || 'User'} />
                <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground">
                  {user.email?.charAt(0).toUpperCase() || 'U'}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-sidebar-foreground truncate">
                  {user.user_metadata?.full_name || user.email?.split('@')[0] || 'User'}
                </p>
                <p className="text-xs text-sidebar-foreground/70 truncate">{user.email}</p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start text-xs text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent"
              onClick={() => {
                logout().catch(error => {
                  console.error('Error signing out:', error);
                  toast.error('Failed to sign out. Please try again.');
                });
              }}
            >
              <LogOut className="mr-2 h-3.5 w-3.5" />
              Sign out
            </Button>
          </div>
        ) : (
          <div className="text-center text-sm text-sidebar-foreground/70">
            Not signed in
          </div>
        )}
      </div>
    </div>
  );
};