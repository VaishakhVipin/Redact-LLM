import { useEffect, useState } from "react";
import { Shield, Home, Clock, BarChart3, Settings, User, Circle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { apiService, Verdict, StatsResponse } from "@/services/api";

export const Sidebar = () => {
  const [recentVerdicts, setRecentVerdicts] = useState<Verdict[]>([]);
  const [stats, setStats] = useState<StatsResponse>({ total_prompts: 0, average_score: 0 });
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    const handleOffline = () => setIsOnline(false);
    window.addEventListener('system-offline', handleOffline);
    return () => window.removeEventListener('system-offline', handleOffline);
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [verdicts, statsData] = await Promise.all([
          apiService.getRecentVerdicts(3),
          apiService.getStats()
        ]);
        
        setRecentVerdicts(verdicts?.recent_verdicts || []);
        if (statsData) setStats(statsData);
        setIsOnline(true);
      } catch (error) {
        console.error('Failed to fetch sidebar data:', error);
        setIsOnline(false);
      }
    };

    fetchData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} min ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)} hour ago`;
    return date.toLocaleDateString();
  };

  const getScoreFromRisk = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low': return 85;
      case 'medium': return 65;
      case 'high': return 35;
      default: return 50;
    }
  };

  const getScoreColor = (score: number) => {
    if (score > 80) return 'text-green-500';
    if (score > 50) return 'text-yellow-500';
    return 'text-red-500';
  };

  const navigation = [
    { name: "Dashboard", icon: Home, active: true },
    { name: "All Tests", icon: Clock, active: false },
    { name: "Analytics", icon: BarChart3, active: false },
    { name: "Settings", icon: Settings, active: false },
  ];

  return (
    <aside className="w-80 h-screen bg-sidebar border-r border-sidebar-border flex flex-col flex-shrink-0">
      {/* Branding */}
      <div className="p-6">
        <div className="flex items-center gap-2 mb-8">
          <Shield className="h-6 w-6 text-sidebar-primary" />
          <span className="font-serif text-xl font-semibold text-sidebar-foreground">Redact</span>
        </div>

        {/* Navigation */}
        <nav className="space-y-1">
          {navigation.map((item) => (
            <Button
              key={item.name}
              variant="ghost"
              className={`w-full justify-start ${
                item.active 
                  ? 'bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary hover:text-sidebar-primary-foreground' 
                  : 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'
              }`}
              size="sm"
            >
              <item.icon className="mr-2 h-4 w-4" />
              {item.name}
            </Button>
          ))}
        </nav>
      </div>

      <Separator className="bg-sidebar-border" />

      {/* Recent Prompts */}
      <div className="p-6 flex-1 space-y-6">
        <div>
          <h3 className="font-serif text-sm font-semibold text-sidebar-foreground mb-4">Recent Prompts</h3>
          <div className="space-y-2">
            {recentVerdicts.length > 0 ? (
              recentVerdicts.map((verdict, index) => (
                <div key={index} className="flex items-center justify-between text-sm p-2 rounded-lg hover:bg-muted/50">
                  <span className="font-mono truncate w-3/4" title={verdict.original_prompt}>{verdict.original_prompt}</span>
                  <span className={`font-semibold ${getScoreColor(verdict.resistance_score)}`}>{verdict.resistance_score}</span>
                </div>
              ))
            ) : (
              <div className="text-sm text-center text-muted-foreground py-4">No recent tests</div>
            )}
          </div>
        </div>

        {/* Enhanced Quick Stats */}
        <div>
          <h3 className="font-serif text-sm font-semibold text-sidebar-foreground mb-4">Quick Stats</h3>
          <div className="space-y-2">
            <Card className="p-3 bg-sidebar-accent border-sidebar-border">
              <div className="flex items-center justify-between">
                <span className="text-xs text-sidebar-foreground/70">Total Tests</span>
                <span className="text-sm font-mono font-medium text-sidebar-foreground">
                  {stats.total_prompts}
                </span>
              </div>
              <Progress value={75} className="mt-2 h-1" />
            </Card>
            
            <Card className="p-3 bg-sidebar-accent border-sidebar-border">
              <div className="flex items-center justify-between">
                <span className="text-xs text-sidebar-foreground/70">Avg. Score</span>
                <span className="text-sm font-mono font-medium text-sidebar-foreground">
                  {!isNaN(stats.average_score) ? `${Math.round(stats.average_score)}%` : '0%'}
                </span>
              </div>
            </Card>
            
            <Card className="p-3 bg-sidebar-accent border-sidebar-border">
              <div className="flex items-center justify-between">
                <span className="text-xs text-sidebar-foreground/70">System Status</span>
                <div className="flex items-center gap-1">
                  <Circle className={`h-2 w-2 ${isOnline ? 'fill-green-500 text-green-500' : 'fill-red-500 text-red-500'}`} />
                  <span className="text-xs font-mono text-sidebar-foreground capitalize">
                    {isOnline ? 'online' : 'offline'}
                  </span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>

      {/* Enhanced Profile Section */}
      <div className="p-6 border-t border-sidebar-border">
        <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-sidebar-accent transition-colors cursor-pointer">
          <Avatar className="h-8 w-8">
            <AvatarImage src="/placeholder.svg" alt="User" />
            <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground text-sm">SA</AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-sidebar-foreground truncate">Security Admin</p>
            <p className="text-xs text-sidebar-foreground/70 truncate">admin@company.com</p>
          </div>
        </div>
      </div>
    </aside>
  );
};