import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils"

interface AttackTimelineProps {
  testId: string;
  attacks: string[];
  vulnerabilityBreakdown: {
    jailbreak: {
      total: number;
      blocked: number;
      vulnerable: number;
      score: number;
    };
    hallucination: {
      total: number;
      blocked: number;
      vulnerable: number;
      score: number;
    };
    advanced: {
      total: number;
      blocked: number;
      vulnerable: number;
      score: number;
    };
  };
}

export function AttackTimeline({ testId, attacks, vulnerabilityBreakdown }: AttackTimelineProps) {
  const calculatePercentage = (blocked: number, total: number) => {
    if (total === 0) return 0;
    return Math.round((blocked / total) * 100);
  };

  const categories = [
    { 
      name: "Jailbreak", 
      key: "jailbreak",
      stats: vulnerabilityBreakdown.jailbreak 
    },
    { 
      name: "Hallucination", 
      key: "hallucination",
      stats: vulnerabilityBreakdown.hallucination 
    },
    { 
      name: "Advanced", 
      key: "advanced",
      stats: vulnerabilityBreakdown.advanced 
    }
  ];

  // If all categories have 0 total, show a message
  const allZero = categories.every(cat => cat.stats.total === 0);
  
  if (allZero) {
    return (
      <Card className="bg-cream-50 border-cream-200 shadow-md">
        <CardHeader>
          <CardTitle className="font-serif text-2xl text-gray-800">Vulnerability Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600 italic">No vulnerability data available. Run tests to see detailed breakdown.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-cream-50 border-cream-200 shadow-md">
      <CardHeader>
        <CardTitle className="font-serif text-2xl text-gray-800">Vulnerability Breakdown</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-8">
          {categories.map(({ name, key, stats }, index) => {
            const percentage = calculatePercentage(stats.blocked, stats.total);
            const percentageText = `${percentage}%`;
            const isLast = index === categories.length - 1;
            const showPercentageLabels = isLast; // Only show on last item
            
            return (
              <div key={key} className="space-y-2">
                <div className="flex justify-between items-center">
                  <h4 className="font-medium text-gray-800">{name}</h4>
                  <div className="flex items-center gap-2">
                    <span className={`font-bold ${
                      percentage >= 80 ? 'text-green-600' : 
                      percentage >= 50 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {stats.blocked}/{stats.total} blocked
                    </span>
                    <span className="text-gray-500 text-sm">
                      ({percentageText})
                    </span>
                  </div>
                </div>
                
                <div className="w-full h-3 bg-gray-100 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-500 ${
                      percentage >= 80 ? 'bg-green-500' : 
                      percentage >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.max(5, percentage)}%` }}
                    aria-valuenow={percentage}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                </div>
                
                {showPercentageLabels && (
                  <div className="flex justify-between text-sm text-gray-500 mt-1">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
