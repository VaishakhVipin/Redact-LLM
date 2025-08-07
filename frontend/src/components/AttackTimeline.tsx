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
  const calculatePercentage = (blocked: number, total: number) => 
    Math.round((blocked / total) * 100);

  const categories = [
    { name: "Jailbreak", stats: vulnerabilityBreakdown.jailbreak },
    { name: "Hallucination", stats: vulnerabilityBreakdown.hallucination },
    { name: "Advanced", stats: vulnerabilityBreakdown.advanced }
  ];

  return (
    <Card className="bg-cream-50 border-cream-200 shadow-md">
      <CardHeader>
        <CardTitle className="font-serif text-2xl text-gray-800">Vulnerability Breakdown</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {categories.map(({ name, stats }) => (
            <div key={name} className="space-y-2">
              {/* Label Row */}
              <div className="flex justify-between items-center">
                <h4 className="font-medium text-gray-800">{name}</h4>
                <span className="font-bold text-green-600">
                  {stats.blocked}/{stats.total}
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full h-2 bg-gray-100 rounded-full">
                <div 
                  className="h-2 bg-green-600 rounded-full transition-all duration-300"
                  style={{ width: `${calculatePercentage(stats.blocked, stats.total)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
