import { useLocation, useNavigate } from 'react-router-dom';
import { ResistanceTestResponse } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AttackTimeline } from "@/components/AttackTimeline";
import { Lightbulb, AlertTriangle } from 'lucide-react';

const AnalysisPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const analysisResult = location.state?.analysisResult as ResistanceTestResponse | null;

  if (!analysisResult) {
    return (
      <div className="flex-1 flex flex-col justify-center items-center h-screen bg-cream-50">
        <h1 className="text-2xl font-bold text-stone-800">No analysis data found.</h1>
        <p className="text-stone-600 mt-2">Please go back and submit a prompt for analysis.</p>
        <Button onClick={() => navigate('/')} className="mt-4">Go Home</Button>
      </div>
    );
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600";
    if (score >= 50) return "text-yellow-600";
    return "text-red-600";
  };

  const getScoreBadgeVariant = (score: number): "default" | "secondary" | "destructive" => {
    if (score >= 80) return "default";
    if (score >= 50) return "secondary";
    return "destructive";
  };

  return (
    <main className="flex-1 p-6 md:p-10 overflow-auto bg-cream-50">
      <div className="max-w-7xl mx-auto w-full flex-1 flex flex-col">
        <div className="space-y-8 w-full animate-fade-in">
          <div className="text-center">
            <p className="text-gray-600 font-semibold text-lg">Overall Resistance Score</p>
            <h2 className={`text-8xl font-bold ${getScoreColor(analysisResult.resistance_score)}`}>
              {analysisResult.resistance_score}
            </h2>
            <Badge variant={getScoreBadgeVariant(analysisResult.resistance_score)} className="mt-2 text-md py-1 px-4 shadow-sm">
              {analysisResult.resistance_score >= 80 ? "Strong Defense" : analysisResult.resistance_score >= 50 ? "Moderate Defense" : "Weak Defense"}
            </Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Card className="bg-cream-100 border-cream-200 shadow-md">
              <CardHeader>
                <CardTitle className="font-serif text-2xl text-gray-800">Original Prompt</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 italic text-md leading-relaxed">"{analysisResult.original_prompt}"</p>
              </CardContent>
            </Card>
            <AttackTimeline testId="" attacks={[]} vulnerabilityBreakdown={analysisResult.vulnerability_breakdown} />
          </div>

          <Card className="bg-cream-100 border-cream-200 shadow-md">
            <CardHeader>
              <CardTitle className="font-serif text-2xl text-gray-800">Security Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {analysisResult.recommendations.map((rec: string, index: number) => (
                  <Alert key={index} className="bg-blue-50 border-blue-200 p-4 rounded-lg shadow-sm">
                    <Lightbulb className="h-5 w-5 text-blue-500 flex-shrink-0" />
                    <AlertDescription className="text-blue-900 text-md ml-4">
                      {rec}
                    </AlertDescription>
                  </Alert>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-cream-100 border-cream-200 shadow-md">
            <CardHeader>
              <CardTitle className="font-serif text-2xl text-gray-800">Sampled Attacks</CardTitle>
              <CardDescription className="text-gray-600">A sample of attacks used to test your prompt's resistance.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {analysisResult.attacks.slice(0, 5).map((attack: string, index: number) => (
                  <Alert key={index} variant="destructive" className="bg-red-50 border-red-200 p-4 rounded-lg shadow-sm">
                    <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0" />
                    <AlertDescription className="text-red-900 font-mono text-sm ml-4">
                      {attack}
                    </AlertDescription>
                  </Alert>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="text-center py-6">
            <Button size="lg" variant="outline" onClick={() => navigate('/')} className="text-lg px-8 py-6 shadow-sm hover:bg-cream-200">
              Test Another Prompt
            </Button>
          </div>
        </div>
      </div>
    </main>
  );
};

export default AnalysisPage;
