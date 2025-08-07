import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { AttackTimeline } from "../components/AttackTimeline";
import { apiService } from "@/services/api";
import { Badge } from "@/components/ui/badge";
import { ResistanceTestResponse } from "@/services/api";

export default function PromptAnalysisPage() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const [testId, setTestId] = useState("");
  const [isTesting, setIsTesting] = useState(false);
  const [testResults, setTestResults] = useState<ResistanceTestResponse | null>(null);

  const handleTestPrompt = async () => {
    if (!prompt.trim()) return;

    setIsTesting(true);
    try {
      const result = await apiService.testResistance(prompt);
      setTestId(result.test_id);
      setTestResults(result);
    } catch (error) {
      console.error("Error testing prompt:", error);
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <div className="container mx-auto py-8">
      <div className="max-w-4xl mx-auto">
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold">Prompt Analysis</h1>
            <p className="text-muted-foreground mt-2">
              Test your prompt's resistance against various attack vectors
            </p>
          </div>

          <Card className="p-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Enter Prompt to Test
                </label>
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your prompt here..."
                  className="min-h-[100px]"
                />
                <Button onClick={handleTestPrompt} disabled={isTesting}>
                  {isTesting ? "Testing..." : "Test Prompt"}
                </Button>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold">Analysis Results</h2>
                <div className="flex gap-2">
                  <Badge variant="outline">Jailbreak</Badge>
                  <Badge variant="outline">Hallucination</Badge>
                  <Badge variant="outline">Advanced</Badge>
                </div>
              </div>
              <div className="space-y-4">
                {testId ? (
                  <div>
                    <p className="text-sm text-muted-foreground">
                      Test ID: {testId}
                    </p>
                    <div className="mt-4">
                      {testResults && (
                        <AttackTimeline
                          testId={testId}
                          attacks={testResults.attacks}
                          vulnerabilityBreakdown={testResults.vulnerability_breakdown}
                        />
                      )}
                    </div>
                  </div>
                ) : (
                  <p className="text-muted-foreground">
                    Run a test to see analysis results
                  </p>
                )}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
