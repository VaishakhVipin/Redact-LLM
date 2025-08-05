import { useState } from "react";
import { Sidebar } from "./Sidebar";
import { MainContent } from "./MainContent";

export const Dashboard = () => {
  const [prompt, setPrompt] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  return (
    <div className="min-h-screen w-full bg-background">
      <Sidebar />
      <div className="ml-80 flex-1">
        <MainContent
          prompt={prompt}
          setPrompt={setPrompt}
          isAnalyzing={isAnalyzing}
          setIsAnalyzing={setIsAnalyzing}
          analysisResult={analysisResult}
          setAnalysisResult={setAnalysisResult}
        />
      </div>
    </div>
  );
};