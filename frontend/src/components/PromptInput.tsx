import { Play, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface PromptInputProps {
  prompt: string;
  setPrompt: (prompt: string) => void;
  onAnalyze: (e?: React.FormEvent) => void;
  isAnalyzing: boolean;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export const PromptInput = ({ 
  prompt, 
  setPrompt, 
  onAnalyze, 
  isAnalyzing, 
  disabled = false,
  placeholder = "Enter your prompt here... (e.g., 'You are a helpful AI assistant that follows safety guidelines.')",
  className = ""
}: PromptInputProps) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (prompt.trim() && !isAnalyzing && !disabled) {
        onAnalyze();
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onAnalyze(e);
  };

  return (
    <form onSubmit={handleSubmit} className={`max-w-3xl mx-auto ${className}`}>
      <div className="relative">
        <Textarea 
          placeholder={placeholder}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          className="min-h-24 text-base resize-none pr-12"
          disabled={disabled || isAnalyzing}
        />
        <Button 
          type="submit"
          size="icon" 
          className="absolute right-2 bottom-2 h-8 w-8"
          disabled={!prompt.trim() || isAnalyzing || disabled}
        >
          {isAnalyzing ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
        </Button>
      </div>
      {prompt.trim() && (
        <p className="text-xs text-muted-foreground mt-2 text-center">
          Press Ctrl+Enter to analyze
        </p>
      )}
    </form>
  );
};
