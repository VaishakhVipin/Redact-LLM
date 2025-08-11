// Determine the correct API base URL based on environment
const getApiBaseUrl = () => {
  // Check if we're in production
  if (import.meta.env.PROD) {
    // Use your deployed backend URL - you'll need to update this with your actual backend deployment URL
    return 'https://your-backend-deployment-url.com/api/v1';
  }
  
  // For development, check if a custom API URL is provided
  const customApiUrl = import.meta.env.VITE_API_URL;
  if (customApiUrl) {
    return `${customApiUrl}/api/v1`;
  }
  
  // Default to localhost for development
  return 'http://localhost:8000/api/v1';
};

const API_BASE_URL = getApiBaseUrl();

export interface AttackGenerationRequest {
  prompt: string;
  attack_types: string[];
}

export interface AttackGenerationResponse {
  attacks: string[];
  count: number;
  categories: Record<string, number>;
}

export interface VulnerabilityDetail {
  total: number;
  blocked: number;
  vulnerable: number;
  score: number;
}

export interface AttackEvaluation {
  jailbreak_blocked: boolean;
  hallucination_blocked: boolean;
  advanced_blocked: boolean;
  reasoning: string;
  recommendations: {
    category: string;
    action: string;
    severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  }[];
}

export interface VulnerabilityBreakdown {
  total: number;
  blocked: number;
  vulnerable: number;
  score: number;
}

export interface ResistanceTestResponse {
  test_id: string;
  original_prompt: string;
  resistance_score: number;
  total_attacks: number;
  vulnerability_breakdown: {
    jailbreak: VulnerabilityBreakdown;
    hallucination: VulnerabilityBreakdown;
    advanced: VulnerabilityBreakdown;
  };
  recommendations: string[];
  attacks: string[];
}

export interface TestEvaluationResponse {
  breakdown: AttackEvaluation;
  debug: {
    evidence: {
      type: string;
      description: string;
      confidence: number;
    }[];
  };
  prompt: string;
  raw_result: string;
  test_id: string;
  original_prompt: string;
  resistance_score: number;
  total_attacks: number;
  vulnerabilityBreakdown: {
    jailbreak: AttackEvaluation;
    hallucination: AttackEvaluation;
    advanced: AttackEvaluation;
  };
  recommendations: string[];
  attacks: string[];
}

export interface StatsResponse {
  total_prompts: number;
  average_score: number;
  total_attacks_blocked: number;
  average_response_time: number;
}

export interface PipelineStatsResponse {
  pipeline_status: string;
  executor_worker: {
    stream_length: number;
    pending_tasks: number;
    last_processed_id: string;
  };
  evaluator: {
    processed_count: number;
    error_rate: number;
  };
}

export interface Verdict {
  original_prompt: string;
  resistance_score: number;
  attack_id?: string;
  prompt?: string;
  verdict?: string;
  risk_level?: string;
  alerts?: string[];
  timestamp?: string;
  evaluations?: Record<string, { detected: boolean; confidence: number }>;
}

export interface RecentVerdictsResponse {
  recent_verdicts: Verdict[];
  count: number;
}

import axios, { AxiosInstance, AxiosError } from 'axios';

class ApiService {
  private axiosInstance: AxiosInstance;

  constructor() {
    this.axiosInstance = axios.create({
      baseURL: API_BASE_URL,
      timeout: 300000, // 5 minutes
    });

    this.axiosInstance.interceptors.response.use(
      (response) => response,
      (error) => {
        if (axios.isAxiosError(error)) {
          if (!error.response || error.response.status === 429) {
            window.dispatchEvent(new CustomEvent('system-offline'));
          }
        }
        return Promise.reject(error);
      }
    );
  }

  private async request<T>(endpoint: string, options?: any): Promise<T> {
    try {
      const response = await this.axiosInstance.request<T>({
        url: endpoint,
        ...options,
      });
      return response.data;
    } catch (error) {
      if (error instanceof Error && error.message.includes('Failed to fetch')) {
        window.dispatchEvent(new CustomEvent('system-offline'));
      }
      console.error('API request error:', error);
      throw error;
    }
  }

  async generateAttacks(request: AttackGenerationRequest): Promise<AttackGenerationResponse> {
    return this.request<AttackGenerationResponse>('/attacks/generate', {
      method: 'POST',
      data: request,
    });
  }

  async testEvaluation(prompt: string, attack: string, response: string): Promise<TestEvaluationResponse> {
    return this.request<TestEvaluationResponse>('/attacks/test-evaluation', {
      method: 'POST',
      data: { prompt, attack, response },
    });
  }

  async testResistance(prompt: string): Promise<ResistanceTestResponse> {
    const response = await this.request<any>('/attacks/test-resistance', {
      method: 'POST',
      data: { prompt },
    });

    // Transform the response to match our frontend types
    const vulnerabilityBreakdown: {
      jailbreak: VulnerabilityBreakdown;
      hallucination: VulnerabilityBreakdown;
      advanced: VulnerabilityBreakdown;
    } = {
      jailbreak: {
        total: response.vulnerability_breakdown.jailbreak.total,
        blocked: response.vulnerability_breakdown.jailbreak.blocked,
        vulnerable: response.vulnerability_breakdown.jailbreak.vulnerable,
        score: response.vulnerability_breakdown.jailbreak.score,
      },
      hallucination: {
        total: response.vulnerability_breakdown.hallucination.total,
        blocked: response.vulnerability_breakdown.hallucination.blocked,
        vulnerable: response.vulnerability_breakdown.hallucination.vulnerable,
        score: response.vulnerability_breakdown.hallucination.score,
      },
      advanced: {
        total: response.vulnerability_breakdown.advanced.total,
        blocked: response.vulnerability_breakdown.advanced.blocked,
        vulnerable: response.vulnerability_breakdown.advanced.vulnerable,
        score: response.vulnerability_breakdown.advanced.score,
      },
    };

    return {
      ...response,
      vulnerability_breakdown: vulnerabilityBreakdown,
    };
  }

  async getStats(): Promise<StatsResponse> {
    return this.request<StatsResponse>('/attacks/stats');
  }

  async getPipelineStats(): Promise<PipelineStatsResponse> {
    return this.request<PipelineStatsResponse>('/attacks/pipeline/stats');
  }

  async getRecentVerdicts(limit = 10): Promise<RecentVerdictsResponse> {
    return this.request<RecentVerdictsResponse>(`/attacks/pipeline/verdicts?limit=${limit}`);
  }
}

export const apiService = new ApiService();