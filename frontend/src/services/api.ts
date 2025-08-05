const API_BASE_URL = 'http://localhost:8000/api/v1';

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

export interface ResistanceTestResponse {
  test_id: string;
  original_prompt: string;
  resistance_score: number;
  total_attacks: number;
  vulnerability_breakdown: { [key: string]: VulnerabilityDetail };
  recommendations: string[];
  attacks: string[];
}

export interface StatsResponse {
  total_prompts: number;
  average_score: number;
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
      baseURL: 'http://localhost:8000/api/v1',
      timeout: 60000, // 60 seconds
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

  async testResistance(prompt: string): Promise<ResistanceTestResponse> {
    return this.request<ResistanceTestResponse>('/attacks/test-resistance', {
      method: 'POST',
      data: { prompt },
    });
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