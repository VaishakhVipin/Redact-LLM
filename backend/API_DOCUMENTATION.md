# Redis-Driven LLM Ops Pipeline - API Documentation

## Overview

This document provides comprehensive API documentation for the Redis-driven LLM Ops Pipeline backend. The system is designed for real-time attack generation, processing, and evaluation with Redis as the orchestration layer.

**Base URL**: `http://localhost:8000`
**API Version**: `v1`
**Content-Type**: `application/json`

## Authentication & Rate Limiting

### Rate Limits
All endpoints are protected by Redis-based rate limiting:

- **Attack Generation**: 50 requests/hour per user, 100 requests/hour per IP
- **API Calls**: 200 requests/hour per user, 500 requests/hour per IP
- **Global Limits**: 1000 requests/hour for attack generation, 5000 requests/hour for API calls

### Rate Limit Headers
When rate limits are exceeded, the API returns:
- **Status Code**: `429 Too Many Requests`
- **Headers**: 
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Time until reset (Unix timestamp)

### User Identification
Since no authentication system is implemented, user identification is based on:
- **IP Address**: Automatically detected from request
- **User ID**: Derived from prompt content (first 20 characters)

## Core Endpoints

### 1. Generate Attack Prompts

**Endpoint**: `POST /api/v1/attacks/generate`

**Description**: Generate adversarial attack prompts for a given input prompt using Gemini API.

**Request Body**:
```json
{
  "prompt": "string (required)",
  "attack_types": ["jailbreak", "hallucination", "advanced"] (optional, default: all)
}
```

**Response**:
```json
{
  "prompt": "string",
  "attacks": ["string"],
  "count": "integer",
  "categories": {
    "jailbreak": "integer",
    "hallucination": "integer", 
    "advanced": "integer"
  }
}
```

**Status Codes**:
- `200`: Success
- `429`: Rate limit exceeded
- `500`: Server error

**Rate Limit**: 50 requests/hour per user

### 2. Test Prompt Resistance

**Endpoint**: `POST /api/v1/attacks/test-resistance`

**Description**: Test prompt resistance and return detailed analysis.

**Request Body**:
```json
{
  "prompt": "string (required)",
  "target_response": "string (optional)"
}
```

**Response**:
```json
{
  "original_prompt": "string",
  "target_response": "string",
  "total_attacks": "integer",
  "attack_categories": {
    "jailbreak": "integer",
    "hallucination": "integer",
    "advanced": "integer"
  },
  "attacks": ["string"],
  "resistance_score": "integer",
  "recommendations": ["string"]
}
```

**Status Codes**:
- `200`: Success
- `429`: Rate limit exceeded
- `500`: Server error

### 3. Comprehensive Statistics

**Endpoint**: `GET /api/v1/attacks/stats`

**Description**: Get comprehensive system statistics including Redis features, metrics, and rate limiting data.

**Response**:
```json
{
  "attack_generator": {
    "attack_types": {
      "jailbreak": "string",
      "hallucination": "string",
      "advanced": "string"
    },
    "max_attacks_per_type": {
      "jailbreak": "integer",
      "hallucination": "integer",
      "advanced": "integer"
    },
    "total_max_attacks": "integer",
    "cache_stats": {
      "total_cached_attacks": "integer",
      "cache_hit_rate": "float",
      "current_rate_limit": "integer"
    },
    "rate_limit": {
      "max_requests_per_minute": "integer",
      "current_requests": "integer"
    },
    "stream_stats": {
      "attack_stream_length": "integer",
      "stream_name": "string"
    }
  },
  "comprehensive_metrics": {
    "total_prompts": "integer",
    "total_attacks": "integer",
    "total_verdicts": "integer",
    "total_api_calls": "integer",
    "attack_type_distribution": {
      "jailbreak": "integer",
      "hallucination": "integer",
      "advanced": "integer"
    },
    "risk_level_distribution": {
      "low": "integer",
      "medium": "integer",
      "high": "integer"
    },
    "jailbreaks_caught": "integer",
    "hallucinations_caught": "integer",
    "safety_concerns": "integer",
    "tone_mismatches": "integer",
    "average_robustness_score": "float",
    "cache_performance": {
      "attack_cache": "integer",
      "response_cache": "integer",
      "verdict_cache": "integer"
    },
    "endpoint_usage": {
      "/generate": "integer",
      "/test-resistance": "integer",
      "/stats": "integer",
      "/pipeline/stats": "integer",
      "/pipeline/verdicts": "integer"
    },
    "average_response_times": {
      "/generate": "float",
      "/test-resistance": "float",
      "/stats": "float"
    },
    "status_code_distribution": {
      "200": "integer",
      "400": "integer",
      "500": "integer"
    },
    "heatmap_data": {
      "jailbreak": {
        "success": "integer",
        "total": "integer",
        "success_rate": "float"
      },
      "hallucination": {
        "success": "integer",
        "total": "integer",
        "success_rate": "float"
      },
      "safety": {
        "success": "integer",
        "total": "integer",
        "success_rate": "float"
      },
      "tone_match": {
        "success": "integer",
        "total": "integer",
        "success_rate": "float"
      }
    },
    "system_health": {
      "redis_connected": "boolean",
      "last_updated": "string (ISO 8601)",
      "uptime_seconds": "float"
    }
  },
  "rate_limiting": {
    "attack_generation": {
      "per_user": {
        "total_requests": "integer",
        "limit": "integer",
        "window_seconds": "integer",
        "utilization_percent": "float"
      },
      "per_ip": {
        "total_requests": "integer",
        "limit": "integer",
        "window_seconds": "integer",
        "utilization_percent": "float"
      },
      "global": {
        "total_requests": "integer",
        "limit": "integer",
        "window_seconds": "integer",
        "utilization_percent": "float"
      }
    },
    "api_calls": {
      "per_user": {
        "total_requests": "integer",
        "limit": "integer",
        "window_seconds": "integer",
        "utilization_percent": "float"
      },
      "per_ip": {
        "total_requests": "integer",
        "limit": "integer",
        "window_seconds": "integer",
        "utilization_percent": "float"
      },
      "global": {
        "total_requests": "integer",
        "limit": "integer",
        "window_seconds": "integer",
        "utilization_percent": "float"
      }
    }
  },
  "redis_features_used": {
    "streams": "attack_stream for real-time processing",
    "pub_sub": "verdict_channel for live updates",
    "cache": "attack_response_cache for deduplication",
    "rate_limiting": "per-user and per-IP protection",
    "metrics": "real-time statistics collection"
  }
}
```

**Status Codes**:
- `200`: Success
- `429`: Rate limit exceeded
- `500`: Server error

## Pipeline Endpoints

### 4. Pipeline Statistics

**Endpoint**: `GET /api/v1/attacks/pipeline/stats`

**Description**: Get real-time pipeline statistics including executor worker and evaluator status.

**Response**:
```json
{
  "pipeline_status": "string",
  "executor_worker": {
    "stream_length": "integer",
    "pending_messages": "integer",
    "processed_results": "integer",
    "worker_status": "string",
    "consumer_group": "string"
  },
  "evaluator": {
    "total_results": "integer",
    "total_verdicts": "integer",
    "evaluated_count": "integer",
    "pending_evaluations": "integer",
    "evaluator_status": "string"
  },
  "timestamp": "float"
}
```

### 5. Recent Verdicts

**Endpoint**: `GET /api/v1/attacks/pipeline/verdicts?limit=10`

**Description**: Get recent verdicts from the evaluator.

**Query Parameters**:
- `limit`: Number of verdicts to return (default: 10, max: 50)

**Response**:
```json
{
  "recent_verdicts": [
    {
      "attack_id": "string",
      "attack_type": "string",
      "attack_subtype": "string",
      "evaluations": {
        "jailbreak": {
          "jailbreak_detected": "boolean",
          "jailbreak_risk": "float",
          "safety_score": "integer",
          "compliance_score": "integer",
          "reasoning": "string"
        },
        "hallucination": {
          "hallucination_detected": "boolean",
          "hallucination_risk": "float",
          "attack_hallucination_score": "integer",
          "response_hallucination_score": "integer",
          "pattern_matches": "integer",
          "reasoning": "string"
        },
        "safety": {
          "safety_concern": "boolean",
          "safety_risk": "float",
          "attack_safety_score": "integer",
          "response_safety_score": "integer",
          "reasoning": "string"
        },
        "tone_match": {
          "tone_mismatch": "boolean",
          "tone_mismatch_score": "float",
          "attack_serious_score": "integer",
          "attack_joking_score": "integer",
          "response_serious_score": "integer",
          "response_joking_score": "integer",
          "reasoning": "string"
        }
      },
      "overall_risk": "float",
      "risk_level": "string (low|medium|high)",
      "alerts": ["string"],
      "evaluated_at": "string (ISO 8601)"
    }
  ],
  "count": "integer"
}
```

### 6. Attack Results

**Endpoint**: `GET /api/v1/attacks/pipeline/results?limit=10`

**Description**: Get recent attack results from the executor worker.

**Query Parameters**:
- `limit`: Number of results to return (default: 10, max: 50)

**Response**:
```json
{
  "recent_results": [
    {
      "attack_id": "string",
      "base_prompt": "string",
      "attack_variant": "string",
      "attack_type": "string",
      "attack_subtype": "string",
      "response": {
        "model": "string",
        "response": "string",
        "timestamp": "string (ISO 8601)",
        "model_used": "string"
      },
      "processed_at": "string (ISO 8601)",
      "cache_hit": "boolean"
    }
  ],
  "count": "integer"
}
```

## Job Queue Endpoints

### 7. Submit Background Job

**Endpoint**: `POST /api/v1/attacks/submit-job`

**Description**: Submit a job for background attack generation.

**Request Body**:
```json
{
  "prompt": "string (required)",
  "attack_types": ["jailbreak", "hallucination", "advanced"] (optional),
  "priority": "integer (optional, default: 1)"
}
```

**Response**:
```json
{
  "job_id": "string",
  "status": "submitted"
}
```

### 8. Get Job Status

**Endpoint**: `GET /api/v1/attacks/job/{job_id}`

**Description**: Get job status and results.

**Response**:
```json
{
  "job_id": "string",
  "status": "string (pending|completed|failed)",
  "result": {
    // Job result data if completed
  },
  "error": "string (if failed)"
}
```

### 9. Queue Statistics

**Endpoint**: `GET /api/v1/attacks/queue/stats`

**Description**: Get job queue statistics.

**Response**:
```json
{
  "total_jobs": "integer",
  "pending_jobs": "integer",
  "completed_jobs": "integer",
  "queue_health": "string (healthy|warning)"
}
```

### 10. Pending Jobs

**Endpoint**: `GET /api/v1/attacks/queue/pending?limit=10`

**Description**: Get list of pending jobs.

**Query Parameters**:
- `limit`: Number of jobs to return (default: 10, max: 100)

**Response**:
```json
{
  "pending_jobs": [
    {
      "job_id": "string",
      "prompt": "string",
      "created_at": "string (ISO 8601)",
      "priority": "integer"
    }
  ],
  "count": "integer"
}
```

## Health & Monitoring

### 11. Health Check

**Endpoint**: `GET /api/v1/attacks/health`

**Description**: Health check for attack generation service.

**Response**:
```json
{
  "status": "healthy",
  "service": "attack_generator",
  "test_attacks_generated": "integer"
}
```

### 12. Root Endpoint

**Endpoint**: `GET /`

**Description**: Basic API information.

**Response**:
```json
{
  "message": "Redis-Driven LLM Ops Pipeline API",
  "version": "v1",
  "status": "running"
}
```

## Error Responses

### Standard Error Format
```json
{
  "detail": "Error message description"
}
```

### Common Error Codes
- `400 Bad Request`: Invalid request body or parameters
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Redis Data Structures

### Streams
- **attack_stream**: Real-time attack queue
- **attack_generation_jobs**: Background job queue

### Pub/Sub Channels
- **verdict_channel**: Real-time verdict updates

### Cache Keys
- **attack_cache:{type}:{hash}**: Cached attack results
- **attack_response:{hash}**: Cached model responses
- **verdict:{attack_id}**: Cached verdicts
- **rate_limit:{service}_{type}:{id}:{window}**: Rate limiting counters
- **metrics:{metric_name}**: System metrics

## Frontend Integration Guidelines

### Real-time Updates
To receive real-time verdict updates, implement WebSocket connection to Redis pub/sub:
- Subscribe to `verdict_channel`
- Parse JSON messages for live updates

### Rate Limiting Handling
- Monitor `429` responses
- Implement exponential backoff
- Show user-friendly rate limit messages
- Display remaining requests count

### Error Handling
- Implement retry logic for `500` errors
- Handle network timeouts gracefully
- Provide fallback UI for service unavailability

### Performance Optimization
- Cache frequently accessed data
- Implement request debouncing
- Use pagination for large result sets
- Monitor response times from `/stats` endpoint

## Environment Variables

The backend requires these environment variables:
```
REDIS_HOST=redis-15632.c330.asia-south1-1.gce.redns.redis-cloud.com
REDIS_PORT=15632
REDIS_USERNAME=default
REDIS_PASSWORD=your_redis_password
GEMINI_API_KEY=your_gemini_api_key
```

## Testing

Use the provided test scripts:
- `test_final_system.py`: Comprehensive system testing
- `test_redis_pipeline.py`: Redis pipeline testing
- `test_optimized_system.py`: Optimization testing

## Security Considerations

1. **Rate Limiting**: All endpoints are rate-limited to prevent abuse
2. **Input Validation**: All inputs are validated and sanitized
3. **Error Handling**: Sensitive information is not exposed in error messages
4. **Redis Security**: SSL is disabled for Redis Cloud compatibility
5. **API Security**: No authentication required, but rate limiting provides protection

## Performance Metrics

Monitor these key metrics:
- **Average Response Time**: From `/stats` endpoint
- **Cache Hit Rate**: Attack and response cache performance
- **Rate Limit Utilization**: Current usage vs limits
- **Pipeline Health**: Executor worker and evaluator status
- **Redis Connection**: System health status 