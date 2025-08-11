
*This is a submission for the [Redis AI Challenge](https://dev.to/challenges/redis-2025-07-23): Real-Time AI Innovators*.

## What I Built

**Redact-LLM** is an AI security testing platform that helps developers identify vulnerabilities in their AI systems. It generates adversarial prompts to test AI models for jailbreaks, hallucinations, and other security issues, then provides detailed analysis of the results.

The application features a React frontend for prompt submission and results visualization, and a FastAPI backend that orchestrates attack generation, execution, and evaluation workflows.

## Demo

Live demo: [https://redact-llm.vercel.app](https://redact-llm.vercel.app)

The platform allows users to:
- Submit prompts for security analysis
- View real-time attack generation and testing
- Analyze vulnerability breakdowns and security scores
- Browse historical test results

## How I Used Redis 8

Redis 8 serves as the backbone for several critical components in Redact-LLM:

### 1. **Semantic Caching System**
- Implements intelligent caching using sentence-transformers (all-MiniLM-L6-v2 model)
- Stores embeddings of prompts and responses to avoid redundant AI model calls
- Uses cosine similarity matching with configurable thresholds (default: 0.85)
- Organized with namespaced storage for different cache types

**Implementation:** `backend/app/services/semantic_cache.py`
```python
class SemanticCache:
    def __init__(self, redis_client, model_name='all-MiniLM-L6-v2', 
                 similarity_threshold=0.85, namespace="semantic_cache"):
        self.redis = redis_client
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
```

### 2. **Redis Streams for Job Processing**
- Uses Redis Streams to queue attack generation and execution jobs
- Enables asynchronous processing with consumer groups for scalability
- Processes prompts through multiple worker instances

**Key Components:**
- `backend/app/services/job_queue.py` - Job submission and status tracking
- `backend/app/workers/redteam_worker.py` - Stream consumer for processing
- `backend/app/services/executor_worker.py` - Attack execution worker

### 3. **Rate Limiting & Throttling**
- Multi-dimensional rate limiting (per-user, per-IP, global limits)
- Sliding window algorithm for accurate request counting
- Prevents abuse of expensive AI model calls

**Implementation:** `backend/app/services/rate_limiter.py`

### 4. **Response Caching**
- Caches attack execution results to avoid redundant model calls
- TTL-based expiration with stale cache fallback
- Significantly reduces response times for similar prompts

### 5. **Connection Management**
- Robust Redis connection handling with automatic reconnection
- Connection pooling for optimal performance
- Health checks and graceful degradation

**Key Features:**
- Intelligent cache invalidation
- Horizontal scaling support through Redis Streams
- Cost optimization through semantic deduplication
- Real-time job status tracking

The Redis integration enables Redact-LLM to handle high-throughput security testing while minimizing costs and latency through intelligent caching and efficient job processing.

---

*By submitting this entry, I agree to receive communications from Redis regarding products, services, events, and special offers. I can unsubscribe at any time. My information will be handled in accordance with [Redis's Privacy Policy](https://redis.io/legal/privacy-policy/).*
