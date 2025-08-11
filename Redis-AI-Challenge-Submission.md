# Redact: Real-Time AI-Powered Prompt Security Platform
*This is a submission for the [Redis AI Challenge](https://dev.to/challenges/redis-2025-07-23): Real-Time AI Innovators*.

## What I Built

Redact is a cutting-edge security platform that protects AI systems from prompt injection attacks in real-time. Our solution acts as a secure middleware between users and LLM applications, analyzing and sanitizing prompts before they reach the target model.

Key Features:
- Real-time detection of prompt injection attempts
- Multi-layered defense against various attack vectors
- Semantic analysis of suspicious patterns
- Instant feedback and attack visualization
- Seamless integration with existing AI workflows

## Demo

[Insert video demo link here]

### Screenshots

1. **Dashboard Overview**
   ![Dashboard](https://via.placeholder.com/800x450.png?text=Redact+Dashboard)

2. **Attack Detection**
   ![Attack Detection](https://via.placeholder.com/800x450.png?text=Attack+Detection+View)

3. **Real-time Analytics**
   ![Analytics](https://via.placeholder.com/800x450.png?text=Real-time+Analytics)

## How I Used Redis 8

Redis 8 is at the core of Redact's architecture, providing several critical functions:

1. **Semantic Caching**
   - Implements a semantic cache using sentence-transformers (all-MiniLM-L6-v2 model)
   - Caches embeddings of processed prompts to avoid redundant computations
   - Uses cosine similarity with configurable threshold (default: 0.85) for cache lookups
   - Namespaced storage for different cache types (e.g., embeddings, responses)

2. **Rate Limiting & Throttling**
   - Implements multi-dimensional rate limiting (per-user, per-IP, global)
   - Configurable rate limits for different services (e.g., attack generation, API calls)
   - Sliding window algorithm for accurate request counting
   - Automatic cleanup of expired rate limit windows

3. **Stream Processing**
   - Processes prompt queue through Redis Streams
   - Enables asynchronous processing of security checks
   - Supports horizontal scaling of worker processes

4. **Connection Management**
   - Implements connection pooling with configurable limits
   - Automatic reconnection and health checks
   - Thread-safe Redis client management

## Technical Stack

- **Frontend**: React, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Python
- **AI/ML**: Sentence Transformers (all-MiniLM-L6-v2)
- **Caching & Messaging**: Redis 8 (Streams, Pub/Sub, Hashes, Sorted Sets)
- **Deployment**: Docker, Kubernetes
- **Monitoring**: Built-in rate limit tracking and metrics

## Future Enhancements

1. Implement Redis Search for more sophisticated querying of attack patterns
2. Add RedisTimeSeries for detailed metrics and analytics
3. Enhance semantic caching with adaptive similarity thresholds
4. Implement RedisAI for on-the-fly model inference
5. Add Redis Streams-based event sourcing for audit trails

---
*Team: [Your Name] (DEV: @yourusername)*

![Redact Logo](https://via.placeholder.com/200x50.png?text=Redact+Logo)

*By submitting this entry, I agree to receive communications from Redis regarding products, services, events, and special offers. I can unsubscribe at any time. My information will be handled in accordance with [Redis's Privacy Policy](https://redis.io/legal/privacy-policy/).*
