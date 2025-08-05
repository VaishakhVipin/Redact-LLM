# Redis Integration in the Redact-LLM Backend

This document provides a detailed explanation of how Redis is used in the Redact-LLM project. Redis serves as a critical component for caching, real-time communication, and rate limiting, significantly enhancing the performance, scalability, and responsiveness of the application.

## 1. Core Use Cases

### a. Caching

**Purpose:** To reduce latency and minimize redundant, expensive API calls to the generative AI models (e.g., Google Gemini).

**Implementation:**
- The `AttackGenerator` service in `app/services/attack_generator.py` uses Redis to cache the results of generated attack prompts.
- Before generating new attacks for a given prompt, the system first checks if a cached result already exists.
- A unique cache key is generated for each prompt and attack type (e.g., `jailbreak`, `hallucination`, `advanced`).
- If a cached entry is found, it is served directly, bypassing the need to call the external AI model. This dramatically speeds up repeated analyses of the same prompt.
- Cached attacks are stored as lists in Redis with a specific Time-To-Live (TTL) to ensure the cache remains fresh.

**Key Functions:**
- `_generate_cache_key()`: Creates a consistent key for caching.
- `_get_cached_attacks()`: Retrieves attack lists from the cache.
- `_cache_attacks()`: Stores newly generated attacks in the cache.

### b. Real-Time Communication (Pub/Sub)

**Purpose:** To provide real-time feedback to the frontend during the analysis process.

**Implementation:**
- The backend uses Redis Pub/Sub to stream events and data from the backend worker to the frontend client.
- When an analysis begins, the system publishes messages to a specific Redis channel associated with the unique `test_id`.
- A separate worker process (`worker.py`) subscribes to these channels and forwards the messages to the appropriate client via Server-Sent Events (SSE).
- This allows the frontend to display progress, individual attack results, and other real-time updates as they happen, rather than waiting for the entire analysis to complete.

**Key Components:**
- **`_push_attack_to_stream()`**: In `attack_generator.py`, this function publishes individual attack details to the Redis stream (channel).
- **`worker.py`**: This standalone script listens to Redis channels and streams data to the frontend.
- **`main.py`**: The main FastAPI application includes an SSE endpoint (`/stream/{test_id}`) that the frontend connects to.

### c. Rate Limiting

**Purpose:** To prevent abuse of the AI model endpoints and manage costs.

**Implementation:**
- A simple rate-limiting mechanism is implemented within the `AttackGenerator`.
- Before making a call to the generative AI, the system checks a counter in Redis associated with a user or a general-purpose key.
- If the number of requests within a specific time window exceeds the defined limit, the system falls back to a set of pre-defined, less resource-intensive attacks instead of calling the external API.

**Key Function:**
- `_check_rate_limit()`: Implements the logic to check and increment the request counter in Redis.

## 2. Setup and Configuration

- **Connection:** The Redis client is initialized in `app/services/attack_generator.py` and `worker.py`. Connection details (host, port, etc.) are typically managed through environment variables (as seen in the `.env` file).
- **Dependencies:** The `redis` Python library is a key dependency, listed in `requirements.txt`.

## 3. Redis Data Structures Used

- **Strings:** Used for simple counters in the rate-limiting mechanism.
- **Lists:** Used to store the cached attack prompts for each category.
- **Pub/Sub Channels:** Used as the backbone for real-time event streaming between the backend and frontend.

## Summary

Redis is integral to the Redact-LLM architecture. It acts as a high-speed data store and messaging broker that decouples components, improves performance, and enables a more dynamic and responsive user experience. Its roles in caching, real-time updates, and rate limiting are essential for the application's efficiency and robustness.
