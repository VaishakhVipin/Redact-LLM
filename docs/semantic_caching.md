# Semantic Caching Implementation Plan

## Overview
This document outlines the implementation of semantic caching for the attack generator service using Redis and sentence embeddings.

## Architecture

### Components
1. **Embedding Model**
   - Using `all-MiniLM-L6-v2` from Sentence Transformers
   - Generates 384-dimensional embeddings
   - Lightweight and fast for production use

2. **Redis Data Structure**
   ```python
   Key: semantic_cache:<md5_hash_of_embedding>
   Value: {
       "attacks": List[str],  # Generated attack prompts
       "embedding": bytes     # Binary embedding vector
   }

### Cache Invalidation
TTL: 7 days
Size limit: 10,000 entries
LRU eviction policy

### Implementation Details
1. Similarity Search
Calculate cosine similarity between query and cached embeddings
Threshold: 0.85 similarity score
Returns cached results if similarity > threshold

2. Performance Considerations
Batch embedding generation
Async Redis operations
Connection pooling

3. Monitoring
Cache hit/miss metrics
Average similarity scores
Response time tracking

### API Changes
New Methods
_get_semantic_cache_key(prompt: str) -> Tuple[str, np.ndarray]
_find_similar_cached(prompt: str) -> Optional[List[str]]
_cache_semantic_result(prompt: str, attacks: List[str]) -> None

### Testing Plan
Unit Tests
Test exact match caching
Test semantic similarity matching
Test cache invalidation
Test concurrent access
Integration Tests
End-to-end attack generation
Cache hit/miss scenarios
Performance benchmarks

### Future Enhancements
Dynamic similarity threshold adjustment
Multi-vector search
Distributed caching
A/B testing for threshold optimization

### Dependencies
sentence-transformers>=2.2.2
numpy>=1.24.0
redis>=4.5.0
scikit-learn>=1.0.0 (for cosine similarity)

### Implementation Steps
[ ] Set up embedding model
[ ] Implement semantic caching methods
[ ] Integrate with attack generation
[ ] Add monitoring
[ ] Write tests
[ ] Deploy and monitorh