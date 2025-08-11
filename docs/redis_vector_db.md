# Redis Vector DB Integration

This document outlines how Redis is used as a vector database in the Redact-LLM project, specifically for semantic caching and attack pattern storage.

## Overview

Redis is used in this project for:
1. **Semantic Caching**: Storing and retrieving semantically similar prompts and their corresponding attacks
2. **Attack Pattern Storage**: Persisting known attack patterns for quick retrieval
3. **Rate Limiting**: Managing API request rates
4. **Temporary Storage**: Storing intermediate results during attack generation

## Key Components

### 1. Semantic Cache

#### Purpose
- Stores embeddings of prompts and their corresponding attacks
- Enables efficient similarity search to find semantically similar cached results

#### Implementation
```python
class SemanticCache:
    def __init__(
        self, 
        redis_client: redis.Redis,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.85,
        namespace: str = "semantic_cache"
    ):
        self.redis = redis_client
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.namespace = namespace
```

#### Key Methods
- `get_similar`: Find similar cached entries
- `set`: Add new entries to the cache
- `_get_embedding`: Generate embeddings for text

### 2. Redis Connection Management

#### Configuration
- Connection is established using environment variables
- Connection pooling is enabled for better performance
- Automatic reconnection is configured

#### Implementation
```python
async def get_redis_client() -> Optional[redis.Redis]:
    try:
        return redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            password=os.getenv('REDIS_PASSWORD', None),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None
```

### 3. Vector Search

#### How It Works
1. Text is converted to embeddings using a pre-trained model
2. Embeddings are stored in Redis with their corresponding metadata
3. Similarity search is performed using Redis' vector search capabilities

#### Example Usage
```python
# Store a new embedding
embedding = model.encode("example prompt")
metadata = {"response": "example response", "type": "example"}
redis_client.hset("embedding:1", mapping={
    "embedding": embedding.tobytes(),
    "metadata": json.dumps(metadata)
})

# Search for similar embeddings
query_embedding = model.encode("similar prompt")
results = redis_client.ft("idx:embeddings").search(
    f"*=>[KNN 5 @embedding $vec as score]",
    {"vec": query_embedding.tobytes()}
)
```

## Performance Considerations

1. **Embedding Model**: Using 'all-MiniLM-L6-v2' as it provides a good balance between speed and accuracy
2. **Batch Processing**: Process embeddings in batches when possible
3. **Connection Pooling**: Reuse Redis connections to reduce overhead
4. **Cache Invalidation**: Implement TTL for cached entries

## Error Handling

- Connection errors are caught and logged
- Fallback mechanisms are in place when Redis is unavailable
- Circuit breakers prevent cascading failures

## Monitoring

Key metrics to monitor:
1. Redis memory usage
2. Cache hit/miss ratio
3. Query latency
4. Error rates

## Best Practices

1. **Indexing**: Create appropriate indexes for vector search
2. **Memory Management**: Monitor and configure maxmemory settings
3. **Security**: Use authentication and TLS where applicable
4. **Backup**: Regular backups of the Redis database

## Troubleshooting

### Common Issues
1. **Connection Timeouts**:
   - Check network connectivity
   - Verify Redis server is running
   - Adjust timeout settings if needed

2. **Memory Issues**:
   - Monitor memory usage
   - Set appropriate maxmemory policy
   - Consider sharding for large datasets

3. **Performance Problems**:
   - Check for slow queries
   - Optimize index settings
   - Consider scaling Redis instances

## Future Improvements

1. Implement Redis Cluster for horizontal scaling
2. Add support for more embedding models
3. Enhance monitoring and alerting
4. Implement cache warming for frequently accessed items

## Dependencies

- `redis`: Python Redis client
- `sentence-transformers`: For generating text embeddings
- `numpy`: For numerical operations on embeddings
