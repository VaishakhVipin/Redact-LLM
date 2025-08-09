from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from numpy.linalg import norm
import json
import logging
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    A semantic cache that stores and retrieves items based on semantic similarity.
    Uses sentence-transformers for embeddings and Redis for storage.
    """
    
    def __init__(self, redis_client: redis.Redis, model_name: str = 'all-MiniLM-L6-v2', 
                 similarity_threshold: float = 0.85, namespace: str = "semantic_cache"):
        """
        Initialize the semantic cache.
        
        Args:
            redis_client: Redis client instance
            model_name: Name of the sentence-transformer model to use
            similarity_threshold: Minimum cosine similarity score (0-1) to consider items a match
            namespace: Redis key namespace for this cache
        """
        self.redis = redis_client
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.namespace = namespace
        self.embedding_size = 384  # Default for all-MiniLM-L6-v2
        
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        try:
            # Check if we have a cached embedding
            cache_key = f"{self.namespace}:embeddings:{hash(text) & 0xffffffff}"
            cached = await self.redis.get(cache_key)
            
            if cached:
                return json.loads(cached)
                
            # Generate new embedding
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            
            # Cache the embedding (no TTL for embeddings as they're deterministic)
            await self.redis.set(cache_key, json.dumps(embedding))
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (norm(a) * norm(b))
    
    async def find_similar(self, text: str, namespace: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar items in the cache based on semantic similarity.
        
        Args:
            text: The text to find similar items for
            namespace: The namespace to search in
            top_k: Maximum number of similar items to return
            
        Returns:
            List of similar items with their scores, sorted by similarity (highest first)
        """
        try:
            # Get embedding for the query text
            query_embedding = await self._get_embedding(text)
            
            # Get all items in the namespace
            pattern = f"{self.namespace}:{namespace}:*"
            keys = await self.redis.keys(pattern)
            
            if not keys:
                return []
                
            # Get all items and their embeddings
            items = []
            for key in keys:
                item_data = await self.redis.get(key)
                if item_data:
                    item = json.loads(item_data)
                    items.append(item)
            
            # Calculate similarity scores
            scored_items = []
            for item in items:
                if 'embedding' not in item:
                    continue
                    
                similarity = self._cosine_similarity(query_embedding, item['embedding'])
                if similarity >= self.similarity_threshold:
                    scored_items.append({
                        'key': item['key'],
                        'text': item['text'],
                        'metadata': item.get('metadata', {}),
                        'similarity': similarity
                    })
            
            # Sort by similarity (highest first) and return top-k
            scored_items.sort(key=lambda x: x['similarity'], reverse=True)
            return scored_items[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar items: {e}")
            return []
    
    async def store(self, key: str, text: str, namespace: str, 
                   metadata: Optional[Dict[str, Any]] = None, ttl: Optional[int] = None) -> bool:
        """
        Store an item in the semantic cache.
        
        Args:
            key: Unique identifier for the item
            text: The text content to store
            namespace: Namespace for the item (e.g., 'attacks', 'evaluations')
            metadata: Additional metadata to store with the item
            ttl: Time to live in seconds (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate embedding for the text
            embedding = await self._get_embedding(text)
            
            # Prepare the item data
            item = {
                'key': key,
                'text': text,
                'embedding': embedding,
                'metadata': metadata or {}
            }
            
            # Store in Redis
            redis_key = f"{self.namespace}:{namespace}:{key}"
            await self.redis.set(redis_key, json.dumps(item), ex=ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing item in semantic cache: {e}")
            return False
    
    async def get(self, key: str, namespace: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an item from the semantic cache by key.
        
        Args:
            key: The key of the item to retrieve
            namespace: The namespace the item is stored in
            
        Returns:
            The cached item or None if not found
        """
        try:
            redis_key = f"{self.namespace}:{namespace}:{key}"
            item_data = await self.redis.get(redis_key)
            
            if item_data:
                return json.loads(item_data)
                
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving item from semantic cache: {e}")
            return None
    
    async def delete(self, key: str, namespace: str) -> bool:
        """
        Delete an item from the semantic cache.
        
        Args:
            key: The key of the item to delete
            namespace: The namespace the item is stored in
            
        Returns:
            bool: True if the item was deleted, False otherwise
        """
        try:
            redis_key = f"{self.namespace}:{namespace}:{key}"
            result = await self.redis.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting item from semantic cache: {e}")
            return False
