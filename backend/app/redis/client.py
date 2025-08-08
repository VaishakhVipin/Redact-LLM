import redis.asyncio as redis
from typing import Optional, Dict, Any
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Global Redis client instance
_redis_client: Optional[redis.Redis] = None
_redis_initialized: bool = False

class RedisManager:
    _instance = None
    _client: Optional[redis.Redis] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    async def get_client(cls) -> Optional[redis.Redis]:
        """Get or create a Redis client with connection pooling"""
        global _redis_client, _redis_initialized
        
        if _redis_client is not None and _redis_initialized:
            return _redis_client
            
        try:
            # Prepare connection kwargs
            connection_kwargs = {
                'host': settings.REDIS_HOST,
                'port': settings.REDIS_PORT,
                'username': settings.REDIS_USERNAME or None,  # Convert empty string to None
                'password': settings.REDIS_PASSWORD or None,  # Convert empty string to None
                'decode_responses': True,
                'max_connections': 20,
                'retry_on_timeout': True,
                'socket_connect_timeout': 5,
                'socket_timeout': 5,
                'health_check_interval': 30,
            }
            
            # Filter out None values
            connection_kwargs = {k: v for k, v in connection_kwargs.items() if v is not None}
            
            _redis_client = redis.Redis(**connection_kwargs)
            
            # Test the connection
            if await _redis_client.ping():
                _redis_initialized = True
                logger.info("Successfully connected to Redis")
                return _redis_client
            else:
                raise ConnectionError("Failed to ping Redis server")
                
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            _redis_client = None
            _redis_initialized = False
            return None
    
    @classmethod
    async def close(cls):
        """Close Redis connections"""
        global _redis_client, _redis_initialized
        
        if _redis_client:
            try:
                await _redis_client.close()
                logger.info("Closed Redis connection")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                _redis_client = None
                _redis_initialized = False

# Backward compatibility functions
async def get_redis_client() -> Optional[redis.Redis]:
    """Get the Redis client instance"""
    return await RedisManager.get_client()

async def close_redis():
    """Close Redis connections"""
    await RedisManager.close()

# For backward compatibility with synchronous code
# This will be None until get_redis_client() is called
redis_client = None

# Export RedisManager for use in other modules
__all__ = ['get_redis_client', 'close_redis', 'RedisManager']