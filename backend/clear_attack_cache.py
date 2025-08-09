# clear_attack_cache.py
import asyncio
import redis.asyncio as redis
from app.redis.client import get_redis_client
import os

async def clear_attack_cache():
    """
    Clear all attack-related cache keys including semantic cache
    """
    redis_client = None
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            print("❌ Failed to connect to Redis")
            return False
            
        # Patterns to match all attack-related cache keys
        cache_patterns = [
            "attack_*",           # Direct attack cache keys
            "semantic:attack_*",  # Semantic cache keys
            "rate_limit:*",       # Rate limiting keys
            "semantic:emb_*"      # Semantic embedding cache
        ]
        
        total_deleted = 0
        
        # Clear keys matching each pattern
        for pattern in cache_patterns:
            try:
                keys = await redis_client.keys(pattern)
                if keys:
                    await redis_client.delete(*keys)
                    print(f"✅ Cleared {len(keys)} keys matching pattern: {pattern}")
                    total_deleted += len(keys)
            except Exception as e:
                print(f"⚠️ Error clearing pattern {pattern}: {str(e)}")
        
        if total_deleted == 0:
            print("ℹ️ No matching cache keys found")
        else:
            print(f"✨ Successfully cleared {total_deleted} cache keys")
            
        # Clear the entire cache if explicitly configured via environment variable
        if os.getenv('CLEAR_ALL_CACHE', '').lower() in ('true', '1', 't'):
            print("⚠️ Clearing entire Redis cache (CLEAR_ALL_CACHE=True)")
            await redis_client.flushdb()
            print("✅ Successfully cleared entire Redis cache")
            
        return True
        
    except Exception as e:
        print(f"❌ Error clearing attack cache: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if redis_client:
            await redis_client.close()

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Run the cache clearing
    asyncio.run(clear_attack_cache())