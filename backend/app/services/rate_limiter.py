import asyncio
import logging
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from app.redis.client import RedisManager

logger = logging.getLogger(__name__)

class RateLimiter:
    """comprehensive rate limiting for security and spam prevention"""
    
    def __init__(self):
        self.redis_manager = RedisManager()
        self.rate_limits = {
            "attack_generation": {
                "per_user": {"requests": 50, "window": 3600},  # 50 requests per hour per user
                "per_ip": {"requests": 100, "window": 3600},   # 100 requests per hour per IP
                "global": {"requests": 1000, "window": 3600}   # 1000 requests per hour globally
            },
            "api_calls": {
                "per_user": {"requests": 200, "window": 3600},
                "per_ip": {"requests": 500, "window": 3600},
                "global": {"requests": 5000, "window": 3600}
            }
        }
    
    def _generate_key(self, prefix: str, identifier: str, window: int) -> str:
        """generate redis key for rate limiting"""
        current_window = int(asyncio.get_event_loop().time()) // window
        return f"rate_limit:{prefix}:{identifier}:{current_window}"
    
    async def check_rate_limit(self, 
                              service: str, 
                              user_id: str = None, 
                              ip_address: str = None) -> Dict[str, Any]:
        """check if request is within rate limits"""
        try:
            if service not in self.rate_limits:
                return {"allowed": True, "reason": "No rate limit configured"}
            
            limits = self.rate_limits[service]
            results = {}
            
            # Get the Redis client
            redis_client = await self.redis_manager.get_client()
            if not redis_client:
                raise RuntimeError("Failed to initialize Redis client")
            
            # check per-user limit
            if user_id and "per_user" in limits:
                user_key = self._generate_key(f"{service}_user", user_id, limits["per_user"]["window"])
                user_count = await redis_client.get(user_key)
                user_count = int(user_count) if user_count else 0
                
                if user_count >= limits["per_user"]["requests"]:
                    return {
                        "allowed": False,
                        "reason": f"User rate limit exceeded: {user_count}/{limits['per_user']['requests']}",
                        "limit_type": "per_user",
                        "current": user_count,
                        "limit": limits["per_user"]["requests"],
                        "window": limits["per_user"]["window"]
                    }
                results["user"] = {"current": user_count, "limit": limits["per_user"]["requests"]}
            
            # check per-IP limit
            if ip_address and "per_ip" in limits:
                ip_key = self._generate_key(f"{service}_ip", ip_address, limits["per_ip"]["window"])
                ip_count = await redis_client.get(ip_key)
                ip_count = int(ip_count) if ip_count else 0
                
                if ip_count >= limits["per_ip"]["requests"]:
                    return {
                        "allowed": False,
                        "reason": f"IP rate limit exceeded: {ip_count}/{limits['per_ip']['requests']}",
                        "limit_type": "per_ip",
                        "current": ip_count,
                        "limit": limits["per_ip"]["requests"],
                        "window": limits["per_ip"]["window"]
                    }
                results["ip"] = {"current": ip_count, "limit": limits["per_ip"]["requests"]}
            
            # check global limit
            if "global" in limits:
                global_key = self._generate_key(f"{service}_global", "global", limits["global"]["window"])
                global_count = await redis_client.get(global_key)
                global_count = int(global_count) if global_count else 0
                
                if global_count >= limits["global"]["requests"]:
                    return {
                        "allowed": False,
                        "reason": f"Global rate limit exceeded: {global_count}/{limits['global']['requests']}",
                        "limit_type": "global",
                        "current": global_count,
                        "limit": limits["global"]["requests"],
                        "window": limits["global"]["window"]
                    }
                results["global"] = {"current": global_count, "limit": limits["global"]["requests"]}
            
            return {
                "allowed": True,
                "reason": "Within rate limits",
                "limits": results
            }
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return {"allowed": True, "reason": "Rate limit check failed"}
    
    async def increment_rate_limit(self, 
                                  service: str, 
                                  user_id: str = None, 
                                  ip_address: str = None) -> bool:
        """increment rate limit counters"""
        try:
            if service not in self.rate_limits:
                return True
            
            limits = self.rate_limits[service]
            
            # Get the Redis client
            redis_client = await self.redis_manager.get_client()
            if not redis_client:
                raise RuntimeError("Failed to initialize Redis client")
            
            async with redis_client.pipeline() as pipe:
                # increment per-user counter
                if user_id and "per_user" in limits:
                    user_key = self._generate_key(f"{service}_user", user_id, limits["per_user"]["window"])
                    await pipe.incr(user_key)
                    await pipe.expire(user_key, limits["per_user"]["window"])
                
                # increment per-IP counter
                if ip_address and "per_ip" in limits:
                    ip_key = self._generate_key(f"{service}_ip", ip_address, limits["per_ip"]["window"])
                    await pipe.incr(ip_key)
                    await pipe.expire(ip_key, limits["per_ip"]["window"])
                
                # increment global counter
                if "global" in limits:
                    global_key = self._generate_key(f"{service}_global", "global", limits["global"]["window"])
                    await pipe.incr(global_key)
                    await pipe.expire(global_key, limits["global"]["window"])
                
                await pipe.execute()        
            return True
            
        except Exception as e:
            logger.error(f"Error incrementing rate limit: {e}")
            return False
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """get comprehensive rate limiting statistics"""
        try:
            # Get Redis client
            redis_client = await self.redis_manager.get_client()
            if not redis_client:
                logger.error("Failed to initialize Redis client in get_rate_limit_stats")
                return {"error": "Failed to initialize Redis client"}
                
            stats = {}
            
            for service, limits in self.rate_limits.items():
                service_stats = {}
                
                for limit_type, config in limits.items():
                    # get current window key
                    current_window = int(asyncio.get_event_loop().time()) // config["window"]
                    key_pattern = f"rate_limit:{service}_{limit_type}:*:{current_window}"
                    
                    # get all keys for this pattern
                    keys = await redis_client.keys(key_pattern)
                    total_requests = 0
                    
                    for key in keys:
                        count = await redis_client.get(key)
                        if count:
                            total_requests += int(count)
                    
                    service_stats[limit_type] = {
                        "total_requests": total_requests,
                        "limit": config["requests"],
                        "window_seconds": config["window"],
                        "utilization_percent": (total_requests / config["requests"]) * 100 if config["requests"] > 0 else 0
                    }
                
                stats[service] = service_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting rate limit stats: {e}")
            return {"error": str(e)}

# global instance
rate_limiter = RateLimiter() 