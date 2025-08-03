import asyncio
import json
import logging
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
import aiohttp
from app.redis.client import redis_client

logger = logging.getLogger(__name__)

class ExecutorWorker:
    """executor worker that processes attacks from redis stream"""
    
    def __init__(self):
        self.attack_stream = "attack_stream"
        self.response_cache_prefix = "attack_response:"
        self.verdict_channel = "verdict_channel"
        self.target_models = {
            "gpt-4": "https://api.openai.com/v1/chat/completions",
            "claude": "https://api.anthropic.com/v1/messages",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        }
        self.default_model = "gpt-4"  # for demo purposes
        
    def _generate_response_cache_key(self, attack_variant: str, base_prompt: str, model: str) -> str:
        """generate cache key for attack response"""
        content_hash = hashlib.md5(f"{attack_variant}:{base_prompt}:{model}".encode()).hexdigest()
        return f"{self.response_cache_prefix}{content_hash}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """get cached response from redis"""
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def _cache_response(self, cache_key: str, response_data: Dict[str, Any]) -> None:
        """cache response in redis"""
        try:
            await redis_client.setex(
                cache_key,
                3600,  # 1 hour cache
                json.dumps(response_data)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def _send_to_target_model(self, attack_variant: str, model: str = None) -> Dict[str, Any]:
        """send attack to target model and get response"""
        model = model or self.default_model
        
        try:
            # for demo purposes, simulate API call
            # in production, you'd make actual API calls to the target models
            
            if model == "gpt-4":
                # simulate GPT-4 response
                await asyncio.sleep(0.5)  # simulate API delay
                response = {
                    "model": "gpt-4",
                    "response": f"I cannot provide information about that topic as it may be harmful or illegal. I aim to be helpful while following safety guidelines.",
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_used": model
                }
            elif model == "claude":
                # simulate Claude response
                await asyncio.sleep(0.3)
                response = {
                    "model": "claude",
                    "response": f"I apologize, but I cannot assist with that request as it may involve harmful or illegal activities.",
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_used": model
                }
            else:
                # simulate Gemini response
                await asyncio.sleep(0.4)
                response = {
                    "model": "gemini",
                    "response": f"I'm unable to help with that request as it may violate safety guidelines.",
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_used": model
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Error sending to target model {model}: {e}")
            return {
                "model": model,
                "response": "Error: Failed to get response from model",
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": model,
                "error": str(e)
            }
    
    async def _publish_verdict(self, attack_id: str, verdict_data: Dict[str, Any]) -> None:
        """publish verdict to redis pub/sub channel"""
        try:
            verdict_message = {
                "attack_id": attack_id,
                "verdict": verdict_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await redis_client.publish(
                self.verdict_channel,
                json.dumps(verdict_message)
            )
            
            logger.info(f"Published verdict for attack {attack_id}")
            
        except Exception as e:
            logger.error(f"Error publishing verdict: {e}")
    
    async def process_attack(self, attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """process a single attack"""
        try:
            attack_id = attack_data.get("attack_id")
            base_prompt = attack_data.get("base_prompt")
            attack_variant = attack_data.get("attack_variant")
            attack_type = attack_data.get("attack_type")
            attack_subtype = attack_data.get("attack_subtype")
            
            logger.info(f"Processing attack {attack_id}: {attack_type}/{attack_subtype}")
            
            # check cache first
            cache_key = self._generate_response_cache_key(attack_variant, base_prompt, self.default_model)
            cached_response = await self._get_cached_response(cache_key)
            
            if cached_response:
                logger.info(f"Using cached response for attack {attack_id}")
                response_data = cached_response
            else:
                # send to target model
                response_data = await self._send_to_target_model(attack_variant, self.default_model)
                
                # cache the response
                await self._cache_response(cache_key, response_data)
            
            # create result data
            result = {
                "attack_id": attack_id,
                "base_prompt": base_prompt,
                "attack_variant": attack_variant,
                "attack_type": attack_type,
                "attack_subtype": attack_subtype,
                "response": response_data,
                "processed_at": datetime.utcnow().isoformat(),
                "cache_hit": cached_response is not None
            }
            
            # store result in redis
            result_key = f"attack_result:{attack_id}"
            await redis_client.setex(
                result_key,
                3600,  # 1 hour
                json.dumps(result)
            )
            
            logger.info(f"Stored result for attack {attack_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing attack: {e}")
            return {
                "attack_id": attack_data.get("attack_id"),
                "error": str(e),
                "processed_at": datetime.utcnow().isoformat()
            }
    
    async def consume_attacks(self):
        """consume attacks from redis stream"""
        logger.info(f"Starting executor worker, listening to {self.attack_stream}")
        
        # create consumer group if it doesn't exist
        try:
            await redis_client.xgroup_create(
                self.attack_stream,
                "executor_workers",
                id="0",
                mkstream=True
            )
            logger.info("Created executor worker group")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Error creating consumer group: {e}")
        
        while True:
            try:
                # read from stream
                messages = await redis_client.xreadgroup(
                    "executor_workers",
                    "worker-1",
                    {self.attack_stream: ">"},
                    count=1,
                    block=5000  # 5 second timeout
                )
                
                if messages:
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            try:
                                # process the attack
                                result = await self.process_attack(fields)
                                
                                # acknowledge the message
                                await redis_client.xack(self.attack_stream, "executor_workers", msg_id)
                                
                                logger.info(f"Processed attack {fields.get('attack_id')}")
                                
                            except Exception as e:
                                logger.error(f"Error processing message {msg_id}: {e}")
                                # still acknowledge to avoid infinite retry
                                await redis_client.xack(self.attack_stream, "executor_workers", msg_id)
                
            except Exception as e:
                logger.error(f"Error in consume_attacks: {e}")
                await asyncio.sleep(1)  # wait before retrying
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """get worker statistics"""
        try:
            # get stream stats
            stream_length = await redis_client.xlen(self.attack_stream)
            
            # get pending messages
            pending = await redis_client.xpending_range(
                self.attack_stream,
                "executor_workers",
                "-",
                "+",
                100
            )
            
            # get result stats
            result_keys = await redis_client.keys("attack_result:*")
            
            return {
                "stream_length": stream_length,
                "pending_messages": len(pending),
                "processed_results": len(result_keys),
                "worker_status": "running",
                "consumer_group": "executor_workers"
            }
            
        except Exception as e:
            logger.error(f"Error getting worker stats: {e}")
            return {"error": str(e)}

# global instance
executor_worker = ExecutorWorker() 