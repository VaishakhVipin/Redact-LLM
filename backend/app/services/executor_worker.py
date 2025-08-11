import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import redis.asyncio as redis
from cerebras.cloud.sdk import Cerebras
from app.redis.client import get_redis_client

logger = logging.getLogger(__name__)

executor_worker = None

async def init_executor_worker():
    """Initialize the global executor worker with Redis client"""
    global executor_worker
    redis_client = await get_redis_client()
    executor_worker = ExecutorWorker(redis_client=redis_client)
    return executor_worker


# Initialize Cerebras client
cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
MODEL_NAME = "gpt-oss-120b"

class ExecutorWorker:
    """Worker that processes attacks from Redis stream and sends them to target models"""
    
    def __init__(self, redis_client=None):
        logger.info("Initializing ExecutorWorker...")
        self.attack_stream = "attack_stream"
        self.response_cache_prefix = "attack_response:"
        self.verdict_channel = "verdict_channel"
        self.client = cerebras_client
        self.model_name = MODEL_NAME
        self.redis_client = redis_client
        self.semantic_cache = None
        logger.debug(f"Initialized ExecutorWorker with model: {self.model_name}")
        
        # Initialize semantic cache if Redis is available
        if redis_client:
            from app.services.semantic_cache import SemanticCache
            try:
                self.semantic_cache = SemanticCache(
                    redis_client=redis_client,
                    model_name='all-MiniLM-L6-v2',
                    similarity_threshold=0.85,
                    namespace="executor"
                )
                logger.info("Semantic cache initialized for executor")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic cache: {e}")
        
    def _generate_response_cache_key(self, attack_variant: str, base_prompt: str, model: str) -> str:
        """Generate semantic cache key for attack response"""
        # Use a consistent key format for semantic lookup
        return f"response_{hashlib.md5(f'{base_prompt}:{attack_variant}:{model}'.encode()).hexdigest()}"
    
    async def _get_cached_response(self, attack_variant: str, base_prompt: str, model: str) -> Optional[Dict[str, Any]]:
        """Get cached response using semantic cache"""
        if not self.semantic_cache:
            return None
            
        try:
            # Search for semantically similar responses
            similar_items = await self.semantic_cache.find_similar(
                text=f"{base_prompt}\n\n{attack_variant}",
                namespace=f"responses:{model}",
                top_k=1
            )
            
            if similar_items and similar_items[0]['similarity'] >= self.semantic_cache.similarity_threshold:
                logger.info(f"Semantic cache hit with similarity {similar_items[0]['similarity']:.2f}")
                return similar_items[0].get('metadata', {}).get('response')
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def _cache_response(self, cache_key: str, response_data: Dict[str, Any]) -> None:
        """
        Execute an attack against a prompt with semantic caching and retry logic.
        
        Args:
            prompt: The original prompt to attack
            attack: The attack string or template to use
            model_config: Configuration for the target model
            attack_type: Type of attack being executed
            max_retries: Maximum number of retry attempts
            use_semantic_cache: Whether to use semantic caching
            
        Returns:
            Dictionary containing execution results and metadata
        """
        logger.info(f"Executing {attack_type} attack...")
        start_time = time.time()
        
        # Generate cache key based on prompt, attack, and model config
        cache_key = await self._generate_cache_key(prompt, attack, model_config, attack_type)
        
        try:
            # Check exact cache first
            if self.redis_client:
                try:
                    cached_result = await self.redis_client.get(cache_key)
                    if cached_result:
                        logger.info(f"Cache hit for execution: {cache_key}")
                        result = json.loads(cached_result)
                        result['cached'] = True
                        return result
                except Exception as e:
                    logger.warning(f"Cache lookup failed: {e}")
            
            # Try semantic cache if enabled
            if use_semantic_cache and self.semantic_cache:
                try:
                    # Create a semantic key based on prompt and attack
                    semantic_key = f"{prompt[:500]}:{attack[:500]}"
                    similar = await self.semantic_cache.find_similar(
                        text=semantic_key,
                        namespace=f"executions:{attack_type}",
                        top_k=1
                    )
                    
                    if similar and similar[0]['similarity'] >= self.semantic_cache.similarity_threshold:
                        logger.info(f"Semantic cache hit with similarity {similar[0]['similarity']:.2f}")
                        cached_result = similar[0]['metadata'].get('result')
                        if cached_result:
                            # Cache the result for faster access
                            if self.redis_client:
                                await self.redis_client.setex(
                                    cache_key,
                                    self.cache_ttl,
                                    json.dumps(cached_result)
                                )
                            cached_result['cached'] = True
                            cached_result['similarity'] = similar[0]['similarity']
                            return cached_result
                            
                except Exception as e:
                    logger.warning(f"Semantic cache lookup failed: {e}")
            
            # Execute the attack if not found in cache
            result = await self._execute_with_retries(
                prompt=prompt,
                attack=attack,
                model_config=model_config,
                max_retries=max_retries
            )
            
            # Add metadata
            result.update({
                'cached': False,
                'attack_type': attack_type,
                'model_config': model_config,
                'execution_time': time.time() - start_time,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Cache the results
            if self.redis_client:
                try:
                    # Cache exact match
                    await self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        json.dumps(result)
                    )
                    
                    # Store stale version with longer TTL
                    await self.redis_client.setex(
                        f"{cache_key}:stale",
                        self.cache_ttl * 24,  # 24 hours
                        json.dumps(result)
                    )
                    
                    # Update semantic cache
                    if use_semantic_cache and self.semantic_cache:
                        semantic_key = f"{prompt[:500]}:{attack[:500]}"
                        await self.semantic_cache.store(
                            key=cache_key,
                            text=semantic_key,
                            namespace=f"executions:{attack_type}",
                            metadata={
                                'result': result,
                                'prompt_hash': hashlib.md5(prompt.encode()).hexdigest(),
                                'attack_hash': hashlib.md5(attack.encode()).hexdigest(),
                                'attack_type': attack_type,
                                'model_name': model_config.get('model_name', 'unknown'),
                                'executed_at': datetime.utcnow().isoformat()
                            },
                            ttl=self.cache_ttl
                        )
                        
                except Exception as e:
                    logger.warning(f"Failed to cache execution results: {e}")
            
            # Log metrics
            duration = time.time() - start_time
            logger.info(f"Executed {attack_type} attack in {duration:.2f}s")
            
            # Record metrics if available
            if metrics_collector:
                metrics_collector.record_metric(
                    'attack_execution_time',
                    duration,
                    tags={
                        'attack_type': attack_type,
                        'model': model_config.get('model_name', 'unknown'),
                        'cache_hit': False
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute attack: {e}")
            logger.error(traceback.format_exc())
            
            # Try to return stale cache if available
            if self.redis_client:
                try:
                    stale = await self.redis_client.get(f"{cache_key}:stale")
                    if stale:
                        logger.warning("Returning stale cache due to execution failure")
                        result = json.loads(stale)
                        result['stale'] = True
                        return result
                except Exception as cache_err:
                    logger.warning(f"Failed to get stale cache: {cache_err}")
            
            # If we get here, re-raise the original error
            raise
    
    async def _execute_with_retries(
        self,
        prompt: str,
        attack: str,
        model_config: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Execute the attack with retry logic and exponential backoff.
        
        Args:
            prompt: The original prompt
            attack: The attack string or template
            model_config: Configuration for the target model
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing the execution result
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Apply the attack to the prompt
                attacked_prompt = self._apply_attack_template(prompt, attack)
                
                # Execute the model with the attacked prompt
                response = await self._call_model(
                    prompt=attacked_prompt,
                    model_config=model_config
                )
                
                # Return successful response
                return {
                    'success': True,
                    'prompt': prompt,
                    'attacked_prompt': attacked_prompt,
                    'response': response,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # If we have retries left, wait before retrying
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    base_delay = min(2 ** attempt, 10)  # Cap at 10 seconds
                    jitter = random.uniform(0, 1)  # Add up to 1 second of jitter
                    delay = base_delay + jitter
                    
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        error_msg = f"Failed after {max_retries + 1} attempts: {str(last_error)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error': error_msg,
            'prompt': prompt,
            'attacked_prompt': attacked_prompt if 'attacked_prompt' in locals() else None,
            'attempts': max_retries + 1
        }
    
    async def _call_model(
        self,
        prompt: str,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call the target model with the given prompt.
        
        Args:
            prompt: The prompt to send to the model
            model_config: Configuration for the target model
            
        Returns:
            Dictionary containing the model response
        """
        try:
            # Call the model using the provided configuration
            response = await asyncio.get_event_loop().run_in_executor(
                None,  # uses default executor
                lambda: self.client.chat.completions.create(
                    model=model_config['model_name'],
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
            )
            
            # Process the response
            response_text = response.choices[0].message.content.strip() if hasattr(response, 'choices') else str(response)
            return {
                'response': response_text,
                'metadata': {
                    'finish_reason': response.choices[0].finish_reason,
                    'completion_tokens': response.usage.completion_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
        
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            raise
    
    async def _apply_attack_template(
        self,
        prompt: str,
        attack: str
    ) -> str:
        """
        Apply the attack template to the prompt.
        
        Args:
            prompt: The original prompt
            attack: The attack string or template
            
        Returns:
            The attacked prompt
        """
        # Apply the attack template to the prompt
        # This is a placeholder for the actual attack logic
        return prompt + " " + attack
    
    async def process_attack(self, attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """process a single attack"""
        start_time = time.time()
        attack_id = attack_data.get("attack_id")
        
        try:
            base_prompt = attack_data.get("base_prompt")
            attack_variant = attack_data.get("attack_variant")
            attack_type = attack_data.get("attack_type")
            attack_subtype = attack_data.get("attack_subtype")
            
            logger.info(f"Processing attack {attack_id}: {attack_type}/{attack_subtype}")
            logger.debug(f"Base prompt: {base_prompt[:100]}...")
            logger.trace(f"Attack variant: {attack_variant}")
            
            # Execute the attack
            result = await self.execute_attack(
                prompt=base_prompt,
                attack=attack_variant,
                model_config={'model_name': self.model_name},
                attack_type=attack_type
            )
            
            # create result data
            result.update({
                "attack_id": attack_id,
                "base_prompt": base_prompt,
                "attack_variant": attack_variant,
                "attack_type": attack_type,
                "attack_subtype": attack_subtype,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "cache_hit": result.get("cached", False),
                "cache_type": result.get("cache_type", "none")
            })
            
            # store result in redis
            result_key = f"attack_result:{attack_id}"
            try:
                await self.redis_client.setex(
                    result_key,
                    3600,  # 1 hour
                    json.dumps(result)
                )
                logger.info(f"Stored result for attack {attack_id} in Redis")
                logger.debug(f"Result key: {result_key}")
            except Exception as e:
                logger.error(f"Failed to store result in Redis: {e}", exc_info=True)
                raise
                
            # Log processing time
            duration = time.time() - start_time
            logger.info(f"Completed processing attack {attack_id} in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            error_id = str(uuid.uuid4())
            error_time = datetime.now(timezone.utc).isoformat()
            logger.error(f"Error processing attack {attack_id} (Error ID: {error_id}): {e}", exc_info=True)
            
            # Log detailed error information
            error_details = {
                "error_id": error_id,
                "timestamp": error_time,
                "attack_id": attack_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "attack_type": attack_data.get("attack_type"),
                "attack_subtype": attack_data.get("attack_subtype"),
                "base_prompt_preview": (attack_data.get("base_prompt") or "")[:100] + "..."
            }
            
            logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
            
            return {
                "attack_id": attack_id,
                "error": f"{type(e).__name__}: {str(e)}",
                "error_id": error_id,
                "processed_at": error_time,
                "error_details": error_details
            }
    
    async def consume_attacks(self):
        """consume attacks from redis stream"""
        logger.info(f"Starting executor worker, listening to {self.attack_stream}")
        logger.debug(f"Using model: {self.model_name}")
        
        # create consumer group if it doesn't exist
        try:
            await self.redis_client.xgroup_create(
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
                messages = await self.redis_client.xreadgroup(
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

# Initialize executor_worker as None, it will be initialized with Redis client via init_executor_worker
executor_worker = None