import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.redis.client import redis_client

logger = logging.getLogger(__name__)

class JobQueue:
    """redis-based job queue for attack generation"""
    
    def __init__(self):
        self.stream_name = "attack_generation_jobs"
        self.results_prefix = "job_result:"
        self.job_timeout = 300  # 5 minutes
        
    async def submit_job(self, prompt: str, attack_types: List[str] = None, priority: int = 1) -> str:
        """submit a new job to the queue"""
        try:
            job_id = str(uuid.uuid4())
            job_data = {
                "id": job_id,
                "prompt": prompt,
                "attack_types": attack_types or ["jailbreak", "hallucination", "advanced"],
                "priority": priority,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "attempts": 0
            }
            
            # add job to stream
            await redis_client.xadd(
                self.stream_name,
                job_data
            )
            
            logger.info(f"Submitted job {job_id} for prompt: {prompt[:50]}...")
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """get job status and results"""
        try:
            # check for results
            result_key = f"{self.results_prefix}{job_id}"
            result = await redis_client.get(result_key)
            
            if result:
                result_data = json.loads(result)
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "result": result_data,
                    "retrieved_at": datetime.utcnow().isoformat()
                }
            
            # check if job is still in queue
            jobs = await redis_client.xrange(self.stream_name, "-", "+")
            for job in jobs:
                job_data = job[1]
                if job_data.get("id") == job_id:
                    return {
                        "job_id": job_id,
                        "status": job_data.get("status", "pending"),
                        "created_at": job_data.get("created_at"),
                        "attempts": job_data.get("attempts", 0)
                    }
            
            return {
                "job_id": job_id,
                "status": "not_found",
                "error": "Job not found in queue or results"
            }
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }
    
    async def get_pending_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """get list of pending jobs"""
        try:
            jobs = await redis_client.xrange(self.stream_name, "-", "+", count=limit)
            pending_jobs = []
            
            for job in jobs:
                job_data = job[1]
                if job_data.get("status") == "pending":
                    pending_jobs.append({
                        "job_id": job_data.get("id"),
                        "prompt": job_data.get("prompt"),
                        "created_at": job_data.get("created_at"),
                        "priority": job_data.get("priority", 1)
                    })
            
            return pending_jobs
            
        except Exception as e:
            logger.error(f"Error getting pending jobs: {e}")
            return []
    
    async def store_job_result(self, job_id: str, result: Dict[str, Any]) -> bool:
        """store job result in redis"""
        try:
            result_key = f"{self.results_prefix}{job_id}"
            result_data = {
                "job_id": job_id,
                "result": result,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            await redis_client.setex(
                result_key,
                self.job_timeout,
                json.dumps(result_data)
            )
            
            logger.info(f"Stored result for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing job result: {e}")
            return False
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """cleanup old completed jobs"""
        try:
            # get all result keys
            result_keys = await redis_client.keys(f"{self.results_prefix}*")
            cleaned_count = 0
            
            for key in result_keys:
                try:
                    # check if result is old
                    result = await redis_client.get(key)
                    if result:
                        result_data = json.loads(result)
                        completed_at = datetime.fromisoformat(result_data.get("completed_at", ""))
                        age_hours = (datetime.utcnow() - completed_at).total_seconds() / 3600
                        
                        if age_hours > max_age_hours:
                            await redis_client.delete(key)
                            cleaned_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error processing result key {key}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} old job results")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {e}")
            return 0
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """get queue statistics"""
        try:
            # get total jobs in stream
            total_jobs = len(await redis_client.xrange(self.stream_name, "-", "+"))
            
            # get pending jobs
            pending_jobs = await self.get_pending_jobs(limit=1000)
            
            # get completed jobs (results)
            result_keys = await redis_client.keys(f"{self.results_prefix}*")
            
            return {
                "total_jobs": total_jobs,
                "pending_jobs": len(pending_jobs),
                "completed_jobs": len(result_keys),
                "queue_health": "healthy" if total_jobs < 1000 else "warning"
            }
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {"error": str(e)}

# global instance
job_queue = JobQueue() 