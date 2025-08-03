from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import asyncio
import time

from app.services.attack_generator import generate_attacks, test_prompt_resistance, get_attack_stats
from app.services.job_queue import job_queue
from app.services.executor_worker import executor_worker
from app.services.evaluator import evaluator
from app.services.rate_limiter import rate_limiter
from app.services.metrics import metrics_collector
from app.redis.client import redis_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attacks", tags=["attacks"])

class AttackRequest(BaseModel):
    prompt: str
    attack_types: Optional[List[str]] = ["jailbreak", "hallucination", "advanced"]

class ResistanceTestRequest(BaseModel):
    prompt: str
    target_response: Optional[str] = None

class AttackResponse(BaseModel):
    prompt: str
    attacks: List[str]
    count: int
    categories: Dict[str, int]

class ResistanceTestResponse(BaseModel):
    original_prompt: str
    target_response: Optional[str]
    total_attacks: int
    attack_categories: Dict[str, int]
    attacks: List[str]
    resistance_score: int
    recommendations: List[str]

class JobSubmitRequest(BaseModel):
    prompt: str
    attack_types: Optional[List[str]] = ["jailbreak", "hallucination", "advanced"]
    priority: Optional[int] = 1

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

async def get_client_ip(request: Request) -> str:
    """extract client IP address"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

@router.post("/generate", response_model=AttackResponse)
async def generate_attack_prompts(
    request: AttackRequest, 
    http_request: Request,
    client_ip: str = Depends(get_client_ip)
):
    """generate attack prompts for a given input prompt with rate limiting and metrics"""
    start_time = time.time()
    
    try:
        # check rate limits
        rate_check = await rate_limiter.check_rate_limit(
            service="attack_generation",
            user_id=request.prompt[:20],  # simple user ID from prompt
            ip_address=client_ip
        )
        
        if not rate_check["allowed"]:
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded: {rate_check['reason']}"
            )
        
        logger.info(f"Generating attacks for prompt: {request.prompt[:50]}...")
        
        attacks = await generate_attacks(request.prompt)
        
        # categorize attacks
        categories = {
            "jailbreak": len([a for a in attacks if any(keyword in a.lower() for keyword in ["ignore", "bypass", "override", "ceo", "authority"])]),
            "hallucination": len([a for a in attacks if any(keyword in a.lower() for keyword in ["study", "research", "according to", "breaking news", "latest data"])]),
            "advanced": len([a for a in attacks if any(keyword in a.lower() for keyword in ["emergency", "protocol", "system", "critical"])])
        }
        
        # record metrics
        attack_types = list(categories.keys())
        await metrics_collector.record_attack_generation(
            prompt=request.prompt,
            attack_count=len(attacks),
            attack_types=attack_types
        )
        
        # increment rate limit
        await rate_limiter.increment_rate_limit(
            service="attack_generation",
            user_id=request.prompt[:20],
            ip_address=client_ip
        )
        
        response_time = time.time() - start_time
        await metrics_collector.record_api_call("/generate", 200, response_time)
        
        return AttackResponse(
            prompt=request.prompt,
            attacks=attacks,
            count=len(attacks),
            categories=categories
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating attacks: {e}")
        response_time = time.time() - start_time
        await metrics_collector.record_api_call("/generate", 500, response_time)
        raise HTTPException(status_code=500, detail=f"Failed to generate attacks: {str(e)}")

@router.post("/test-resistance", response_model=ResistanceTestResponse)
async def test_prompt_resistance_endpoint(request: ResistanceTestRequest):
    """test prompt resistance and return detailed analysis"""
    try:
        logger.info(f"Testing resistance for prompt: {request.prompt[:50]}...")
        
        results = await test_prompt_resistance(request.prompt, request.target_response)
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return ResistanceTestResponse(**results)
        
    except Exception as e:
        logger.error(f"Error testing resistance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test resistance: {str(e)}")

@router.get("/health")
async def attack_service_health():
    """health check for attack generation service"""
    try:
        # test with a simple prompt
        test_attacks = await generate_attacks("test")
        return {
            "status": "healthy",
            "service": "attack_generator",
            "test_attacks_generated": len(test_attacks)
        }
    except Exception as e:
        logger.error(f"Attack service health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Attack service unhealthy: {str(e)}")

@router.get("/stats")
async def get_attack_stats_endpoint(http_request: Request, client_ip: str = Depends(get_client_ip)):
    """get comprehensive statistics for judges and monitoring"""
    start_time = time.time()
    
    try:
        # check rate limits
        rate_check = await rate_limiter.check_rate_limit(
            service="api_calls",
            ip_address=client_ip
        )
        
        if not rate_check["allowed"]:
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded: {rate_check['reason']}"
            )
        
        # get comprehensive stats
        attack_stats = await get_attack_stats()
        comprehensive_stats = await metrics_collector.get_comprehensive_stats()
        rate_limit_stats = await rate_limiter.get_rate_limit_stats()
        
        # combine all stats
        combined_stats = {
            "attack_generator": attack_stats,
            "comprehensive_metrics": comprehensive_stats,
            "rate_limiting": rate_limit_stats,
            "redis_features_used": {
                "streams": "attack_stream for real-time processing",
                "pub_sub": "verdict_channel for live updates",
                "cache": "attack_response_cache for deduplication",
                "rate_limiting": "per-user and per-IP protection",
                "metrics": "real-time statistics collection"
            }
        }
        
        # increment rate limit
        await rate_limiter.increment_rate_limit(
            service="api_calls",
            ip_address=client_ip
        )
        
        response_time = time.time() - start_time
        await metrics_collector.record_api_call("/stats", 200, response_time)
        
        return combined_stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comprehensive stats: {e}")
        response_time = time.time() - start_time
        await metrics_collector.record_api_call("/stats", 500, response_time)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.post("/submit-job", response_model=Dict[str, str])
async def submit_attack_job(request: JobSubmitRequest):
    """submit a job for background attack generation"""
    try:
        job_id = await job_queue.submit_job(
            prompt=request.prompt,
            attack_types=request.attack_types,
            priority=request.priority
        )
        return {"job_id": job_id, "status": "submitted"}
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """get job status and results"""
    try:
        status = await job_queue.get_job_status(job_id)
        return JobStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.get("/queue/stats")
async def get_queue_stats():
    """get job queue statistics"""
    try:
        stats = await job_queue.get_queue_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")

@router.get("/queue/pending")
async def get_pending_jobs(limit: int = 10):
    """get list of pending jobs"""
    try:
        jobs = await job_queue.get_pending_jobs(limit=limit)
        return {"pending_jobs": jobs, "count": len(jobs)}
    except Exception as e:
        logger.error(f"Error getting pending jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pending jobs: {str(e)}")

@router.get("/pipeline/stats")
async def get_pipeline_stats():
    """get comprehensive pipeline statistics"""
    try:
        executor_stats = await executor_worker.get_worker_stats()
        evaluator_stats = await evaluator.get_evaluator_stats()
        
        return {
            "pipeline_status": "running",
            "executor_worker": executor_stats,
            "evaluator": evaluator_stats,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"Error getting pipeline stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline stats: {str(e)}")

@router.get("/pipeline/verdicts")
async def get_recent_verdicts(limit: int = 10):
    """get recent verdicts from the evaluator"""
    try:
        # get recent verdict keys
        verdict_keys = await redis_client.keys("verdict:*")
        recent_verdicts = []
        
        for key in verdict_keys[-limit:]:  # get most recent
            try:
                verdict_data = await redis_client.get(key)
                if verdict_data:
                    verdict = json.loads(verdict_data)
                    recent_verdicts.append(verdict)
            except Exception as e:
                logger.warning(f"Error processing verdict key {key}: {e}")
        
        return {
            "recent_verdicts": recent_verdicts,
            "count": len(recent_verdicts)
        }
    except Exception as e:
        logger.error(f"Error getting recent verdicts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent verdicts: {str(e)}")

@router.get("/pipeline/results")
async def get_attack_results(limit: int = 10):
    """get recent attack results"""
    try:
        # get recent result keys
        result_keys = await redis_client.keys("attack_result:*")
        recent_results = []
        
        for key in result_keys[-limit:]:  # get most recent
            try:
                result_data = await redis_client.get(key)
                if result_data:
                    result = json.loads(result_data)
                    recent_results.append(result)
            except Exception as e:
                logger.warning(f"Error processing result key {key}: {e}")
        
        return {
            "recent_results": recent_results,
            "count": len(recent_results)
        }
    except Exception as e:
        logger.error(f"Error getting attack results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get attack results: {str(e)}") 