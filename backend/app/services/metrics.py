import asyncio
import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from app.redis.client import redis_client

logger = logging.getLogger(__name__)

class MetricsCollector:
    """comprehensive metrics collection for judges and monitoring"""
    
    def __init__(self):
        self.metrics_prefix = "metrics:"
        self.session_prefix = "session:"
        
    async def record_attack_generation(self, prompt: str, attack_count: int, attack_types: List[str]):
        """record attack generation metrics"""
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # increment total prompts tested
            await redis_client.incr(f"{self.metrics_prefix}total_prompts")
            
            # increment total attacks generated
            await redis_client.incr(f"{self.metrics_prefix}total_attacks", attack_count)
            
            # record attack types distribution
            for attack_type in attack_types:
                await redis_client.incr(f"{self.metrics_prefix}attack_types:{attack_type}")
            
            # store session data
            session_data = {
                "timestamp": timestamp,
                "prompt": prompt[:100],  # truncate for storage
                "attack_count": attack_count,
                "attack_types": attack_types
            }
            
            session_id = f"{self.session_prefix}{timestamp}"
            await redis_client.setex(
                session_id,
                86400,  # 24 hours
                json.dumps(session_data)
            )
            
        except Exception as e:
            logger.error(f"Error recording attack generation: {e}")
    
    async def record_verdict(self, attack_id: str, verdict_data: Dict[str, Any]):
        """record verdict metrics"""
        try:
            # increment total verdicts
            await redis_client.incr(f"{self.metrics_prefix}total_verdicts")
            
            # record risk levels
            risk_level = verdict_data.get("risk_level", "unknown")
            await redis_client.incr(f"{self.metrics_prefix}risk_levels:{risk_level}")
            
            # record jailbreak detections
            evaluations = verdict_data.get("evaluations", {})
            jailbreak_eval = evaluations.get("jailbreak", {})
            if jailbreak_eval.get("jailbreak_detected"):
                await redis_client.incr(f"{self.metrics_prefix}jailbreaks_caught")
            
            # record hallucination detections
            hallucination_eval = evaluations.get("hallucination", {})
            if hallucination_eval.get("hallucination_detected"):
                await redis_client.incr(f"{self.metrics_prefix}hallucinations_caught")
            
            # record safety concerns
            safety_eval = evaluations.get("safety", {})
            if safety_eval.get("safety_concern"):
                await redis_client.incr(f"{self.metrics_prefix}safety_concerns")
            
            # record tone mismatches
            tone_eval = evaluations.get("tone_match", {})
            if tone_eval.get("tone_mismatch"):
                await redis_client.incr(f"{self.metrics_prefix}tone_mismatches")
            
            # store verdict for heatmap
            verdict_key = f"{self.metrics_prefix}verdict:{attack_id}"
            await redis_client.setex(
                verdict_key,
                86400,  # 24 hours
                json.dumps(verdict_data)
            )
            
        except Exception as e:
            logger.error(f"Error recording verdict: {e}")
    
    async def record_cache_hit(self, cache_type: str):
        """record cache hit metrics"""
        try:
            await redis_client.incr(f"{self.metrics_prefix}cache_hits:{cache_type}")
        except Exception as e:
            logger.error(f"Error recording cache hit: {e}")
    
    async def record_api_call(self, endpoint: str, status_code: int, response_time: float):
        """record API call metrics"""
        try:
            # increment total API calls
            await redis_client.incr(f"{self.metrics_prefix}total_api_calls")
            
            # record endpoint usage
            await redis_client.incr(f"{self.metrics_prefix}endpoints:{endpoint}")
            
            # record status codes
            await redis_client.incr(f"{self.metrics_prefix}status_codes:{status_code}")
            
            # record response times
            response_time_key = f"{self.metrics_prefix}response_times:{endpoint}"
            current_avg = await redis_client.get(response_time_key)
            
            if current_avg:
                current_avg = float(current_avg)
                count = await redis_client.get(f"{self.metrics_prefix}response_time_count:{endpoint}")
                count = int(count) if count else 1
                
                # calculate new average
                new_avg = ((current_avg * count) + response_time) / (count + 1)
                await redis_client.set(response_time_key, str(new_avg))
                await redis_client.incr(f"{self.metrics_prefix}response_time_count:{endpoint}")
            else:
                await redis_client.set(response_time_key, str(response_time))
                await redis_client.set(f"{self.metrics_prefix}response_time_count:{endpoint}", "1")
            
        except Exception as e:
            logger.error(f"Error recording API call: {e}")
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """get comprehensive statistics for judges"""
        try:
            stats = {}
            
            # basic counts
            stats["total_prompts"] = int(await redis_client.get(f"{self.metrics_prefix}total_prompts") or 0)
            stats["total_attacks"] = int(await redis_client.get(f"{self.metrics_prefix}total_attacks") or 0)
            stats["total_verdicts"] = int(await redis_client.get(f"{self.metrics_prefix}total_verdicts") or 0)
            stats["total_api_calls"] = int(await redis_client.get(f"{self.metrics_prefix}total_api_calls") or 0)
            
            # attack type distribution
            attack_types = ["jailbreak", "hallucination", "advanced"]
            stats["attack_type_distribution"] = {}
            for attack_type in attack_types:
                count = await redis_client.get(f"{self.metrics_prefix}attack_types:{attack_type}")
                stats["attack_type_distribution"][attack_type] = int(count) if count else 0
            
            # risk level distribution
            risk_levels = ["low", "medium", "high"]
            stats["risk_level_distribution"] = {}
            for risk_level in risk_levels:
                count = await redis_client.get(f"{self.metrics_prefix}risk_levels:{risk_level}")
                stats["risk_level_distribution"][risk_level] = int(count) if count else 0
            
            # security metrics
            stats["jailbreaks_caught"] = int(await redis_client.get(f"{self.metrics_prefix}jailbreaks_caught") or 0)
            stats["hallucinations_caught"] = int(await redis_client.get(f"{self.metrics_prefix}hallucinations_caught") or 0)
            stats["safety_concerns"] = int(await redis_client.get(f"{self.metrics_prefix}safety_concerns") or 0)
            stats["tone_mismatches"] = int(await redis_client.get(f"{self.metrics_prefix}tone_mismatches") or 0)
            
            # cache performance
            cache_types = ["attack_cache", "response_cache", "verdict_cache"]
            stats["cache_performance"] = {}
            for cache_type in cache_types:
                hits = await redis_client.get(f"{self.metrics_prefix}cache_hits:{cache_type}")
                stats["cache_performance"][cache_type] = int(hits) if hits else 0
            
            # calculate average robustness score
            if stats["total_verdicts"] > 0:
                total_risk = 0
                for risk_level, count in stats["risk_level_distribution"].items():
                    risk_score = {"low": 0, "medium": 50, "high": 100}.get(risk_level, 0)
                    total_risk += risk_score * count
                
                stats["average_robustness_score"] = 100 - (total_risk / stats["total_verdicts"])
            else:
                stats["average_robustness_score"] = 0
            
            # API endpoint usage
            endpoints = ["/generate", "/test-resistance", "/stats", "/pipeline/stats", "/pipeline/verdicts"]
            stats["endpoint_usage"] = {}
            for endpoint in endpoints:
                count = await redis_client.get(f"{self.metrics_prefix}endpoints:{endpoint}")
                stats["endpoint_usage"][endpoint] = int(count) if count else 0
            
            # response time averages
            stats["average_response_times"] = {}
            for endpoint in endpoints:
                avg_time = await redis_client.get(f"{self.metrics_prefix}response_times:{endpoint}")
                if avg_time:
                    stats["average_response_times"][endpoint] = float(avg_time)
            
            # status code distribution
            status_codes = ["200", "400", "500"]
            stats["status_code_distribution"] = {}
            for status_code in status_codes:
                count = await redis_client.get(f"{self.metrics_prefix}status_codes:{status_code}")
                stats["status_code_distribution"][status_code] = int(count) if count else 0
            
            # heatmap data (last 24 hours)
            stats["heatmap_data"] = await self._get_heatmap_data()
            
            # system health
            stats["system_health"] = {
                "redis_connected": True,  # if we got here, redis is working
                "last_updated": datetime.utcnow().isoformat(),
                "uptime_seconds": asyncio.get_event_loop().time()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting comprehensive stats: {e}")
            return {"error": str(e)}
    
    async def _get_heatmap_data(self) -> Dict[str, Any]:
        """get heatmap data for attack success rates"""
        try:
            # get recent verdicts (last 24 hours)
            verdict_keys = await redis_client.keys(f"{self.metrics_prefix}verdict:*")
            recent_verdicts = []
            
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            for key in verdict_keys:
                try:
                    verdict_data = await redis_client.get(key)
                    if verdict_data:
                        verdict = json.loads(verdict_data)
                        verdict_time = datetime.fromisoformat(verdict.get("evaluated_at", ""))
                        
                        if verdict_time > cutoff_time:
                            recent_verdicts.append(verdict)
                except Exception as e:
                    logger.warning(f"Error processing verdict key {key}: {e}")
            
            # calculate heatmap
            heatmap = {
                "jailbreak": {"success": 0, "total": 0, "success_rate": 0},
                "hallucination": {"success": 0, "total": 0, "success_rate": 0},
                "safety": {"success": 0, "total": 0, "success_rate": 0},
                "tone_match": {"success": 0, "total": 0, "success_rate": 0}
            }
            
            for verdict in recent_verdicts:
                evaluations = verdict.get("evaluations", {})
                
                for eval_type in heatmap.keys():
                    eval_data = evaluations.get(eval_type, {})
                    heatmap[eval_type]["total"] += 1
                    
                    # check if attack was successful
                    if eval_type == "jailbreak" and eval_data.get("jailbreak_detected"):
                        heatmap[eval_type]["success"] += 1
                    elif eval_type == "hallucination" and eval_data.get("hallucination_detected"):
                        heatmap[eval_type]["success"] += 1
                    elif eval_type == "safety" and eval_data.get("safety_concern"):
                        heatmap[eval_type]["success"] += 1
                    elif eval_type == "tone_match" and eval_data.get("tone_mismatch"):
                        heatmap[eval_type]["success"] += 1
            
            # calculate success rates
            for eval_type in heatmap.keys():
                total = heatmap[eval_type]["total"]
                success = heatmap[eval_type]["success"]
                heatmap[eval_type]["success_rate"] = (success / total * 100) if total > 0 else 0
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Error getting heatmap data: {e}")
            return {}

# global instance
metrics_collector = MetricsCollector() 