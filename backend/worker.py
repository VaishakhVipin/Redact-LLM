import asyncio
import logging
import sys
import os
from app.core.config import settings
from app.services.executor_worker import executor_worker
from app.services.evaluator import evaluator

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Worker:
    """main worker that orchestrates the redis-driven pipeline"""
    
    def __init__(self):
        self.executor_worker = executor_worker
        self.evaluator = evaluator
        
    async def start_pipeline(self):
        """start the complete redis-driven pipeline"""
        logger.info("🚀 Starting Redis-driven LLM Ops Pipeline")
        logger.info("=" * 60)
        
        try:
            # start executor worker and evaluator concurrently
            tasks = [
                self.executor_worker.consume_attacks(),
                self.evaluator.consume_results()
            ]
            
            logger.info("✅ Started executor worker (processing attacks from stream)")
            logger.info("✅ Started evaluator (analyzing responses and publishing verdicts)")
            logger.info("=" * 60)
            logger.info("🎯 Pipeline is now running!")
            logger.info("📊 Monitor stats at: http://localhost:8000/api/v1/attacks/stats")
            logger.info("🔄 Queue stats at: http://localhost:8000/api/v1/attacks/queue/stats")
            logger.info("=" * 60)
            
            # run both services
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline stopped by user")
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            raise
    
    async def get_pipeline_stats(self):
        """get comprehensive pipeline statistics"""
        try:
            executor_stats = await self.executor_worker.get_worker_stats()
            evaluator_stats = await self.evaluator.get_evaluator_stats()
            
            return {
                "pipeline_status": "running",
                "executor_worker": executor_stats,
                "evaluator": evaluator_stats,
                "timestamp": asyncio.get_event_loop().time()
            }
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {"error": str(e)}

# global instance
worker = Worker()

async def main():
    """main worker function"""
    try:
        await worker.start_pipeline()
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
