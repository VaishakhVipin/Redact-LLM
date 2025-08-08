import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from app.core.config import settings
from app.redis.client import get_redis_client, close_redis, RedisManager
from app.services.evaluator import init_evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress Pydantic v2 deprecation warnings
import warnings
warnings.filterwarnings(
    "ignore",
    message="Valid config keys have changed in V2*",
    category=UserWarning,
    module="pydantic._internal._config"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting up FastAPI application...")
    
    # Initialize Redis client
    redis_client = None
    try:
        redis_client = await RedisManager().get_client()
        if redis_client:
            try:
                await redis_client.ping()
                app.state.redis = redis_client
                logger.info("✅ Successfully connected to Redis")
            except Exception as e:
                logger.error(f"❌ Redis connection test failed: {e}")
                app.state.redis = None
                redis_client = None
        else:
            logger.warning("⚠️  Redis client initialization returned None")
            app.state.redis = None
    except Exception as e:
        logger.error(f"❌ Failed to initialize Redis client: {e}")
        app.state.redis = None
    
    # Store the Redis client in app state
    app.state.redis = redis_client
    
    # Initialize evaluator
    try:
        await init_evaluator()
        logger.info("✅ Successfully initialized evaluator")
    except Exception as e:
        logger.error(f"❌ Failed to initialize evaluator: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FastAPI application...")
    try:
        if redis_client:
            try:
                await redis_client.close()
                logger.info("✅ Successfully closed Redis connection")
            except Exception as e:
                logger.error(f"❌ Error closing Redis connection: {e}")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
    finally:
        # Ensure we clean up even if there was an error
        try:
            await RedisManager().close()
        except Exception as e:
            logger.error(f"❌ Error in Redis cleanup: {e}")

# Create FastAPI app with proper Pydantic v2 config
app = FastAPI(
    title="Redact-LLM API",
    description="API for generating and managing attack prompts",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # configure this properly for production
)

# health check endpoint
@app.get("/health")
async def health_check():
    """health check endpoint"""
    redis_client = app.state.redis
    if not redis_client:
        return {"status": "degraded", "redis": "not_configured"}
    
    try:
        await redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {"status": "degraded", "redis": "connection_error"}

# import and include routers
from app.api.routes.attacks import router as attacks_router
app.include_router(attacks_router, prefix="/api/v1")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

@app.get("/")
async def root():
    """root endpoint"""
    return {"message": "Redact-LLM API is running"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
