from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.redis.client import redis_client

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("Starting up FastAPI application...")
    try:
        # test redis connection
        await redis_client.ping()
        logger.info("Redis connection established successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        # don't raise the exception, just log it
        # the app can still run without redis for development
    
    yield
    
    # shutdown
    logger.info("Shutting down FastAPI application...")
    try:
        await redis_client.close()
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")

# create fastapi app
app = FastAPI(
    title="Redact-LLM API",
    description="API for generating and managing attack prompts",
    version="1.0.0",
    lifespan=lifespan
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
    try:
        await redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis connection failed: {e}")

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
