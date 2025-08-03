import redis.asyncio as redis
from app.core.config import settings

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    username=settings.REDIS_USERNAME,
    password=settings.REDIS_PASSWORD,
    ssl=False,  # redis cloud doesn't require SSL on this port
    decode_responses=True
)
