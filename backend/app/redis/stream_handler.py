from app.redis.client import redis_client

STREAM_NAME = "prompt_queue"

async def push_prompt_to_stream(prompt: str):
    await redis_client.xadd(STREAM_NAME, {"prompt": prompt})
