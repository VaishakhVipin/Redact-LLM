import asyncio
from app.redis.client import redis_client
from app.services.attack_generator import generate_attacks
from app.services.executor import run_prompt
from app.services.evaluator import score_response
from app.redis.pubsub import publish_verdict

STREAM_NAME = "prompt_queue"

async def redteam_loop():
    last_id = "0-0"
    while True:
        messages = await redis_client.xread({STREAM_NAME: last_id}, block=0, count=1)
        for stream, msgs in messages:
            for msg_id, msg_data in msgs:
                prompt = msg_data[b'prompt'].decode()

                attack_variants = await generate_attacks(prompt)
                for attack in attack_variants:
                    response = await run_prompt(attack)
                    verdict = await score_response(prompt, attack, response)
                    await publish_verdict(verdict)

                last_id = msg_id
