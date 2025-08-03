from fastapi import APIRouter
from pydantic import BaseModel
from app.redis.stream_handler import push_prompt_to_stream

router = APIRouter()

class PromptPayload(BaseModel):
    prompt: str

@router.post("/test")
async def start_red_team(payload: PromptPayload):
    await push_prompt_to_stream(payload.prompt)
    return {"status": "queued"}
