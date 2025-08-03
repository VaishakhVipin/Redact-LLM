from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis-15632.c330.asia-south1-1.gce.redns.redis-cloud.com")
    REDIS_USERNAME: str = os.getenv("REDIS_USERNAME", "default")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "15632"))
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    class Config:
        env_file = ".env"
        extra = "ignore"  # ignore extra fields

settings = Settings()
