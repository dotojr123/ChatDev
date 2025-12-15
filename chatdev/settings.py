from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Project Info
    PROJECT_NAME: str = "ChatDev"
    VERSION: str = "2.0.0"

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"

    # Vector DB (Memory)
    QDRANT_URL: Optional[str] = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    MEMORY_COLLECTION_NAME: str = "chatdev_memory"

    # Logging
    LOG_LEVEL: str = "INFO"
    JSON_LOGS: bool = True

    # Feature Flags
    ENABLE_TELEMETRY: bool = False
    ENABLE_ASYNC: bool = True

settings = Settings()
