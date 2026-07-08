# Service settings, loaded from environment / .env.
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5vl:7b"
    OLLAMA_TIMEOUT_SECONDS: float = 120.0

    # Shared secret checked against the X-API-Key request header.
    API_KEY: str = "change-me"


settings = Settings()
