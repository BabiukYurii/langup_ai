# Service settings, loaded from environment / .env.
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # llama.cpp server (OpenAI-compatible /v1 API). It serves one always-resident
    # model out of the iGPU's dedicated VRAM via the Vulkan backend.
    LLAMACPP_BASE_URL: str = "http://localhost:8080"
    # Human-readable label reported to callers; the server itself decides which
    # weights it loaded (LLAMACPP_HF_REPO in docker-compose).
    LLAMACPP_MODEL: str = "gemma"
    LLAMACPP_TIMEOUT_SECONDS: float = 180.0

    # Shared secret checked against the X-API-Key request header.
    API_KEY: str = "change-me"


settings = Settings()
