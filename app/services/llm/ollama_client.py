# Thin async client for the local Ollama server (native /api/chat endpoint).
import httpx

from app.core.config import settings
from app.core.exc import LLMUnavailableException
from app.schemas.chat import ChatMessage


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL,
        timeout: float = settings.OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def chat(
        self,
        messages: list[ChatMessage],
        json_format: bool,
        temperature: float,
        model: str | None = None,
        keep_alive: int | str | None = None,
    ) -> str:
        """One-shot chat completion; returns the raw content string."""
        payload = {
            "model": model or self.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_format:
            payload["format"] = "json"
        if keep_alive is not None:
            # Sent only when asked for, so Ollama keeps applying its own default.
            payload["keep_alive"] = keep_alive
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMUnavailableException(f"Ollama request failed: {e}") from e
        return resp.json()["message"]["content"]

    async def list_models(self) -> list[str] | None:
        """Names of locally pulled models; None when Ollama is unreachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
        except httpx.HTTPError:
            return None


def get_ollama_client() -> OllamaClient:
    return OllamaClient()
