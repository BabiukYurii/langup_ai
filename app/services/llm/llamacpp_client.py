# Thin async client for a local llama.cpp server (OpenAI-compatible /v1 API).
# The server runs one model, offloaded to the AMD iGPU via the Vulkan backend.
import re

import httpx

from app.core.config import settings
from app.core.exc import LLMUnavailableException
from app.schemas.chat import ChatMessage

# Gemma wraps its JSON in a ```json … ``` markdown block even when asked for a
# raw object (llama.cpp's response_format=json_object does not suppress it), so
# the backend's json.loads would choke. Peel the fence off before returning.
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_json_fence(text: str) -> str:
    m = _JSON_FENCE.match(text)
    return m.group(1).strip() if m else text.strip()


def _to_openai(m: ChatMessage) -> dict:
    # llama.cpp's /v1 endpoint takes plain {role, content}. We never send images
    # in production (the backend only asks for text JSON), so vision content is
    # not translated here; add multimodal parts if that ever changes.
    return {"role": m.role, "content": m.content}


class LlamaCppClient:
    def __init__(
        self,
        base_url: str = settings.LLAMACPP_BASE_URL,
        model: str = settings.LLAMACPP_MODEL,
        timeout: float = settings.LLAMACPP_TIMEOUT_SECONDS,
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
        """One-shot chat completion; returns the raw content string.

        `model` and `keep_alive` are accepted for gateway API compatibility but
        ignored: llama.cpp serves a single, always-resident model — there is
        nothing to switch or unload. Because the weights live in the iGPU's
        dedicated VRAM, keeping them resident costs no system RAM.
        """
        payload: dict = {
            "model": self.model,
            "messages": [_to_openai(m) for m in messages],
            "stream": False,
            "temperature": temperature,
        }
        if json_format:
            # Constrains decoding to valid JSON (grammar-backed), the same
            # guarantee the previous format="json" backend gave us.
            payload["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMUnavailableException(f"llama.cpp request failed: {e}") from e
        content = resp.json()["choices"][0]["message"]["content"]
        return _strip_json_fence(content) if json_format else content

    async def list_models(self) -> list[str] | None:
        """The model the server currently has loaded; None when it is unreachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                resp.raise_for_status()
                return [m["id"] for m in resp.json().get("data", [])]
        except httpx.HTTPError:
            return None


def get_llm_client() -> LlamaCppClient:
    return LlamaCppClient()
