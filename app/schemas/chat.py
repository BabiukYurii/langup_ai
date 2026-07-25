from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    # Base64-encoded images for vision-language models. Accepted for forward
    # compatibility; the current llama.cpp text path does not forward them.
    images: list[str] | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    # When true, the server constrains decoding to valid JSON (response_format).
    json_format: bool = True
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    # Accepted for backward compatibility but ignored: the llama.cpp server runs
    # a single always-resident model, so there is nothing to switch (`model`) or
    # unload (`keep_alive`). See LlamaCppClient.chat.
    model: str | None = None
    keep_alive: int | str | None = None


class ChatResponse(BaseModel):
    content: str  # raw model output; the caller owns parsing/validation
    model: str
