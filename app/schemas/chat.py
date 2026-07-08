from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    # Base64-encoded images for vision-language models (Ollama /api/chat format).
    images: list[str] | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    # When true, Ollama constrains decoding to valid JSON (format="json").
    json_format: bool = True
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    # Optional override of the default model (must already be pulled on the server).
    model: str | None = None


class ChatResponse(BaseModel):
    content: str  # raw model output; the caller owns parsing/validation
    model: str
