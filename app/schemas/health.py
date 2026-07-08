from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str  # "ok" | "degraded"
    ollama: bool  # is the LLM backend reachable
    model: str  # default model (OLLAMA_MODEL)
    available_models: list[str] = []  # models pulled on the server (for experiments)
