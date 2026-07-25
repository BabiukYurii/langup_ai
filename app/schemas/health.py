from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str  # "ok" | "degraded"
    llm: bool  # is the llama.cpp backend reachable
    model: str  # default model label (LLAMACPP_MODEL)
    available_models: list[str] = []  # what the server currently has loaded
