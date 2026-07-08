# DI aliases (same pattern as the main backend's dependencies.py).
from typing import Annotated

from fastapi import Depends

from app.services.llm.ollama_client import OllamaClient, get_ollama_client

OllamaClientDep = Annotated[OllamaClient, Depends(get_ollama_client)]
