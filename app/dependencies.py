# DI aliases (same pattern as the main backend's dependencies.py).
from typing import Annotated

from fastapi import Depends

from app.services.llm.llamacpp_client import LlamaCppClient, get_llm_client

LLMClientDep = Annotated[LlamaCppClient, Depends(get_llm_client)]
