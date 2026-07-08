from fastapi import APIRouter

from app.core.config import settings
from app.dependencies import OllamaClientDep
from app.schemas.health import HealthOut

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthOut)
async def health(llm: OllamaClientDep) -> HealthOut:
    models = await llm.list_models()
    return HealthOut(
        status="ok" if models is not None else "degraded",
        ollama=models is not None,
        model=settings.OLLAMA_MODEL,
        available_models=models or [],
    )
