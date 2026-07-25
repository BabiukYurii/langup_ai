from fastapi import APIRouter

from app.core.config import settings
from app.dependencies import LLMClientDep
from app.schemas.health import HealthOut

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthOut)
async def health(llm: LLMClientDep) -> HealthOut:
    models = await llm.list_models()
    return HealthOut(
        status="ok" if models is not None else "degraded",
        llm=models is not None,
        model=settings.LLAMACPP_MODEL,
        available_models=models or [],
    )
