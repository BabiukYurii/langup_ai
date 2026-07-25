from fastapi import APIRouter, Depends

from app.core.security import require_api_key
from app.dependencies import LLMClientDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["Chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(data: ChatRequest, llm: LLMClientDep) -> ChatResponse:
    """Generic inference gateway: messages in, raw model output out.

    All domain logic (prompts, parsing, validation) lives in the main backend.
    """
    content = await llm.chat(
        data.messages,
        data.json_format,
        data.temperature,
        model=data.model,
        keep_alive=data.keep_alive,
    )
    # The server runs one model, so we report that — a per-request `model`
    # override is accepted but has no effect (see LlamaCppClient.chat).
    return ChatResponse(content=content, model=llm.model)
