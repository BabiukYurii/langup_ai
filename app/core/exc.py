# Exceptions + FastAPI handlers.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class LLMUnavailableException(Exception):
    """The llama.cpp server is unreachable or timed out."""

    def __init__(self, message: str = "LLM backend is unavailable") -> None:
        self.message = message
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMUnavailableException)
    async def handle_llm_unavailable(_: Request, exc: LLMUnavailableException) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": exc.message})
