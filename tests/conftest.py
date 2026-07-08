import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import create_app
from app.schemas.chat import ChatMessage
from app.services.llm.ollama_client import OllamaClient, get_ollama_client


class FakeOllama(OllamaClient):
    """In-memory stand-in: returns a canned response, no network."""

    def __init__(self, output: str = '{"ok": true}', alive: bool = True) -> None:
        super().__init__()
        self.output = output
        self.alive = alive
        self.last_messages: list[ChatMessage] | None = None
        self.last_model: str | None = None

    async def chat(
        self,
        messages: list[ChatMessage],
        json_format: bool,
        temperature: float,
        model: str | None = None,
    ) -> str:
        self.last_messages = messages
        self.last_model = model
        return self.output

    async def list_models(self) -> list[str] | None:
        return [self.model, "tiny-test-model"] if self.alive else None


@pytest_asyncio.fixture
async def app():
    application = create_app()
    application.dependency_overrides[get_ollama_client] = lambda: FakeOllama()
    return application


@pytest_asyncio.fixture
async def client(app) -> AsyncClient:
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.API_KEY}
    async with AsyncClient(transport=transport, base_url="http://test", headers=headers) as c:
        yield c
