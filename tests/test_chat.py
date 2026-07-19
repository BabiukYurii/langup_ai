import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.schemas.chat import ChatMessage
from app.services.llm.ollama_client import OllamaClient, get_ollama_client
from tests.conftest import FakeOllama

BODY = {
    "messages": [
        {"role": "system", "content": "You output JSON."},
        {"role": "user", "content": "Say hi as JSON."},
    ]
}


async def test_chat_returns_raw_content(client):
    resp = await client.post("/chat", json=BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == '{"ok": true}'
    assert data["model"]


async def test_requires_api_key(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as anon:
        resp = await anon.post("/chat", json=BODY)
    assert resp.status_code == 401


async def test_rejects_empty_messages(client):
    resp = await client.post("/chat", json={"messages": []})
    assert resp.status_code == 422


async def test_rejects_unknown_role(client):
    resp = await client.post("/chat", json={"messages": [{"role": "hacker", "content": "x"}]})
    assert resp.status_code == 422


async def test_accepts_images_for_vision_models(client):
    body = {"messages": [{"role": "user", "content": "What is on the image?", "images": ["aGVsbG8="]}]}
    resp = await client.post("/chat", json=body)
    assert resp.status_code == 200


async def test_model_override(client):
    resp = await client.post("/chat", json={**BODY, "model": "tiny-test-model"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "tiny-test-model"


# --- keep_alive ------------------------------------------------------------


@pytest.fixture
def ollama_payload(monkeypatch):
    """Capture the body OllamaClient actually sends to Ollama."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **kw: real_client(*a, **{**kw, "transport": httpx.MockTransport(handler)}),
    )
    return captured


MESSAGES = [ChatMessage(role="user", content="hi")]


async def test_keep_alive_reaches_ollama(ollama_payload):
    # 0 means "unload as soon as this call is done" — the point of the field
    await OllamaClient().chat(MESSAGES, json_format=True, temperature=0.1, keep_alive=0)
    assert ollama_payload["keep_alive"] == 0


async def test_keep_alive_accepts_a_duration_string(ollama_payload):
    await OllamaClient().chat(MESSAGES, json_format=True, temperature=0.1, keep_alive="10m")
    assert ollama_payload["keep_alive"] == "10m"


async def test_keep_alive_is_omitted_when_not_asked_for(ollama_payload):
    # leaving it out must not override Ollama's own default
    await OllamaClient().chat(MESSAGES, json_format=True, temperature=0.1)
    assert "keep_alive" not in ollama_payload


async def test_request_keep_alive_is_passed_through(app):
    # the whole path: HTTP body -> ChatRequest -> OllamaClient.chat
    fake = FakeOllama()
    app.dependency_overrides[get_ollama_client] = lambda: fake

    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.API_KEY}
    async with AsyncClient(transport=transport, base_url="http://test", headers=headers) as c:
        resp = await c.post("/chat", json={**BODY, "keep_alive": 0})

    assert resp.status_code == 200
    assert fake.last_keep_alive == 0


async def test_rejects_invalid_keep_alive(client):
    resp = await client.post("/chat", json={**BODY, "keep_alive": {"bad": "type"}})
    assert resp.status_code == 422
