import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.schemas.chat import ChatMessage
from app.services.llm.llamacpp_client import LlamaCppClient, get_llm_client
from tests.conftest import FakeLLM

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


async def test_model_override_is_ignored(client):
    # the server runs one model; an override in the request must not change it
    resp = await client.post("/chat", json={**BODY, "model": "tiny-test-model"})
    assert resp.status_code == 200
    assert resp.json()["model"] != "tiny-test-model"


# --- what actually goes over the wire to llama.cpp -------------------------


@pytest.fixture
def llamacpp_payload(monkeypatch):
    """Capture the body LlamaCppClient sends to the llama.cpp /v1 endpoint."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["__url__"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **kw: real_client(*a, **{**kw, "transport": httpx.MockTransport(handler)}),
    )
    return captured


MESSAGES = [ChatMessage(role="user", content="hi")]


async def test_sends_openai_chat_completions(llamacpp_payload):
    out = await LlamaCppClient().chat(MESSAGES, json_format=True, temperature=0.3)
    assert out == "{}"
    assert llamacpp_payload["__url__"].endswith("/v1/chat/completions")
    assert llamacpp_payload["temperature"] == 0.3
    assert llamacpp_payload["stream"] is False
    assert llamacpp_payload["messages"] == [{"role": "user", "content": "hi"}]


async def test_json_format_sets_response_format(llamacpp_payload):
    await LlamaCppClient().chat(MESSAGES, json_format=True, temperature=0.1)
    assert llamacpp_payload["response_format"] == {"type": "json_object"}


async def test_plain_format_omits_response_format(llamacpp_payload):
    await LlamaCppClient().chat(MESSAGES, json_format=False, temperature=0.1)
    assert "response_format" not in llamacpp_payload


async def test_model_and_keep_alive_never_reach_the_server(llamacpp_payload):
    # both are accepted at the gateway API for compatibility, but a single
    # always-resident model means there is nothing to send downstream
    await LlamaCppClient().chat(MESSAGES, json_format=True, temperature=0.1, model="x", keep_alive=0)
    assert "keep_alive" not in llamacpp_payload
    assert llamacpp_payload["model"] == LlamaCppClient().model  # the server's model, not "x"


# --- the compat fields still flow through the gateway API ------------------


async def test_request_fields_reach_the_client(app):
    # the whole path: HTTP body -> ChatRequest -> LlamaCppClient.chat
    fake = FakeLLM()
    app.dependency_overrides[get_llm_client] = lambda: fake

    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.API_KEY}
    async with AsyncClient(transport=transport, base_url="http://test", headers=headers) as c:
        resp = await c.post("/chat", json={**BODY, "keep_alive": 0})

    assert resp.status_code == 200
    assert fake.last_keep_alive == 0


async def test_rejects_invalid_keep_alive(client):
    resp = await client.post("/chat", json={**BODY, "keep_alive": {"bad": "type"}})
    assert resp.status_code == 422


# --- fenced JSON gets unwrapped -------------------------------------------


def _server_returning(content: str, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **kw: real_client(*a, **{**kw, "transport": httpx.MockTransport(handler)}),
    )


async def test_strips_markdown_json_fence(monkeypatch):
    # Gemma wraps JSON in ```json … ```; the backend needs the bare object
    _server_returning('```json\n{"translation": "берег"}\n```', monkeypatch)
    out = await LlamaCppClient().chat(MESSAGES, json_format=True, temperature=0.1)
    assert out == '{"translation": "берег"}'


async def test_leaves_plain_json_untouched(monkeypatch):
    _server_returning('{"ok": true}', monkeypatch)
    out = await LlamaCppClient().chat(MESSAGES, json_format=True, temperature=0.1)
    assert out == '{"ok": true}'


async def test_does_not_strip_when_not_json_mode(monkeypatch):
    # plain-text mode returns content verbatim, fences and all
    _server_returning("```\nsome text\n```", monkeypatch)
    out = await LlamaCppClient().chat(MESSAGES, json_format=False, temperature=0.1)
    assert out == "```\nsome text\n```"
