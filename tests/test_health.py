from app.services.llm.ollama_client import get_ollama_client
from tests.conftest import FakeOllama


async def test_health_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ollama"] is True
    assert body["available_models"]  # lists pulled models for experimenting


async def test_health_degraded_when_ollama_down(app, client):
    app.dependency_overrides[get_ollama_client] = lambda: FakeOllama(alive=False)
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
