from app.services.llm.llamacpp_client import get_llm_client
from tests.conftest import FakeLLM


async def test_health_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm"] is True
    assert body["available_models"]  # what the server currently has loaded


async def test_health_degraded_when_llm_down(app, client):
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM(alive=False)
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
