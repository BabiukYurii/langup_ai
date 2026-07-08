from httpx import ASGITransport, AsyncClient

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
