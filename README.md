# LangUp AI

Thin **inference gateway** in front of a local [Ollama](https://ollama.com) server.
Runs on the mini-PC; the main LangUp backend calls it over HTTP with an `X-API-Key`.

**No domain logic lives here.** Prompts, output parsing and validation belong to
the main backend (`langup_backend/app/services/ai/`). This service only provides:
authentication, a stable minimal API surface in front of Ollama (which has no
auth of its own), health reporting, and a single place for future model
management / queueing.

## Architecture

```
app/
  core/        config, security (API key), exceptions
  schemas/     ChatRequest/ChatResponse
  services/
    llm/       OllamaClient (async httpx -> /api/chat)
  routers/     /health, /chat
  main.py      create_app() factory
tests/         pytest, Ollama faked — no model needed
```

## Setup

1. Install Ollama and pull the model:
   `ollama pull qwen2.5vl:7b` (vision-language: text + images)
2. `cp .env.sample .env` and set a real `API_KEY`.
3. `uv sync`

## Commands

- Run: `uv run python -m app.main` → http://localhost:8100/docs
- Tests: `uv run pytest -q` (no Ollama required)
- Lint: `uv run ruff check . --fix && uv run ruff format .`

## Docker / server deploy

`docker compose up -d --build` starts two containers: the gateway (host port
**8002**, per the server-config port convention) and Ollama (internal only, not
exposed to the host; models cached in the `ollama_models` volume). On start the
ollama container pulls `OLLAMA_MODEL` from `.env` if missing — **switching models
= edit `.env`, `docker compose up -d`**. First boot downloads ~5 GB.

On the NucBox server (see the `server-config` repo): clone into
`~/Desktop/apps/langup_ai`, scp the `.env`, add an ingress rule
`ai.piatek-magazyn.com → http://localhost:8002` + DNS route, and `startup.sh`
manages it from the next boot.

## API

- `GET /health` — service + Ollama reachability.
- `POST /chat` (X-API-Key) — `{messages, json_format, temperature, model?}` →
  `{content, model}` where `content` is the raw model output string.
  A message may carry `images: [<base64>, ...]` for vision-language models.
