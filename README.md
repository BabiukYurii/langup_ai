# LangUp AI

Thin **inference gateway** in front of a local [llama.cpp](https://github.com/ggml-org/llama.cpp)
server. Runs on the mini-PC; the main LangUp backend calls it over HTTP with an
`X-API-Key`.

**No domain logic lives here.** Prompts, output parsing and validation belong to
the main backend (`langup_backend/app/services/ai/`). This service only provides:
authentication, a stable minimal API surface in front of llama.cpp, health
reporting, and a single place for future model management / queueing.

The llama.cpp server runs **Gemma-4-26B-A4B** (a MoE: 26B total, ~4B active) on
the AMD **Radeon 780M iGPU via Vulkan** (RADV) — no ROCm needed. The weights live
in the iGPU's dedicated VRAM carveout, so they do not consume the system RAM the
other containers on the box need. Gemma-4's thinking mode is turned OFF
(`--reasoning-budget 0` in docker-compose): left on, it burns ~4000 reasoning
tokens per exercise (~150 s); off, generation is a few seconds with no quality
loss on our JSON tasks.

## Architecture

```
app/
  core/        config, security (API key), exceptions
  schemas/     ChatRequest/ChatResponse
  services/
    llm/       LlamaCppClient (async httpx -> /v1/chat/completions)
  routers/     /health, /chat
  main.py      create_app() factory
tests/         pytest, the LLM faked — no model needed
```

## Setup

1. `cp .env.sample .env`, set a real `API_KEY`, pick `LLAMACPP_HF_REPO`.
2. `uv sync`
3. Run the model server (or use Docker below):
   `llama-server -hf <repo> -ngl 99 --port 8080 --jinja`

## Commands

- Run: `uv run python -m app.main` → http://localhost:8100/docs
- Tests: `uv run pytest -q` (no model required)
- Lint: `uv run ruff check . --fix && uv run ruff format .`

## Docker / server deploy

Pre-download the GGUF once (a 14 GB pull at container start is unreliable):

```
wget -c -P models https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/resolve/main/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
```

`docker compose up -d --build` then starts two containers: the gateway (host port
**8002**, per the server-config port convention) and the llama.cpp server
(internal only, not exposed to the host). The llama.cpp container has `/dev/dri`
passed through for Vulkan and serves `models/${LLAMACPP_MODEL_FILE}` on the GPU —
**switching models = drop a new GGUF in `models/`, edit `.env`,
`docker compose up -d`**.

The model must fit the iGPU VRAM carveout (set in BIOS "UMA Frame Buffer"):
gemma-4-26B-A4B at QAT-Q4 is ~14 GB and fits the 16 GB carveout with room for the
KV cache (it runs at ~15.6/16 GB — tight but stable).

On the NucBox server (see the `server-config` repo): clone into
`~/Desktop/apps/langup_ai`, scp the `.env`, add an ingress rule
`ai.piatek-magazyn.com → http://localhost:8002` + DNS route, and `startup.sh`
manages it from the next boot.

## API

- `GET /health` — service + llama.cpp reachability.
- `POST /chat` (X-API-Key) — `{messages, json_format, temperature, model?}` →
  `{content, model}` where `content` is the raw model output string.
  `model` and `keep_alive` are accepted for backward compatibility but ignored:
  the server runs a single, always-resident model.
