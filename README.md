# Ollama GGUF Converter WebUI

A small, dockerized web app for getting models into Ollama:

- **GGUF** (`/`) — paste a GGUF download URL, it's downloaded and `ollama create`d.
- **Convert from HF** (`/convert`) — paste a non-GGUF Hugging Face repo (safetensors),
  it's converted with llama.cpp, optionally quantized, then imported the same way.

Both delete the local file after a successful import, keep a shared history, run in
background workers (safe to close the browser), and can optionally publish the result
to the Ollama hub.

## Quick start

```bash
cp .env.example .env
# set APP_PASSWORD, SECRET_KEY (openssl rand -hex 32), and OLLAMA_INSTANCES
# in .env - point it at an Ollama you already run, e.g.
# http://host.docker.internal:11434 for a native install

docker compose up -d                                # GGUF page only
docker compose -f docker-compose.convert.yml up -d  # + "Convert from HF" page
```

Each compose file is standalone (no `--profile` needed) and shares the same data.
Open <http://localhost:8080> and sign in with `APP_PASSWORD`. For local HTTP testing
set `COOKIE_SECURE=false` in `.env`.

## Features

- Selectable Ollama instance, with a live online/version check
- Model named after the source, or a name/tag you supply; optional SYSTEM/TEMPLATE/PARAMETERs
- Async pipeline (Redis + arq workers), full conversion history, live progress
- Final model size and info (params, quantization, context length, …)
- Password login, CSRF, security headers, SSRF protection, rate-limited login
- Convert page: safetensors-only download, `convert_hf_to_gguf.py` → optional
  `llama-quantize`, picked quantization level
- Optional, off-by-default publish to the Ollama hub after import
- GitHub Action publishing both images to GHCR

## Configuration

Full list in `.env.example`; the essentials:

| Variable | Meaning |
|----------|---------|
| `APP_PASSWORD` | **Required.** UI login password |
| `SECRET_KEY` | **Required in production.** Session-cookie signing key |
| `OLLAMA_INSTANCES` | JSON `[{"name","url"}, ...]` shown in the instance picker |
| `COOKIE_SECURE` | `true` by default — set `false` only for local HTTP |
| `ALLOW_PRIVATE_DOWNLOAD_URLS` | Keep `false` unless you need it (SSRF guard) |
| `MAX_GGUF_SIZE_GB` / `MAX_HF_REPO_SIZE_GB` | Size limits |
| `HF_TOKEN` | For gated/private Hugging Face downloads |

## Security

Argon2 password hashing, signed sessions, CSRF tokens, per-IP login rate limiting,
strict CSP/security headers, non-root containers. The GGUF page re-validates every
redirect hop against private/loopback/link-local addresses (SSRF); the convert page
only ever downloads `*.safetensors` + config/tokenizer files — no pickled weights, no
custom code execution. Modelfile `PARAMETER`s are whitelisted. See `app/security.py`
and `.env.example` for details. Publishing to the Ollama hub only works if the target
Ollama instance is already signed in (`ollama signin`) — this app holds no hub
credentials of its own.

## Publishing the images

`.github/workflows/publish.yml` builds and pushes both images to GHCR on every push to
`main` and every `v*` tag: `ghcr.io/<owner>/<repo>` (multi-arch) and
`ghcr.io/<owner>/<repo>-converter` (amd64). To deploy published images instead of
building locally:

```bash
IMAGE=ghcr.io/<owner>/<repo>:latest \
CONVERTER_IMAGE=ghcr.io/<owner>/<repo>-converter:latest \
docker compose -f docker-compose.convert.yml up -d
```

## Development

```bash
pip install -r requirements-dev.txt
export DATA_DIR=./data COOKIE_SECURE=false APP_PASSWORD=dev SECRET_KEY=dev
export REDIS_URL=redis://localhost:6379
export OLLAMA_INSTANCES='[{"name":"local","url":"http://localhost:11434"}]'

uvicorn app.main:app --reload            # terminal 1
arq app.worker.WorkerSettings            # terminal 2 (GGUF page)
arq app.converter_worker.WorkerSettings  # terminal 3 (convert page; needs converter/requirements.txt)
```

Tests: `pytest`.
