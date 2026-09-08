# Ollama GGUF Converter WebUI

A small, dockerized web app that takes a **GGUF download URL**, downloads the
file, imports it into a configured **Ollama** instance (`ollama create`), deletes
the local GGUF afterwards, and keeps a history of every conversion.

Downloads and imports run in a **background worker**, so you can close the browser
and come back later.

## Features

| # | Feature |
|---|---------|
| 1 | Dockerized (`Dockerfile` + `docker-compose.yml`) |
| 2 | Web UI takes any GGUF URL, imports into a selectable Ollama instance |
| 3 | Model named after the GGUF file, or a name/tag you supply |
| 4 | Runs `ollama create` via the Ollama HTTP API (blob upload + `files`) |
| 5 | Deletes the downloaded GGUF after a successful import |
| 6 | Clear "Import successful" state |
| 7 | Password login, argon2 hashing, signed sessions, CSRF, security headers, SSRF protection, login rate-limiting |
| 8 | GitHub Action publishing the image to GitHub Container Registry |
| 9 | Async download + conversion in a separate worker (Redis queue) — no need to keep the page open |
| 10 | Conversion history in the UI with live-updating detail pages |
| 11 | Optional Modelfile options (SYSTEM, TEMPLATE, PARAMETERs) |
| 12 | Final model size + model info (params, quantization, context length, …) shown |

## Architecture

```
browser ──HTTPS──> web (FastAPI/uvicorn) ──enqueue──> Redis ──> worker (arq)
                        │                                          │
                        └──────── SQLite (shared volume) ──────────┘
                                                                   │
                                        download GGUF  ─────────────┤
                                        POST /api/blobs  ───────────┤──> Ollama instance
                                        POST /api/create ───────────┤
                                        GET  /api/show, /api/tags ──┘
```

The GGUF never has to be on the same filesystem as Ollama — it is streamed to the
target instance as a blob, so remote Ollama instances work too.
Requires **Ollama ≥ 0.5** (the `files` parameter of `/api/create`).

## Quick start

```bash
cp .env.example .env
# edit .env: set APP_PASSWORD and SECRET_KEY (openssl rand -hex 32),
# and OLLAMA_INSTANCES

# bring your own Ollama, or start the bundled one:
docker compose --profile ollama up -d      # includes an ollama container
# or
docker compose up -d                       # web + worker + redis only
```

Open <http://localhost:8080> and sign in with `APP_PASSWORD`.

> For local HTTP testing set `COOKIE_SECURE=false` in `.env`, otherwise the
> session cookie is only sent over HTTPS and you can never stay logged in.

## Configuration

All configuration is via environment variables (see `.env.example`). Key ones:

| Variable | Default | Meaning |
|----------|---------|---------|
| `APP_PASSWORD` / `APP_PASSWORD_HASH` | — | **Required.** UI password (plaintext hashed at startup, or a pre-computed argon2 hash) |
| `SECRET_KEY` | ephemeral | Session-cookie signing key. Set it in production |
| `OLLAMA_INSTANCES` | `[{"name":"default","url":OLLAMA_URL}]` | JSON list of `{name,url}` targets shown in the dropdown |
| `COOKIE_SECURE` | `true` | Only send the session cookie over HTTPS |
| `TRUST_PROXY` | `false` | Honour `X-Forwarded-For` (enable only behind a trusted proxy) |
| `ALLOW_PRIVATE_DOWNLOAD_URLS` | `false` | Allow download URLs resolving to private/loopback/link-local IPs (SSRF guard) |
| `MAX_GGUF_SIZE_GB` | `100` | Reject larger downloads |
| `DELETE_GGUF_ON_FAILURE` | `true` | Also remove the partial/complete GGUF when a job fails |
| `HF_TOKEN` | — | Bearer token, sent **only** to `huggingface.co` / `hf.co` hosts |
| `REDIS_URL` | `redis://redis:6379` | Redis connection for the job queue + login rate-limiter |
| `DATA_DIR` | `/data` | SQLite DB + in-flight downloads |

## Security notes

- **Auth**: single shared password, argon2id hashed. Sessions are signed
  (itsdangerous) cookies, `HttpOnly`, `SameSite=Lax`, `Secure` (configurable),
  with an absolute max age.
- **CSRF**: synchroniser token in the session, required on every state-changing
  request (form field or `X-CSRF-Token` header).
- **Brute force**: failed logins are counted per IP in Redis and locked out after
  `LOGIN_MAX_ATTEMPTS` for `LOGIN_LOCKOUT_MINUTES`; every failure also costs ~1s.
- **SSRF**: the server fetches a user-supplied URL. Every redirect hop is
  re-resolved and rejected if it points at a private/loopback/link-local/reserved
  address (including cloud metadata `169.254.169.254`) unless
  `ALLOW_PRIVATE_DOWNLOAD_URLS=true`. Only `http`/`https` are allowed.
- **Input validation**: model names are restricted to a safe character set;
  Modelfile `PARAMETER`s are whitelisted, so arbitrary directives cannot be
  injected. `SYSTEM`/`TEMPLATE` are passed as structured JSON fields, never
  concatenated into a Modelfile.
- **Downloads** are verified to start with the `GGUF` magic bytes and are checked
  against a size limit and available disk space.
- **Headers**: strict `Content-Security-Policy` (no inline JS/CSS, no external
  origins), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, HSTS
  (when `COOKIE_SECURE`).
- The container runs as a non-root user; API docs endpoints are disabled.

Run `pip install -r requirements-dev.txt && pytest` for the validation test suite.

## Publishing the image

`.github/workflows/publish.yml` builds a multi-arch (`amd64`/`arm64`) image and
pushes it to `ghcr.io/<owner>/<repo>` on every push to `main` and every `v*` tag,
authenticating with the built-in `GITHUB_TOKEN` (needs `packages: write`, already
set in the workflow). Tags produced: branch name, `sha-<short>`, semver
(`1.2.3`, `1.2`), and `latest` for the default branch.

To deploy a published image instead of building locally:

```bash
IMAGE=ghcr.io/<owner>/<repo>:latest docker compose up -d
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

export DATA_DIR=./data COOKIE_SECURE=false APP_PASSWORD=dev SECRET_KEY=dev
export REDIS_URL=redis://localhost:6379
export OLLAMA_INSTANCES='[{"name":"local","url":"http://localhost:11434"}]'

uvicorn app.main:app --reload            # terminal 1
arq app.worker.WorkerSettings            # terminal 2  (needs Redis running)
```
