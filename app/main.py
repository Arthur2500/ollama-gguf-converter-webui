from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from . import auth
from .config import get_settings
from .db import DB, new_job_id
from .ollama import OllamaClient
from .security import (
    ALLOWED_PARAM_KEYS,
    ValidationError,
    model_name_from_url,
    parse_parameters,
    validate_download_url,
    validate_model_name,
)
from .util import flatten_info, format_ts, human_bytes

log = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).parent
settings = get_settings()
db = DB(settings.db_path)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.filters["bytes"] = human_bytes
templates.env.filters["dt"] = format_ts

PUBLIC_PATHS = ("/login", "/static", "/healthz", "/favicon.ico")
ALLOWED_PARAMS_HINT = ", ".join(sorted(ALLOWED_PARAM_KEYS))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (settings.app_password or settings.app_password_hash):
        raise RuntimeError("Set APP_PASSWORD or APP_PASSWORD_HASH before starting.")
    app.state.pwd_hash = settings.app_password_hash or auth.hash_password(settings.app_password)

    if not settings.secret_key:
        log.warning("SECRET_KEY is not set - using an ephemeral key; sessions reset on restart.")

    try:
        names = [i.name for i in settings.instances]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Invalid OLLAMA_INSTANCES: {exc}") from exc
    log.info("Configured Ollama instances: %s", ", ".join(names))

    await db.init()
    app.state.arq = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        yield
    finally:
        try:
            await app.state.arq.aclose()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(
    title="Ollama GGUF Converter",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# --------------------------------------------------------------------------- #
# middleware  (added last == outermost, so SessionMiddleware runs first)
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path
    if any(path == p or path.startswith(p + "/") for p in PUBLIC_PATHS):
        return await call_next(request)
    if not request.session.get("auth"):
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Authentication required."}, status_code=401)
        return RedirectResponse("/login", status_code=303)
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; form-action 'self'; "
        "frame-ancestors 'none'; base-uri 'none'"
    )
    if settings.cookie_secure:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


app.add_middleware(
    SessionMiddleware,
    secret_key=settings.effective_secret_key,
    max_age=settings.session_max_age,
    same_site="lax",
    https_only=settings.cookie_secure,
    session_cookie="oggc_session",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _client_ip(request: Request) -> str:
    if settings.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_password(request: Request, plain: str) -> bool:
    return auth.verify_password(request.app.state.pwd_hash, plain)


async def _csrf(request: Request, form_token: str | None) -> None:
    token = form_token or request.headers.get("x-csrf-token")
    auth.verify_csrf(request, token)


async def _index_context(request: Request, error: str | None = None) -> dict:
    return {
        "request": request,
        "authed": True,
        "csrf_token": auth.get_or_create_csrf(request),
        "instances": settings.instances,
        "jobs": await db.list_jobs(limit=100),
        "allowed_params": ALLOWED_PARAMS_HINT,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# public routes
# --------------------------------------------------------------------------- #
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("auth"):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(
        request, "login.html",
        {"csrf_token": auth.get_or_create_csrf(request), "error": None, "authed": False},
    )


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, password: str = Form(...), csrf_token: str = Form("")):
    auth.verify_csrf(request, csrf_token)
    ip = _client_ip(request)
    redis = request.app.state.arq
    lockout = settings.login_lockout_minutes * 60

    def _render(message: str, code: int):
        return templates.TemplateResponse(
            request, "login.html",
            {"csrf_token": auth.get_or_create_csrf(request), "error": message, "authed": False},
            status_code=code,
        )

    try:
        await auth.check_login_rate_limit(redis, ip, settings.login_max_attempts)
    except Exception:  # HTTPException 429
        return _render("Too many failed attempts. Please wait and try again.", 429)

    if not _check_password(request, password):
        await auth.register_login_failure(redis, ip, lockout)
        await asyncio.sleep(1.0)
        return _render("Incorrect password.", 401)

    await auth.clear_login_failures(redis, ip)
    request.session.clear()
    request.session["auth"] = True
    request.session["ts"] = time.time()
    auth.get_or_create_csrf(request)
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# --------------------------------------------------------------------------- #
# authenticated HTML routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", await _index_context(request))


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str):
    job = await db.get_job(job_id)
    if job is None:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(
        request, "job.html",
        {
            "request": request,
            "authed": True,
            "csrf_token": auth.get_or_create_csrf(request),
            "job": job,
            "info_rows": flatten_info(job.get("model_info")),
        },
    )


# --------------------------------------------------------------------------- #
# JSON API
# --------------------------------------------------------------------------- #
@app.get("/api/instances")
async def api_instances():
    async def probe(inst):
        version = await OllamaClient(inst.url).version()
        return {"name": inst.name, "url": inst.url, "reachable": version is not None, "version": version}

    return {"instances": await asyncio.gather(*(probe(i) for i in settings.instances))}


@app.get("/api/jobs")
async def api_jobs():
    return {"jobs": await db.list_jobs(limit=100)}


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = await db.get_job(job_id)
    if job is None:
        return JSONResponse({"detail": "Not found."}, status_code=404)
    return job


@app.post("/api/jobs")
async def api_create_job(
    request: Request,
    source_url: str = Form(...),
    instance: str = Form(...),
    name: str = Form(""),
    tag: str = Form(""),
    system: str = Form(""),
    template: str = Form(""),
    parameters: str = Form(""),
    csrf_token: str = Form(""),
):
    await _csrf(request, csrf_token)
    instance_names = {i.name for i in settings.instances}

    try:
        if instance not in instance_names:
            raise ValidationError("Unknown Ollama instance.")

        model_name = validate_model_name(name) if name.strip() else model_name_from_url(source_url)
        tag = tag.strip().lower()
        if tag:
            if ":" in model_name:
                raise ValidationError("Provide the tag either in the name or the tag field, not both.")
            model_name = validate_model_name(f"{model_name}:{tag}")

        await run_in_threadpool(
            validate_download_url, source_url, settings.allow_private_download_urls
        )
        parsed_params = parse_parameters(parameters)
    except ValidationError as exc:
        return templates.TemplateResponse(
            request, "index.html", await _index_context(request, str(exc)), status_code=400
        )

    options = {
        "system": system.strip(),
        "template": template.strip(),
        "parameters": parsed_params,
    }
    job_id = new_job_id()
    now = time.time()
    await db.create_job(
        {
            "id": job_id,
            "created_at": now,
            "updated_at": now,
            "source_url": source_url.strip(),
            "model_name": model_name,
            "instance_name": instance,
            "status": "queued",
            "phase": "Queued",
            "progress": 0.0,
            "options": json.dumps(options),
        }
    )
    try:
        await request.app.state.arq.enqueue_job("convert_model", job_id)
    except Exception as exc:  # noqa: BLE001
        await db.update_job(job_id, status="failed", phase="failed",
                            error=f"Could not queue job: {exc}")

    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/api/jobs/{job_id}/retry")
async def api_retry_job(request: Request, job_id: str, csrf_token: str = Form("")):
    await _csrf(request, csrf_token)
    job = await db.get_job(job_id)
    if job is None:
        return RedirectResponse("/", status_code=303)

    new_id = new_job_id()
    now = time.time()
    await db.create_job(
        {
            "id": new_id,
            "created_at": now,
            "updated_at": now,
            "source_url": job["source_url"],
            "model_name": job["model_name"],
            "instance_name": job["instance_name"],
            "status": "queued",
            "phase": "Queued",
            "progress": 0.0,
            "options": json.dumps(job.get("options") or {}),
        }
    )
    try:
        await request.app.state.arq.enqueue_job("convert_model", new_id)
    except Exception as exc:  # noqa: BLE001
        await db.update_job(new_id, status="failed", phase="failed",
                            error=f"Could not queue job: {exc}")
    return RedirectResponse(f"/jobs/{new_id}", status_code=303)


@app.post("/api/jobs/{job_id}/delete")
async def api_delete_job(request: Request, job_id: str, csrf_token: str = Form("")):
    await _csrf(request, csrf_token)
    await db.delete_job(job_id)
    return RedirectResponse("/", status_code=303)
