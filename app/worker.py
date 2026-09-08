from __future__ import annotations

import json
import time
from pathlib import Path

from arq.connections import RedisSettings

from .config import get_settings
from .db import DB
from .downloader import DownloadError, download_gguf
from .ollama import OllamaClient, OllamaError


def _make_progress_writer(db: DB, job_id: str, label: str):
    state = {"last": 0.0}

    async def callback(done: int, total: int | None) -> None:
        now = time.time()
        frac = (done / total) if total else 0.0
        if now - state["last"] < 1.5 and frac < 1.0:
            return
        state["last"] = now
        if total:
            phase = f"{label} {done / 1e9:.2f} / {total / 1e9:.2f} GB ({frac * 100:.0f}%)"
        else:
            phase = f"{label} {done / 1e9:.2f} GB"
        await db.update_job(job_id, phase=phase, progress=min(frac, 1.0))

    return callback


def render_modelfile(model_name: str, options: dict) -> str:
    basename = model_name.split(":", 1)[0].split("/")[-1]
    lines = [f"FROM ./{basename}.gguf"]
    if options.get("template"):
        lines.append(f'TEMPLATE """{options["template"]}"""')
    if options.get("system"):
        lines.append(f'SYSTEM """{options["system"]}"""')
    for key, value in (options.get("parameters") or {}).items():
        if isinstance(value, list):
            for item in value:
                lines.append(f"PARAMETER {key} {item}")
        else:
            lines.append(f"PARAMETER {key} {value}")
    return "\n".join(lines)


async def _fail(db: DB, job_id: str, message: str) -> None:
    await db.update_job(job_id, status="failed", phase="failed", error=message)
    await db.append_log(job_id, f"FAILED: {message}")


def _cleanup(gguf_path: Path, *, always: bool, delete_on_failure: bool) -> None:
    if always or delete_on_failure:
        gguf_path.unlink(missing_ok=True)
        gguf_path.with_suffix(gguf_path.suffix + ".part").unlink(missing_ok=True)


async def convert_model(ctx: dict, job_id: str) -> None:
    settings = get_settings()
    db: DB = ctx["db"]

    job = await db.get_job(job_id)
    if not job or job["status"] in ("success", "failed"):
        return

    instances = {i.name: i for i in settings.instances}
    instance = instances.get(job["instance_name"])
    if instance is None:
        await _fail(db, job_id, f"Unknown Ollama instance '{job['instance_name']}'.")
        return

    options: dict = job.get("options") or {}
    model_name: str = job["model_name"]
    gguf_path = settings.downloads_dir / f"{job_id}.gguf"
    client = OllamaClient(instance.url, timeout=settings.download_timeout_seconds or None)

    try:
        # 1. Download ----------------------------------------------------------
        await db.update_job(job_id, status="downloading", phase="Starting download", progress=0.0)
        await db.append_log(job_id, f"Downloading GGUF from {job['source_url']}")
        size, digest = await download_gguf(
            job["source_url"],
            gguf_path,
            allow_private=settings.allow_private_download_urls,
            max_bytes=settings.max_gguf_bytes,
            hf_token=settings.hf_token,
            timeout=settings.download_timeout_seconds or None,
            on_progress=_make_progress_writer(db, job_id, "Downloaded"),
        )
        await db.update_job(
            job_id, download_size=size, gguf_sha256=digest, progress=1.0,
            phase=f"Downloaded {size / 1e9:.2f} GB",
        )
        await db.append_log(job_id, f"Download complete: {size} bytes, sha256={digest}")

        # 2. Upload the blob to Ollama ---------------------------------------
        await db.update_job(job_id, status="uploading", phase="Checking blob on Ollama", progress=0.0)
        if await client.blob_exists(digest):
            await db.append_log(job_id, "Blob already present on Ollama; skipping upload.")
        else:
            await client.upload_blob(
                digest, gguf_path, on_progress=_make_progress_writer(db, job_id, "Uploaded"),
            )
            await db.append_log(job_id, "Blob uploaded to Ollama.")
        await db.update_job(job_id, progress=1.0)

        # 3. ollama create --------------------------------------------------
        await db.update_job(job_id, status="creating", phase="Running ollama create", progress=0.0)
        basename = model_name.split(":", 1)[0].split("/")[-1]
        async for event in client.create_model(
            model_name,
            f"{basename}.gguf",
            digest,
            system=options.get("system") or None,
            template=options.get("template") or None,
            parameters=options.get("parameters") or None,
        ):
            status_line = event.get("status") if isinstance(event, dict) else None
            if status_line:
                await db.update_job(job_id, phase=f"create: {status_line}")
        await db.append_log(job_id, "Model created successfully.")

        # 4. Collect final info -------------------------------------------
        await db.update_job(job_id, status="collecting_info", phase="Collecting model information", progress=0.0)
        info: dict = {}
        modelfile = render_modelfile(model_name, options)
        try:
            shown = await client.show(model_name)
            info["details"] = shown.get("details", {})
            info["model_info"] = shown.get("model_info", {})
            info["parameters"] = shown.get("parameters", "")
            info["template"] = shown.get("template", "")
            if shown.get("modelfile"):
                modelfile = shown["modelfile"]
        except Exception as exc:  # noqa: BLE001 - info is best-effort
            await db.append_log(job_id, f"Could not read /api/show: {exc}")

        final_size = None
        try:
            for entry in (await client.tags()).get("models", []):
                if model_name in (entry.get("name"), entry.get("model")):
                    final_size = entry.get("size")
                    info["digest"] = entry.get("digest")
                    break
        except Exception as exc:  # noqa: BLE001 - info is best-effort
            await db.append_log(job_id, f"Could not read /api/tags: {exc}")

        # 5. Delete the local GGUF ---------------------------------------
        _cleanup(gguf_path, always=True, delete_on_failure=True)
        await db.append_log(job_id, "Deleted local GGUF file.")

        # 6. Done -------------------------------------------------------
        await db.update_job(
            job_id,
            status="success",
            phase="Import successful",
            progress=1.0,
            error=None,
            final_size=final_size,
            model_info=json.dumps(info),
            modelfile=modelfile,
        )
        await db.append_log(job_id, "Import successful.")

    except (DownloadError, OllamaError) as exc:
        _cleanup(gguf_path, always=False, delete_on_failure=settings.delete_gguf_on_failure)
        await _fail(db, job_id, str(exc))
    except Exception as exc:  # noqa: BLE001
        _cleanup(gguf_path, always=False, delete_on_failure=settings.delete_gguf_on_failure)
        await _fail(db, job_id, f"Unexpected error: {exc!r}")


async def _startup(ctx: dict) -> None:
    settings = get_settings()
    db = DB(settings.db_path)
    await db.init()
    await db.mark_interrupted()
    ctx["db"] = db


class WorkerSettings:
    functions = [convert_model]
    on_startup = _startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 2
    job_timeout = 60 * 60 * 24
    keep_result = 3600
