from __future__ import annotations

from arq.connections import RedisSettings

from .config import get_settings
from .db import DB
from .downloader import DownloadError, download_gguf
from .import_pipeline import make_progress_writer, cleanup_gguf, import_gguf_into_ollama
from .ollama import OllamaError
from .queues import GGUF_HEALTH_KEY as HEALTH_CHECK_KEY
from .queues import GGUF_QUEUE as QUEUE_NAME


async def _fail(db: DB, job_id: str, message: str) -> None:
    await db.update_job(job_id, status="failed", phase="failed", error=message)
    await db.append_log(job_id, f"FAILED: {message}")


async def convert_model(ctx: dict, job_id: str) -> None:
    """Job kind 'gguf_download': download a ready-made GGUF from a URL and
    import it into Ollama. See import_pipeline.import_gguf_into_ollama for
    everything from the blob upload onward."""
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

    try:
        await db.update_job(job_id, status="downloading", phase="Starting download", progress=0.0)
        await db.append_log(job_id, f"Downloading GGUF from {job['source_url']}")
        size, digest = await download_gguf(
            job["source_url"],
            gguf_path,
            allow_private=settings.allow_private_download_urls,
            max_bytes=settings.max_gguf_bytes,
            hf_token=settings.hf_token,
            timeout=settings.download_timeout_seconds or None,
            on_progress=make_progress_writer(db, job_id, "Downloaded"),
        )
        await db.update_job(
            job_id, download_size=size, gguf_sha256=digest, progress=1.0,
            phase=f"Downloaded {size / 1e9:.2f} GB",
        )
        await db.append_log(job_id, f"Download complete: {size} bytes, sha256={digest}")

        await import_gguf_into_ollama(
            db, job_id,
            instance=instance,
            model_name=model_name,
            options=options,
            gguf_path=gguf_path,
            digest=digest,
            size=size,
            push_to_hub=bool(job.get("push_to_hub")),
            timeout=settings.download_timeout_seconds or None,
        )

    except (DownloadError, OllamaError) as exc:
        if settings.delete_gguf_on_failure:
            cleanup_gguf(gguf_path)
        await _fail(db, job_id, str(exc))
    except Exception as exc:  # noqa: BLE001
        if settings.delete_gguf_on_failure:
            cleanup_gguf(gguf_path)
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
    queue_name = QUEUE_NAME
    health_check_key = HEALTH_CHECK_KEY
    health_check_interval = 30
    max_jobs = 2
    job_timeout = 60 * 60 * 24
    keep_result = 3600
