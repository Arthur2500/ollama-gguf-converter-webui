from __future__ import annotations

import json
import time
from pathlib import Path

from .config import OllamaInstance
from .db import DB
from .ollama import OllamaClient, OllamaError


def make_progress_writer(db: DB, job_id: str, label: str):
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


def cleanup_gguf(gguf_path: Path) -> None:
    gguf_path.unlink(missing_ok=True)
    gguf_path.with_suffix(gguf_path.suffix + ".part").unlink(missing_ok=True)


async def import_gguf_into_ollama(
    db: DB,
    job_id: str,
    *,
    instance: OllamaInstance,
    model_name: str,
    options: dict,
    gguf_path: Path,
    digest: str,
    size: int,
    push_to_hub: bool,
    timeout: float | None,
) -> None:
    """Shared tail end of both conversion pipelines: upload the GGUF blob to
    Ollama, run `ollama create`, collect model info, delete the local GGUF,
    and (optionally, best-effort) publish to the Ollama hub.

    Raises OllamaError / Exception on failure of the required steps (upload,
    create). Push-to-hub failures are recorded but never fail the job, since
    the model has already been imported successfully at that point.
    """
    client = OllamaClient(instance.url, timeout=timeout)

    await db.update_job(job_id, download_size=size, gguf_sha256=digest)

    # Upload the blob to Ollama --------------------------------------------
    await db.update_job(job_id, status="uploading", phase="Checking blob on Ollama", progress=0.0)
    if await client.blob_exists(digest):
        await db.append_log(job_id, "Blob already present on Ollama; skipping upload.")
    else:
        await client.upload_blob(
            digest, gguf_path, on_progress=make_progress_writer(db, job_id, "Uploaded"),
        )
        await db.append_log(job_id, "Blob uploaded to Ollama.")
    await db.update_job(job_id, progress=1.0)

    # ollama create -----------------------------------------------------
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

    # Collect final info --------------------------------------------------
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
        # Ollama defaults an untagged model to ":latest" - normalize both
        # sides before comparing, or a no-tag lookup never matches.
        target = model_name if ":" in model_name else f"{model_name}:latest"
        for entry in (await client.tags()).get("models", []):
            entry_name = entry.get("name") or entry.get("model") or ""
            if not (":" in entry_name):
                entry_name = f"{entry_name}:latest"
            if entry_name == target:
                final_size = entry.get("size")
                info["digest"] = entry.get("digest")
                break
    except Exception as exc:  # noqa: BLE001 - info is best-effort
        await db.append_log(job_id, f"Could not read /api/tags: {exc}")

    # Delete the local GGUF ------------------------------------------------
    cleanup_gguf(gguf_path)
    await db.append_log(job_id, "Deleted local GGUF file.")

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

    # Optional, best-effort: publish to the Ollama hub ---------------------
    if push_to_hub:
        await db.update_job(job_id, push_status="publishing")
        await db.append_log(job_id, f"Publishing {model_name} to the Ollama hub...")
        try:
            async for event in client.push_model(model_name):
                status_line = event.get("status") if isinstance(event, dict) else None
                if status_line:
                    await db.append_log(job_id, f"push: {status_line}")
            await db.update_job(job_id, push_status="success", push_error=None)
            await db.append_log(job_id, "Published to the Ollama hub.")
        except OllamaError as exc:
            await db.update_job(job_id, push_status="failed", push_error=str(exc))
            await db.append_log(
                job_id,
                f"Hub publish failed (model import itself still succeeded): {exc}. "
                "Is this Ollama instance signed in (`ollama signin` / OLLAMA_API_KEY)?",
            )
