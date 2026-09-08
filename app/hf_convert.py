from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from .config import get_settings
from .db import DB
from .import_pipeline import import_gguf_into_ollama
from .ollama import OllamaError
from .security import HF_ALLOW_PATTERNS

# Vendored at build time from the llama.cpp repo - see Dockerfile.converter.
CONVERT_SCRIPT = Path("/opt/llamacpp/convert_hf_to_gguf.py")
QUANTIZE_BIN = Path("/opt/llamacpp/build/bin/llama-quantize")


class ConversionError(RuntimeError):
    """Raised when the HF->GGUF conversion or quantization step fails."""


async def _fail(db: DB, job_id: str, message: str) -> None:
    await db.update_job(job_id, status="failed", phase="failed", error=message)
    await db.append_log(job_id, f"FAILED: {message}")


def _dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    return total


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _matches_allowed(filename: str) -> bool:
    return any(fnmatch(filename, pattern) for pattern in HF_ALLOW_PATTERNS)


async def _watch_dir_size(db: DB, job_id: str, path: Path, expected_total: int | None,
                          label: str, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        size = await asyncio.to_thread(_dir_size, path)
        frac = (size / expected_total) if expected_total else 0.0
        suffix = f" / {expected_total / 1e9:.2f} GB ({frac * 100:.0f}%)" if expected_total else " GB"
        await db.update_job(job_id, phase=f"{label} {size / 1e9:.2f}{suffix}",
                            progress=min(frac, 1.0) if expected_total else 0.0)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


async def _run_subprocess(db: DB, job_id: str, args: list[str], *, phase_prefix: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    last_flush = 0.0
    last_line = ""
    async for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").rstrip()
        if not line:
            continue
        last_line = line
        now = time.time()
        if now - last_flush > 1.5:
            last_flush = now
            await db.update_job(job_id, phase=f"{phase_prefix}: {line[:200]}")
    code = await proc.wait()
    if last_line:
        await db.append_log(job_id, f"{phase_prefix}: {last_line[:500]}")
    if code != 0:
        raise ConversionError(
            f"{phase_prefix} failed (exit code {code}). See the log for the last output line."
        )


async def convert_hf_repo(ctx: dict, job_id: str) -> None:
    """Job kind 'hf_convert': download a non-GGUF Hugging Face repo, convert
    it to GGUF with llama.cpp's convert_hf_to_gguf.py, optionally quantize
    it, then hand off to the same import pipeline as the GGUF-download page."""
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
    repo_id: str = job["hf_repo"]
    revision: str = job["hf_revision"] or "main"
    quant: str = job["quant_level"] or "Q4_K_M"

    repo_dir = settings.downloads_dir / f"hf-{job_id}"
    f16_path = settings.downloads_dir / f"{job_id}-f16.gguf"
    final_path = settings.downloads_dir / f"{job_id}.gguf"

    def cleanup_all() -> None:
        shutil.rmtree(repo_dir, ignore_errors=True)
        f16_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)

    try:
        # 1. Resolve + validate repo metadata ----------------------------
        await db.update_job(job_id, status="fetching_metadata",
                            phase=f"Looking up {repo_id}@{revision}", progress=0.0)
        api = HfApi(token=settings.hf_token or None)
        info = await asyncio.to_thread(api.model_info, repo_id, revision=revision, files_metadata=True)

        siblings = info.siblings or []
        if not any(s.rfilename.endswith(".safetensors") for s in siblings):
            raise ConversionError(
                "No .safetensors weights found in this repo. Pickled (.bin/.pt) "
                "weights and repos that require custom modelling code are not "
                "supported, for security reasons."
            )
        known_sizes = [s.size for s in siblings if s.size and _matches_allowed(s.rfilename)]
        expected_total = sum(known_sizes) if known_sizes else None
        if expected_total and expected_total > settings.max_hf_repo_bytes:
            raise ConversionError(
                f"Repo is ~{expected_total / 1e9:.1f} GB, over the configured limit "
                f"({settings.max_hf_repo_size_gb:.0f} GB)."
            )
        if expected_total:
            free = shutil.disk_usage(settings.downloads_dir).free
            if expected_total * 1.2 > free:
                raise ConversionError("Not enough free disk space for this download.")
        await db.append_log(job_id, f"Resolved {repo_id}@{revision}.")

        # 2. Download the (filtered, safetensors-only) file set -----------
        await db.update_job(job_id, status="downloading", phase="Starting download", progress=0.0)
        await db.append_log(job_id, f"Downloading {repo_id}@{revision} from Hugging Face")
        stop_event = asyncio.Event()
        watcher = asyncio.create_task(
            _watch_dir_size(db, job_id, repo_dir, expected_total, "Downloaded", stop_event)
        )
        try:
            await asyncio.to_thread(
                snapshot_download,
                repo_id=repo_id,
                revision=revision,
                local_dir=str(repo_dir),
                allow_patterns=HF_ALLOW_PATTERNS,
                token=settings.hf_token or None,
            )
        finally:
            stop_event.set()
            await watcher
        size_on_disk = await asyncio.to_thread(_dir_size, repo_dir)
        await db.update_job(job_id, progress=1.0, phase=f"Downloaded {size_on_disk / 1e9:.2f} GB")
        await db.append_log(job_id, f"Download complete: {size_on_disk} bytes")

        # 3. HF -> GGUF (f16) ---------------------------------------------
        await db.update_job(job_id, status="converting", phase="Converting to GGUF (f16)", progress=0.0)
        await _run_subprocess(
            db, job_id,
            ["python3", str(CONVERT_SCRIPT), "--outfile", str(f16_path), "--outtype", "f16", str(repo_dir)],
            phase_prefix="convert",
        )
        if not f16_path.exists():
            raise ConversionError("Conversion finished but no GGUF file was produced.")
        await db.append_log(job_id, "HF -> GGUF (f16) conversion complete.")

        # 4. Quantize, unless the user asked for plain F16 -----------------
        if quant == "F16":
            gguf_path = f16_path
        else:
            await db.update_job(job_id, status="quantizing", phase=f"Quantizing to {quant}", progress=0.0)
            await _run_subprocess(
                db, job_id,
                [str(QUANTIZE_BIN), str(f16_path), str(final_path), quant],
                phase_prefix="quantize",
            )
            if not final_path.exists():
                raise ConversionError("Quantization finished but no output file was produced.")
            f16_path.unlink(missing_ok=True)
            gguf_path = final_path
            await db.append_log(job_id, f"Quantized to {quant}.")

        # 5. Hash + size, then drop the HF source files ---------------------
        await db.update_job(job_id, phase="Hashing GGUF")
        digest = await asyncio.to_thread(_sha256_file, gguf_path)
        size = gguf_path.stat().st_size
        shutil.rmtree(repo_dir, ignore_errors=True)

        # 6. Shared import + optional hub publish ---------------------------
        await import_gguf_into_ollama(
            db, job_id,
            instance=instance,
            model_name=model_name,
            options=options,
            gguf_path=gguf_path,
            digest=digest,
            size=size,
            push_to_hub=bool(job.get("push_to_hub")),
            timeout=None,
        )

    except (ConversionError, OllamaError, HfHubHTTPError) as exc:
        cleanup_all()
        await _fail(db, job_id, str(exc))
    except Exception as exc:  # noqa: BLE001
        cleanup_all()
        await _fail(db, job_id, f"Unexpected error: {exc!r}")
