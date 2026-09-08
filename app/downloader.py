from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urljoin, urlparse

import httpx

from .security import validate_download_url

GGUF_MAGIC = b"GGUF"
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_CHUNK = 4 * 1024 * 1024

ProgressCb = Callable[[int, int | None], Awaitable[None]]


class DownloadError(RuntimeError):
    """Raised when a GGUF download fails or the file is not a GGUF."""


def _hf_headers(url: str, hf_token: str | None) -> dict[str, str]:
    if not hf_token:
        return {}
    host = (urlparse(url).hostname or "").lower()
    if host in ("huggingface.co", "hf.co") or host.endswith((".huggingface.co", ".hf.co")):
        return {"Authorization": f"Bearer {hf_token}"}
    return {}


async def download_gguf(
    url: str,
    dest: Path,
    *,
    allow_private: bool,
    max_bytes: int,
    hf_token: str | None,
    timeout: float | None,
    on_progress: ProgressCb | None = None,
) -> tuple[int, str]:
    """Stream ``url`` to ``dest``. Redirects are followed manually and every
    hop is re-validated against the SSRF policy. Returns ``(size, sha256hex)``."""
    current = url
    headers = _hf_headers(current, hf_token)
    hops = 0
    client_timeout = httpx.Timeout(timeout, connect=30.0)
    part = dest.with_suffix(dest.suffix + ".part")

    async with httpx.AsyncClient(follow_redirects=False, timeout=client_timeout) as client:
        while True:
            validate_download_url(current, allow_private)
            async with client.stream("GET", current, headers=headers) as resp:
                if resp.status_code in _REDIRECT_CODES:
                    location = resp.headers.get("Location")
                    if not location:
                        raise DownloadError("Redirect response without a Location header.")
                    hops += 1
                    if hops > 10:
                        raise DownloadError("Too many redirects.")
                    current = urljoin(current, location)
                    headers = _hf_headers(current, hf_token)
                    continue

                if resp.status_code != 200:
                    raise DownloadError(f"Download failed: HTTP {resp.status_code}.")

                declared = int(resp.headers.get("Content-Length", "0")) or None
                if declared and declared > max_bytes:
                    raise DownloadError(
                        f"File is ~{declared / 1e9:.1f} GB, over the configured limit."
                    )
                if declared:
                    free = shutil.disk_usage(dest.parent).free
                    if declared + (1 << 30) > free:
                        raise DownloadError("Not enough free disk space for this download.")

                hasher = hashlib.sha256()
                written = 0
                checked_magic = False
                try:
                    with open(part, "wb") as handle:
                        async for chunk in resp.aiter_bytes(_CHUNK):
                            if not checked_magic:
                                if not chunk.startswith(GGUF_MAGIC):
                                    raise DownloadError(
                                        "Downloaded data is not a GGUF file "
                                        "(missing 'GGUF' magic bytes)."
                                    )
                                checked_magic = True
                            written += len(chunk)
                            if written > max_bytes:
                                raise DownloadError("Download exceeded the configured size limit.")
                            handle.write(chunk)
                            hasher.update(chunk)
                            if on_progress:
                                await on_progress(written, declared)
                except BaseException:
                    part.unlink(missing_ok=True)
                    raise

                if not checked_magic:
                    part.unlink(missing_ok=True)
                    raise DownloadError("Downloaded file was empty.")

                part.replace(dest)
                return written, hasher.hexdigest()
