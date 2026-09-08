from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

ProgressCb = Callable[[int, int], Awaitable[None]]


class OllamaError(RuntimeError):
    """Raised when an Ollama API call fails."""


class OllamaClient:
    def __init__(self, base_url: str, timeout: float | None = None):
        self.base_url = base_url.rstrip("/")
        # No read timeout for uploads / create (they can take a long time),
        # but keep a sane connect timeout.
        self._long_timeout = httpx.Timeout(timeout, connect=15.0)
        self._short_timeout = httpx.Timeout(60.0, connect=15.0)

    async def version(self) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                if resp.status_code == 200:
                    return resp.json().get("version", "unknown")
        except httpx.HTTPError:
            return None
        return None

    async def blob_exists(self, digest: str) -> bool:
        async with httpx.AsyncClient(timeout=self._short_timeout) as client:
            resp = await client.head(f"{self.base_url}/api/blobs/sha256:{digest}")
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        raise OllamaError(
            f"Unexpected status {resp.status_code} while checking blob presence."
        )

    async def upload_blob(
        self, digest: str, file_path: str | os.PathLike, on_progress: ProgressCb | None = None
    ) -> None:
        total = os.path.getsize(file_path)
        sent = 0

        async def reader() -> AsyncIterator[bytes]:
            nonlocal sent
            with open(file_path, "rb") as handle:
                while True:
                    chunk = handle.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    sent += len(chunk)
                    if on_progress:
                        await on_progress(sent, total)
                    yield chunk

        async with httpx.AsyncClient(timeout=self._long_timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/blobs/sha256:{digest}",
                content=reader(),
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(total),
                },
            )
        if resp.status_code not in (200, 201):
            raise OllamaError(
                f"Blob upload failed ({resp.status_code}): {resp.text[:500]}"
            )

    async def create_model(
        self,
        name: str,
        gguf_filename: str,
        digest: str,
        *,
        system: str | None = None,
        template: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict]:
        payload: dict[str, Any] = {
            "model": name,
            "files": {gguf_filename: f"sha256:{digest}"},
            "stream": True,
        }
        if system:
            payload["system"] = system
        if template:
            payload["template"] = template
        if parameters:
            payload["parameters"] = parameters

        async with httpx.AsyncClient(timeout=self._long_timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/create", json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise OllamaError(f"create failed ({resp.status_code}): {body[:500]}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("error"):
                        raise OllamaError(str(event["error"]))
                    yield event

    async def show(self, name: str) -> dict:
        async with httpx.AsyncClient(timeout=self._short_timeout) as client:
            resp = await client.post(f"{self.base_url}/api/show", json={"model": name})
        if resp.status_code != 200:
            raise OllamaError(f"/api/show failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json()

    async def tags(self) -> dict:
        async with httpx.AsyncClient(timeout=self._short_timeout) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
        if resp.status_code != 200:
            raise OllamaError(f"/api/tags failed ({resp.status_code}).")
        return resp.json()
