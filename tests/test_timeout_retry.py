import asyncio
from pathlib import Path

import httpx

from app.downloader import DownloadError, download_gguf
from app.ollama import OllamaClient, OllamaError


def test_download_retries_connect_timeout(monkeypatch):
    attempts = 0

    async def flaky_download(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectTimeout("timed out")
        return 123, "abc"

    monkeypatch.setattr("app.downloader._download_gguf_once", flaky_download)

    size, digest = asyncio.run(
        download_gguf(
            "https://example.com/model.gguf",
            Path("/tmp/model.gguf"),
            allow_private=False,
            max_bytes=100,
            hf_token=None,
            timeout=None,
        )
    )
    assert (size, digest) == (123, "abc")
    assert attempts == 3


def test_download_connect_timeout_returns_download_error(monkeypatch):
    async def timed_out(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr("app.downloader._download_gguf_once", timed_out)

    try:
        asyncio.run(
            download_gguf(
                "https://example.com/model.gguf",
                Path("/tmp/model.gguf"),
                allow_private=False,
                max_bytes=100,
                hf_token=None,
                timeout=None,
            )
        )
    except DownloadError as exc:
        assert "3 attempts" in str(exc)
    else:
        raise AssertionError("Expected DownloadError")


def test_ollama_retries_connect_timeout():
    client = OllamaClient("http://ollama:11434")
    attempts = 0

    async def flaky_request():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectTimeout("timed out")
        return "ok"

    result = asyncio.run(client._retry(flaky_request, "Test request"))
    assert result == "ok"
    assert attempts == 3


def test_ollama_wraps_final_connect_timeout():
    client = OllamaClient("http://ollama:11434")

    async def timed_out():
        raise httpx.ConnectTimeout("timed out")

    try:
        asyncio.run(client._retry(timed_out, "Test request"))
    except OllamaError as exc:
        assert "3 attempts" in str(exc)
    else:
        raise AssertionError("Expected OllamaError")
