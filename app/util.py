from __future__ import annotations

import re
from datetime import datetime

_INFO_KEYS_RE = re.compile(
    r"context_length|parameter_count|quantization|block_count|embedding_length"
)


def human_bytes(n: int | float | None) -> str:
    if n is None:
        return "—"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024:
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} PB"


def format_ts(ts: float | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def flatten_info(model_info: dict | None) -> list[tuple[str, str]]:
    """Pick the interesting fields out of an /api/show response for display."""
    mi = model_info or {}
    rows: list[tuple[str, str]] = []

    details = mi.get("details") or {}
    for key in ("family", "format", "parameter_size", "quantization_level"):
        if details.get(key):
            rows.append((key, str(details[key])))

    for key, value in (mi.get("model_info") or {}).items():
        if _INFO_KEYS_RE.search(key):
            rows.append((key, str(value)))

    if mi.get("digest"):
        rows.append(("digest", str(mi["digest"])))
    return rows
