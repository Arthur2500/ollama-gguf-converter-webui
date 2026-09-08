from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

MODEL_NAME_RE = re.compile(
    r"^[a-z0-9]([a-z0-9._-]*[a-z0-9])?"          # model
    r"(/[a-z0-9]([a-z0-9._-]*[a-z0-9])?)?"       # optional /namespace-part
    r"(:[a-z0-9]([a-z0-9._-]*[a-z0-9])?)?$"      # optional :tag
)

# Ollama Modelfile PARAMETER keys we accept. Anything else is rejected so a
# user cannot smuggle arbitrary directives through the options field.
# Kept in sync with the `Options`/`Runner` structs in ollama `api/types.go`.
# (mirostat*, tfs_z and penalize_newline were removed from Ollama and are no
# longer accepted, so they are intentionally not listed here.)
ALLOWED_PARAM_KEYS = {
    "num_ctx", "num_batch", "num_gpu", "main_gpu", "num_thread",
    "draft_num_predict", "num_keep", "seed", "num_predict", "top_k", "top_p",
    "min_p", "typical_p", "repeat_last_n", "temperature", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "stop",
}


class ValidationError(ValueError):
    """Raised for user-supplied input that fails validation."""


# --------------------------------------------------------------------------- #
# model names
# --------------------------------------------------------------------------- #
def validate_model_name(name: str) -> str:
    name = (name or "").strip().lower()
    if not name or len(name) > 128 or ".." in name:
        raise ValidationError("Invalid model name.")
    if not MODEL_NAME_RE.match(name):
        raise ValidationError(
            "Model name may contain lowercase letters, digits, '.', '_', '-', an "
            "optional single '/' namespace separator and an optional ':tag'."
        )
    return name


def model_name_from_url(url: str) -> str:
    path = urlparse(url).path
    base = path.rsplit("/", 1)[-1] or "model"
    base = re.sub(r"\.gguf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[^a-zA-Z0-9._-]+", "-", base)
    base = re.sub(r"-+", "-", base).strip("-._").lower()
    return base or "model"


# --------------------------------------------------------------------------- #
# download URL / SSRF protection
# --------------------------------------------------------------------------- #
def _address_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValidationError(f"Could not resolve host '{host}'.") from exc
    addrs = sorted({info[4][0] for info in infos})
    if not addrs:
        raise ValidationError(f"Could not resolve host '{host}'.")
    return addrs


def validate_download_url(url: str, allow_private: bool) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValidationError("Only http(s) download URLs are supported.")
    if not parsed.hostname:
        raise ValidationError("Download URL has no host.")
    if allow_private:
        return url
    for addr in _resolve(parsed.hostname):
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _address_is_blocked(ip):
            raise ValidationError(
                f"Refusing to fetch from a private/internal address ({addr})."
            )
    return url


# --------------------------------------------------------------------------- #
# Modelfile PARAMETER parsing
# --------------------------------------------------------------------------- #
def _coerce(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_parameters(text: str) -> dict:
    """Parse a textarea of ``key value`` / ``key: value`` lines into a dict
    suitable for Ollama's /api/create ``parameters`` field."""
    params: dict = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        head = line.split(":", 1)[0].strip().lower()
        if ":" in line and head in ALLOWED_PARAM_KEYS:
            key, value = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValidationError(f"Cannot parse parameter line: {raw!r}")
            key, value = parts

        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key not in ALLOWED_PARAM_KEYS:
            raise ValidationError(f"Unknown or disallowed parameter: {key!r}")

        if key == "stop":
            params.setdefault("stop", []).append(value)
        else:
            params[key] = _coerce(value)
    return params
