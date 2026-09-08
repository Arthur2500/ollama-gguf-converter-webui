from __future__ import annotations

import json
import secrets as _secrets
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Generated once per process. Used only when SECRET_KEY is not configured.
_EPHEMERAL_SECRET = _secrets.token_urlsafe(48)


class OllamaInstance(BaseModel):
    name: str
    url: str

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("instance name must not be empty")
        return v

    @field_validator("url")
    @classmethod
    def _clean_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("instance url must start with http:// or https://")
        return v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- auth ---
    app_password: str | None = None
    app_password_hash: str | None = None
    secret_key: str = ""
    session_max_age_hours: int = 12
    cookie_secure: bool = True
    trust_proxy: bool = False
    login_max_attempts: int = 5
    login_lockout_minutes: int = 15

    # --- infra ---
    redis_url: str = "redis://redis:6379"
    data_dir: Path = Path("/data")

    # --- ollama ---
    ollama_instances: str = ""
    ollama_url: str = "http://ollama:11434"

    # --- limits / behaviour ---
    max_gguf_size_gb: float = 100.0
    allow_private_download_urls: bool = False
    delete_gguf_on_failure: bool = True
    hf_token: str | None = None
    download_timeout_seconds: int = 0  # 0 => no read timeout

    @property
    def instances(self) -> list[OllamaInstance]:
        raw = self.ollama_instances.strip()
        if raw:
            data = json.loads(raw)
            if not isinstance(data, list) or not data:
                raise ValueError("OLLAMA_INSTANCES must be a non-empty JSON list")
            instances = [OllamaInstance(**d) for d in data]
            names = [i.name for i in instances]
            if len(names) != len(set(names)):
                raise ValueError("OLLAMA_INSTANCES contains duplicate names")
            return instances
        return [OllamaInstance(name="default", url=self.ollama_url)]

    @property
    def effective_secret_key(self) -> str:
        return self.secret_key or _EPHEMERAL_SECRET

    @property
    def session_max_age(self) -> int:
        return self.session_max_age_hours * 3600

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def downloads_dir(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def max_gguf_bytes(self) -> int:
        return int(self.max_gguf_size_gb * 1_000_000_000)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.downloads_dir.mkdir(parents=True, exist_ok=True)
    return s
