from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    source_url    TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    instance_name TEXT NOT NULL,
    status        TEXT NOT NULL,
    phase         TEXT NOT NULL DEFAULT '',
    progress      REAL NOT NULL DEFAULT 0,
    error         TEXT,
    gguf_sha256   TEXT,
    download_size INTEGER,
    final_size    INTEGER,
    model_info    TEXT,
    modelfile     TEXT,
    options       TEXT NOT NULL DEFAULT '{}',
    logs          TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""

TERMINAL_STATUSES = {"success", "failed"}
_JSON_COLUMNS = ("model_info", "options", "logs")


def new_job_id() -> str:
    return uuid.uuid4().hex


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    data = dict(row)
    for col in _JSON_COLUMNS:
        raw = data.get(col)
        if isinstance(raw, str):
            try:
                data[col] = json.loads(raw)
            except json.JSONDecodeError:
                data[col] = None
    return data


class DB:
    def __init__(self, path: Path):
        self.path = str(path)

    async def _connect(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=8000")
        return conn

    async def init(self) -> None:
        conn = await self._connect()
        try:
            await conn.executescript(SCHEMA)
            await conn.commit()
        finally:
            await conn.close()

    async def mark_interrupted(self) -> None:
        """Called on worker startup: nothing can be running yet, so any
        non-terminal job is a leftover from a crash/restart."""
        conn = await self._connect()
        try:
            await conn.execute(
                "UPDATE jobs SET status='failed', "
                "error=COALESCE(error, 'Interrupted by a restart'), "
                "phase='interrupted', updated_at=? "
                "WHERE status NOT IN ('success', 'failed')",
                (time.time(),),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def create_job(self, job: dict[str, Any]) -> None:
        columns = ", ".join(job)
        placeholders = ", ".join("?" for _ in job)
        conn = await self._connect()
        try:
            await conn.execute(
                f"INSERT INTO jobs ({columns}) VALUES ({placeholders})",
                tuple(job.values()),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def update_job(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        conn = await self._connect()
        try:
            await conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*fields.values(), job_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def append_log(self, job_id: str, message: str) -> None:
        conn = await self._connect()
        try:
            async with conn.execute(
                "SELECT logs FROM jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                return
            try:
                logs = json.loads(row["logs"])
            except json.JSONDecodeError:
                logs = []
            logs.append({"ts": time.time(), "msg": message})
            logs = logs[-250:]
            await conn.execute(
                "UPDATE jobs SET logs = ?, updated_at = ? WHERE id = ?",
                (json.dumps(logs), time.time(), job_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        conn = await self._connect()
        try:
            async with conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
            return _row_to_dict(row) if row else None
        finally:
            await conn.close()

    async def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        conn = await self._connect()
        try:
            async with conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ) as cur:
                rows = await cur.fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            await conn.close()

    async def delete_job(self, job_id: str) -> None:
        conn = await self._connect()
        try:
            await conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            await conn.commit()
        finally:
            await conn.close()
