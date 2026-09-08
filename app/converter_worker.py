from __future__ import annotations

from arq.connections import RedisSettings

from .config import get_settings
from .db import DB
from .hf_convert import convert_hf_repo
from .queues import HF_CONVERT_HEALTH_KEY as HEALTH_CHECK_KEY
from .queues import HF_CONVERT_QUEUE as QUEUE_NAME


async def _startup(ctx: dict) -> None:
    settings = get_settings()
    db = DB(settings.db_path)
    await db.init()
    await db.mark_interrupted()
    ctx["db"] = db


class WorkerSettings:
    functions = [convert_hf_repo]
    on_startup = _startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    queue_name = QUEUE_NAME
    health_check_key = HEALTH_CHECK_KEY
    health_check_interval = 30
    # Conversion is CPU/RAM heavy (model weights held in memory); run one at a time.
    max_jobs = 1
    job_timeout = 60 * 60 * 12
    keep_result = 3600
