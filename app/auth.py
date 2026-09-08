from __future__ import annotations

import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import HTTPException, Request, status

_hasher = PasswordHasher()


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(hashed: str, plain: str) -> bool:
    try:
        _hasher.verify(hashed, plain)
        return True
    except (VerifyMismatchError, InvalidHashError, VerificationError):
        return False


# --------------------------------------------------------------------------- #
# CSRF (synchroniser token stored in the signed session)
# --------------------------------------------------------------------------- #
def get_or_create_csrf(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf"] = token
    return token


def verify_csrf(request: Request, submitted: str | None) -> None:
    expected = request.session.get("csrf")
    if not expected or not submitted or not secrets.compare_digest(str(expected), str(submitted)):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid or missing CSRF token.")


# --------------------------------------------------------------------------- #
# Login rate limiting (backed by Redis so it works across web workers)
# --------------------------------------------------------------------------- #
def _to_int(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    return int(value)


async def check_login_rate_limit(redis, ip: str, max_attempts: int) -> None:
    current = _to_int(await redis.get(f"login_fail:{ip}"))
    if current >= max_attempts:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many failed login attempts. Please wait and try again.",
        )


async def register_login_failure(redis, ip: str, lockout_seconds: int) -> None:
    key = f"login_fail:{ip}"
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, lockout_seconds)


async def clear_login_failures(redis, ip: str) -> None:
    await redis.delete(f"login_fail:{ip}")
