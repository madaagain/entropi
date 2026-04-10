import os
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError


def _get_secret() -> str:
    secret = os.environ.get("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY environment variable is not set")
    return secret


ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
EXPIRE_DAYS = int(os.environ.get("JWT_EXPIRE_DAYS", "7"))


def create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=EXPIRE_DAYS)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> str | None:
    """returns user_id if valid, None otherwise"""
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
