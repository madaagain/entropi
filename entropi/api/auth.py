from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from entropi.db.api_keys import validate_api_key
from entropi.api.jwt import decode_token
from entropi.db.users import get_user_by_id

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_SCHEME = HTTPBearer(auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
    credentials: HTTPAuthorizationCredentials | None = Security(BEARER_SCHEME),
):
    """supports two auth modes:
    1. X-API-Key header -> validates key and checks user is active
    2. Authorization: Bearer <jwt> -> validates JWT (for dashboard routes)
    """
    # try API key first
    if api_key:
        result = validate_api_key(api_key)
        if result is None:
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")
        return api_key

    # try Bearer token
    if credentials:
        user_id = decode_token(credentials.credentials)
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return credentials.credentials

    raise HTTPException(status_code=401, detail="Missing authentication")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(BEARER_SCHEME),
):
    """extract user from JWT token — for dashboard routes only"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    user_id = decode_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_by_id(user_id)
    if user is None or not user["is_active"]:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user
