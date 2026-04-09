from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from entropi.db.api_keys import validate_api_key

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """validate the api key from the X-API-Key header"""
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return api_key
