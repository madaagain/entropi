from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from entropi.api.auth import get_current_user
from entropi.db.api_keys import create_api_key, get_keys_by_user, delete_key

router = APIRouter(prefix="/keys", tags=["keys"])


class CreateKeyRequest(BaseModel):
    name: str


class KeyResponse(BaseModel):
    key: str
    name: str


class KeyListItem(BaseModel):
    id: str
    name: str
    created_at: str
    is_active: bool


@router.get("", response_model=list[KeyListItem])
async def list_keys(user: dict = Depends(get_current_user)):
    keys = get_keys_by_user(user["id"])
    return [KeyListItem(**k) for k in keys]


@router.post("", response_model=KeyResponse, status_code=201)
async def create_key(req: CreateKeyRequest, user: dict = Depends(get_current_user)):
    api_key = create_api_key(req.name, user_id=user["id"])
    return KeyResponse(key=api_key, name=req.name)


@router.delete("/{key_id}")
async def remove_key(key_id: str, user: dict = Depends(get_current_user)):
    deleted = delete_key(key_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Key not found or already deleted")
    return {"status": "deleted"}
