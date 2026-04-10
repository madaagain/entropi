from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr

from entropi.db.users import create_user, get_user_by_email, verify_password
from entropi.api.jwt import create_token
from entropi.api.auth import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    plan: str
    created_at: str


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    try:
        user = create_user(req.email, req.password)
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(status_code=409, detail="Email already registered")
        raise HTTPException(status_code=400, detail="Registration failed")

    token = create_token(user["id"])
    return TokenResponse(token=token)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user["is_active"]:
        raise HTTPException(status_code=401, detail="Account is deactivated")

    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(user["id"])
    return TokenResponse(token=token)


@router.get("/me", response_model=UserResponse)
async def me(user: dict = Depends(get_current_user)):
    return UserResponse(
        id=user["id"],
        email=user["email"],
        plan=user["plan"],
        created_at=user["created_at"],
    )
