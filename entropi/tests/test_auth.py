"""
Auth tests — runs against FastAPI TestClient with in-memory fakes.
"""

import pytest
import uuid
import os
import secrets

# set env before any imports
os.environ["JWT_SECRET_KEY"] = "test-secret-key"

import bcrypt
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

# in-memory stores
_users: dict[str, dict] = {}
_keys: dict[str, dict] = {}


def fake_create_user(email, password):
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    uid = str(uuid.uuid4())
    if any(u["email"] == email for u in _users.values()):
        raise Exception("duplicate key value violates unique constraint")
    user = {
        "id": uid, "email": email, "password_hash": pw_hash,
        "plan": "free", "created_at": "2026-01-01T00:00:00", "is_active": True,
    }
    _users[uid] = user
    return {k: v for k, v in user.items() if k != "password_hash"}


def fake_get_by_email(email):
    for u in _users.values():
        if u["email"] == email:
            return u
    return None


def fake_get_by_id(user_id):
    u = _users.get(user_id)
    if u is None:
        return None
    return {k: v for k, v in u.items() if k != "password_hash"}


def fake_create_key(name, user_id=None):
    kid = str(uuid.uuid4())
    api_key = f"ent_live_{secrets.token_hex(16)}"
    _keys[kid] = {
        "id": kid, "name": name, "user_id": user_id,
        "is_active": True, "created_at": "2026-01-01T00:00:00", "key": api_key,
    }
    return api_key


def fake_get_keys(user_id):
    return [
        {"id": k["id"], "name": k["name"], "created_at": k["created_at"], "is_active": k["is_active"]}
        for k in _keys.values() if k["user_id"] == user_id
    ]


def fake_delete_key(key_id, user_id):
    k = _keys.get(key_id)
    if k and k["user_id"] == user_id and k["is_active"]:
        k["is_active"] = False
        return True
    return False


# patch modules BEFORE importing routes
import entropi.db.users as users_mod
import entropi.db.api_keys as keys_mod
users_mod.create_user = fake_create_user
users_mod.get_user_by_email = fake_get_by_email
users_mod.get_user_by_id = fake_get_by_id
keys_mod.create_api_key = fake_create_key
keys_mod.get_keys_by_user = fake_get_keys
keys_mod.delete_key = fake_delete_key

# now import routes — they import from these modules
import entropi.api.routes.auth as auth_route_mod
import entropi.api.routes.keys as keys_route_mod
import entropi.api.auth as auth_mod

# patch the names in the route modules directly
auth_route_mod.create_user = fake_create_user
auth_route_mod.get_user_by_email = fake_get_by_email
auth_route_mod.verify_password = users_mod.verify_password  # keep real bcrypt verify
keys_route_mod.create_api_key = fake_create_key
keys_route_mod.get_keys_by_user = fake_get_keys
keys_route_mod.delete_key = fake_delete_key
auth_mod.get_user_by_id = fake_get_by_id

app = FastAPI()
app.include_router(auth_route_mod.router)
app.include_router(keys_route_mod.router)


@pytest.fixture(autouse=True)
def clear_stores():
    _users.clear()
    _keys.clear()
    yield


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def register_and_get_token(client, email="test@example.com", password="securepass123"):
    resp = await client.post("/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 200, f"Register failed: {resp.text}"
    return resp.json()["token"]


@pytest.mark.anyio
async def test_register(client):
    resp = await client.post("/auth/register", json={
        "email": "user@test.com", "password": "password123",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.anyio
async def test_login(client):
    await register_and_get_token(client, "login@test.com", "mypassword1")
    resp = await client.post("/auth/login", json={
        "email": "login@test.com", "password": "mypassword1",
    })
    assert resp.status_code == 200
    assert "token" in resp.json()


@pytest.mark.anyio
async def test_login_wrong_password(client):
    await register_and_get_token(client, "wrong@test.com", "correctpass1")
    resp = await client.post("/auth/login", json={
        "email": "wrong@test.com", "password": "wrongpassword",
    })
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_get_me(client):
    token = await register_and_get_token(client, "me@test.com", "password123")
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == "me@test.com"
    assert data["plan"] == "free"


@pytest.mark.anyio
async def test_create_key_authenticated(client):
    token = await register_and_get_token(client, "keys@test.com", "password123")
    resp = await client.post("/keys", json={"name": "my-key"}, headers={
        "Authorization": f"Bearer {token}",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["key"].startswith("ent_live_")
    assert data["name"] == "my-key"

    resp = await client.get("/keys", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.anyio
async def test_api_key_belongs_to_user(client):
    token_a = await register_and_get_token(client, "a@test.com", "password123")
    await client.post("/keys", json={"name": "a-key"}, headers={
        "Authorization": f"Bearer {token_a}",
    })
    resp = await client.get("/keys", headers={"Authorization": f"Bearer {token_a}"})
    key_id = resp.json()[0]["id"]

    token_b = await register_and_get_token(client, "b@test.com", "password123")
    resp = await client.delete(f"/keys/{key_id}", headers={
        "Authorization": f"Bearer {token_b}",
    })
    assert resp.status_code == 404
