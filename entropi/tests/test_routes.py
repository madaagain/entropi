import pytest
import numpy as np
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from entropi.api.routes.compress import router as compress_router
from entropi.api.routes.decompress import router as decompress_router
from entropi.api.auth import verify_api_key


# fake app with mocked auth so we don't need a real db
app = FastAPI()


async def mock_api_key():
    return "ent_live_test"


app.dependency_overrides[verify_api_key] = mock_api_key
app.include_router(compress_router)
app.include_router(decompress_router)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_compress_decompress_roundtrip(client):
    """compress then decompress, vectors should be close to original"""
    dim = 128
    n = 3
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).tolist()

    # compress
    resp = await client.post("/v1/compress", json={
        "vectors": vectors,
        "bit_width": 3,
        "mode": "mse",
    })
    assert resp.status_code == 200
    compress_data = resp.json()
    assert compress_data["original_dim"] == dim
    assert compress_data["n_vectors"] == n

    # decompress
    resp = await client.post("/v1/decompress", json={
        "compressed": compress_data["compressed"],
    })
    assert resp.status_code == 200
    decompress_data = resp.json()
    assert decompress_data["n_vectors"] == n
    assert decompress_data["dim"] == dim

    # check vectors are close
    original = np.array(vectors)
    reconstructed = np.array(decompress_data["vectors"])
    # normalize both and check cosine similarity
    orig_norm = original / np.linalg.norm(original, axis=1, keepdims=True)
    recon_norm = reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True)
    cosines = np.sum(orig_norm * recon_norm, axis=1)
    assert np.all(cosines > 0.9), f"cosine similarities too low: {cosines}"


@pytest.mark.anyio
async def test_compress_decompress_prod_roundtrip(client):
    """same but with prod mode"""
    dim = 128
    vectors = np.random.randn(2, dim).tolist()

    resp = await client.post("/v1/compress", json={
        "vectors": vectors,
        "bit_width": 3,
        "mode": "prod",
    })
    assert resp.status_code == 200
    compressed = resp.json()["compressed"]

    resp = await client.post("/v1/decompress", json={
        "compressed": compressed,
    })
    assert resp.status_code == 200
    assert resp.json()["n_vectors"] == 2


@pytest.mark.anyio
async def test_compress_invalid_bitwidth(client):
    """bit_width=5 should return 422 (pydantic validation error)"""
    vectors = [[0.0] * 128]
    resp = await client.post("/v1/compress", json={
        "vectors": vectors,
        "bit_width": 5,
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_missing_api_key():
    """no auth override -> missing key should return 401/403"""
    clean_app = FastAPI()
    clean_app.include_router(compress_router)
    clean_app.include_router(decompress_router)

    transport = ASGITransport(app=clean_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/v1/compress", json={
            "vectors": [[0.0] * 128],
            "bit_width": 3,
        })
        assert resp.status_code in (401, 403)
