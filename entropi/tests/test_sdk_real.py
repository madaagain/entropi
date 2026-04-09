"""
End-to-end tests against a real running API.
Requires docker-compose up.
Reads TEST_API_KEY and TEST_BASE_URL from .env automatically.
"""

import os
from pathlib import Path

import httpx
import numpy as np
import pytest
from dotenv import load_dotenv

from entropi.sdk.client import EntropiClient

# load .env from repo root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TEST_API_KEY", "")


@pytest.fixture
def client():
    with EntropiClient(api_key=API_KEY, base_url=BASE_URL) as c:
        yield c


# ---- 1. Health check ----

def test_health_check():
    resp = httpx.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    print(f"  health: {data}")


# ---- 2. Invalid API key ----

def test_api_key_invalid():
    with EntropiClient(api_key="invalid_key", base_url=BASE_URL) as bad_client:
        with pytest.raises(RuntimeError, match="401"):
            bad_client.compress([[0.0] * 128], bit_width=3)
    print("  invalid key correctly rejected")


# ---- 3. Single vector ----

def test_compress_single_vector(client):
    vec = np.random.randn(1, 1536).tolist()
    result = client.compress(vec, bit_width=3)
    ratio = result["compression_ratio"]
    print(f"  single vector compression ratio: {ratio:.1f}x")
    assert ratio >= 5.0, f"ratio {ratio} too low"


# ---- 4. Batch ----

def test_compress_batch(client):
    vecs = np.random.randn(100, 1536).tolist()
    result = client.compress(vecs, bit_width=3)
    assert result["n_vectors"] == 100
    print(f"  batch: {result['n_vectors']} vectors compressed")


# ---- 5. Roundtrip shape ----

def test_compress_decompress_roundtrip(client):
    original = np.random.randn(10, 1536).astype(np.float32)
    result = client.compress(original, bit_width=3)
    reconstructed = client.decompress(result["compressed"])
    assert reconstructed.shape == original.shape, \
        f"shape mismatch: {reconstructed.shape} vs {original.shape}"
    print(f"  roundtrip shape ok: {reconstructed.shape}")


# ---- 6. Recall preserved ----

def test_recall_preserved(client):
    # the real question: does compression preserve cluster structure?
    # exact top-1 is unstable when neighbors have near-identical scores,
    # so we check that the top-1 neighbor stays in the same cluster
    n_clusters = 10
    per_cluster = 5
    n = n_clusters * per_cluster
    rng = np.random.default_rng(42)

    centers = rng.standard_normal((n_clusters, 1536)).astype(np.float32)
    vecs = []
    labels = []
    for i, c in enumerate(centers):
        for _ in range(per_cluster):
            vecs.append(c + rng.standard_normal(1536).astype(np.float32) * 0.1)
            labels.append(i)
    vecs = np.array(vecs)
    labels = np.array(labels)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    # compress decompress
    result = client.compress(vecs, bit_width=4, mode="mse")
    recon = client.decompress(result["compressed"])
    recon = recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-8)

    # top-1 on reconstructed vectors
    recon_sims = recon @ recon.T
    np.fill_diagonal(recon_sims, -np.inf)
    recon_top1 = np.argmax(recon_sims, axis=1)

    # check: is the top-1 neighbor in the same cluster?
    same_cluster = np.mean(labels[recon_top1] == labels)
    print(f"  cluster recall: {same_cluster:.2%} ({int(same_cluster * n)}/{n})")
    assert same_cluster >= 0.9, f"cluster recall {same_cluster:.2%} too low"


# ---- 7, 8, 9. Compression ratios ----

def test_compression_ratio_b2(client):
    vecs = np.random.randn(5, 1536).tolist()
    result = client.compress(vecs, bit_width=2)
    ratio = result["compression_ratio"]
    print(f"  b=2 ratio: {ratio:.1f}x")
    assert ratio >= 10.0


def test_compression_ratio_b3(client):
    vecs = np.random.randn(5, 1536).tolist()
    result = client.compress(vecs, bit_width=3)
    ratio = result["compression_ratio"]
    print(f"  b=3 ratio: {ratio:.1f}x")
    assert ratio >= 6.0


def test_compression_ratio_b4(client):
    vecs = np.random.randn(5, 1536).tolist()
    result = client.compress(vecs, bit_width=4)
    ratio = result["compression_ratio"]
    print(f"  b=4 ratio: {ratio:.1f}x")
    assert ratio >= 5.0


# ---- 10. Numpy input ----

def test_sdk_numpy_input(client):
    vecs = np.random.randn(3, 1536).astype(np.float32)
    result = client.compress(vecs, bit_width=3)
    assert result["n_vectors"] == 3
    reconstructed = client.decompress(result["compressed"])
    assert reconstructed.shape == (3, 1536)
    print(f"  numpy input works, got back shape {reconstructed.shape}")
