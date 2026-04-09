import os
import numpy as np
import pytest

from entropi.sdk.client import EntropiClient

# requires docker-compose up
# generate a key: docker exec entropi-api-1 python -c "
#   from entropi.db.api_keys import create_api_key; print(create_api_key('test'))"
# then export TEST_API_KEY=ent_live_xxx

BASE_URL = "http://localhost:8000"
API_KEY = os.environ.get("TEST_API_KEY", "ent_live_0c6da301f4f7e4515927845aca6fcc39")


@pytest.fixture
def client():
    with EntropiClient(api_key=API_KEY, base_url=BASE_URL) as c:
        yield c


def test_sdk_compress_decompress(client):
    """roundtrip complet: compress -> decompress -> vectors proches"""
    dim = 128
    vectors = np.random.randn(3, dim).astype(np.float32)

    result = client.compress(vectors, bit_width=3, mode="mse")
    assert result["original_dim"] == dim
    assert result["n_vectors"] == 3
    assert result["compression_ratio"] == 32.0 / 3

    reconstructed = client.decompress(result["compressed"])
    assert reconstructed.shape == (3, dim)

    # cosine similarity should be high
    for i in range(3):
        orig = vectors[i] / np.linalg.norm(vectors[i])
        recon = reconstructed[i] / np.linalg.norm(reconstructed[i])
        cosine = np.dot(orig, recon)
        assert cosine > 0.9, f"vector {i}: cosine = {cosine}"


def test_sdk_compress_prod_mode(client):
    """test prod mode roundtrip"""
    vectors = np.random.randn(2, 256).astype(np.float32)

    result = client.compress(vectors, bit_width=3, mode="prod")
    reconstructed = client.decompress(result["compressed"])
    assert reconstructed.shape == (2, 256)


def test_sdk_compress_and_store(client):
    """test the helper method"""
    vectors = np.random.randn(5, 128).astype(np.float32)

    compressed, ratio = client.compress_and_store(vectors, bit_width=3)
    assert ratio == 32.0 / 3
    assert "indices" in compressed


def test_sdk_invalid_key():
    """bad api key should fail"""
    with EntropiClient(api_key="ent_live_fake", base_url=BASE_URL) as c:
        with pytest.raises(RuntimeError, match="401"):
            c.compress([[0.0] * 128], bit_width=3)
