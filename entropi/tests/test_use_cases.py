"""
Real startup use cases — no OpenAI API needed, just numpy.
These are the demos you show to early testers.
"""

import time
import numpy as np
import pytest
from pathlib import Path
from dotenv import load_dotenv
import os

from entropi.sdk.client import EntropiClient

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TEST_API_KEY", "")


@pytest.fixture
def client():
    with EntropiClient(api_key=API_KEY, base_url=BASE_URL) as c:
        yield c


def test_rag_search_quality(client):
    """simulate a RAG pipeline: 20 docs, 5 queries, check if top-3 is preserved"""
    rng = np.random.default_rng(123)
    docs = rng.standard_normal((20, 1536)).astype(np.float32)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    # compress and decompress the doc embeddings
    result = client.compress(docs, bit_width=3, mode="mse")
    recon = client.decompress(result["compressed"])
    recon = recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-8)

    # 5 random queries
    queries = rng.standard_normal((5, 1536)).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    perfect = 0
    for i in range(5):
        q = queries[i]
        # full precision top-3
        true_sims = docs @ q
        true_top3 = set(np.argsort(true_sims)[-3:])
        # reconstructed top-3
        recon_sims = recon @ q
        recon_top3 = set(np.argsort(recon_sims)[-3:])

        if true_top3 == recon_top3:
            perfect += 1

    print(f"\n  RAG Recall@3: {perfect}/5 queries perfectly preserved")
    assert perfect >= 3, f"only {perfect}/5 queries preserved, expected at least 3"


def test_batch_performance(client):
    """compress 10k vectors and measure throughput"""
    rng = np.random.default_rng(456)
    vectors = rng.standard_normal((10000, 1536)).astype(np.float32)

    t0 = time.perf_counter()
    result = client.compress(vectors, bit_width=3)
    elapsed = time.perf_counter() - t0

    ms = elapsed * 1000
    throughput = 10000 / elapsed

    print(f"\n  10k vectors in {ms:.0f}ms")
    print(f"  Throughput: {throughput:.0f} vectors/sec")
    assert result["n_vectors"] == 10000


def test_storage_savings():
    """calculate storage savings for different scales
    no API call needed — just math"""
    dim = 1536
    bit_width = 3

    print(f"\n  {'Vectors':>12}  {'Original':>12}  {'Compressed':>12}  {'Saved':>12}  {'Ratio':>6}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}")

    for n in [1_000, 100_000, 1_000_000, 10_000_000]:
        original_bytes = n * dim * 4  # float32 = 4 bytes
        compressed_bytes = n * dim * bit_width / 8
        saved_bytes = original_bytes - compressed_bytes
        ratio = original_bytes / compressed_bytes

        def fmt(b):
            if b >= 1e9:
                return f"{b/1e9:.1f} GB"
            elif b >= 1e6:
                return f"{b/1e6:.1f} MB"
            else:
                return f"{b/1e3:.1f} KB"

        label = f"{n:,}"
        print(f"  {label:>12}  {fmt(original_bytes):>12}  {fmt(compressed_bytes):>12}  {fmt(saved_bytes):>12}  {ratio:>5.1f}x")

    # just verify the math is right for 1M vectors
    original_1m = 1_000_000 * dim * 4
    compressed_1m = 1_000_000 * dim * bit_width / 8
    assert original_1m / compressed_1m == pytest.approx(32 / bit_width, rel=0.01)
