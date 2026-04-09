import httpx
import numpy as np


class EntropiClient:
    """Python SDK for the Entropi vector compression API.
    pip install entropi -> three lines of code to compress your embeddings."""

    def __init__(self, api_key: str, base_url: str = "https://api.entropi.dev"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            headers={"X-API-Key": api_key},
            timeout=60.0,
        )

    def compress(
        self,
        vectors,
        bit_width: int = 3,
        mode: str = "prod",
    ) -> dict:
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        resp = self._client.post(
            f"{self.base_url}/v1/compress",
            json={"vectors": vectors, "bit_width": bit_width, "mode": mode},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Compress failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def decompress(self, compressed: dict) -> np.ndarray:
        resp = self._client.post(
            f"{self.base_url}/v1/decompress",
            json={"compressed": compressed},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Decompress failed ({resp.status_code}): {resp.text}")
        return np.array(resp.json()["vectors"], dtype=np.float32)

    def compress_and_store(
        self,
        vectors,
        bit_width: int = 3,
    ) -> tuple[dict, float]:
        result = self.compress(vectors, bit_width=bit_width)
        return result["compressed"], result["compression_ratio"]

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
