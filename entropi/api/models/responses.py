from pydantic import BaseModel


class CompressResponse(BaseModel):
    compressed: dict
    original_dim: int
    n_vectors: int
    compressed_bits_per_dim: float
    compression_ratio: float


class DecompressResponse(BaseModel):
    vectors: list[list[float]]
    n_vectors: int
    dim: int


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
