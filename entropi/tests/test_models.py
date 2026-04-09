import pytest
from pydantic import ValidationError
from entropi.api.models import CompressRequest, CompressResponse


def test_compress_request_valid():
    """une request normale doit passer sans probleme"""
    vecs = [[float(i) for i in range(128)] for _ in range(3)]
    req = CompressRequest(vectors=vecs, bit_width=3, mode="prod")
    assert req.bit_width == 3
    assert req.mode == "prod"
    assert len(req.vectors) == 3


def test_compress_request_defaults():
    """les valeurs par defaut doivent etre bit_width=3, mode=prod"""
    vecs = [[0.0] * 128]
    req = CompressRequest(vectors=vecs)
    assert req.bit_width == 3
    assert req.mode == "prod"


def test_compress_request_invalid_bitwidth():
    """bit_width=5 ca passe pas"""
    vecs = [[0.0] * 128]
    with pytest.raises(ValidationError):
        CompressRequest(vectors=vecs, bit_width=5)


def test_compress_request_invalid_mode():
    """mode=bad ca passe pas non plus"""
    vecs = [[0.0] * 128]
    with pytest.raises(ValidationError):
        CompressRequest(vectors=vecs, mode="bad")


def test_compress_request_empty_vectors():
    """liste vide = erreur"""
    with pytest.raises(ValidationError):
        CompressRequest(vectors=[])


def test_compress_request_inconsistent_dims():
    """des vecteurs de tailles differentes ca marche pas"""
    vecs = [[0.0] * 128, [0.0] * 256]
    with pytest.raises(ValidationError):
        CompressRequest(vectors=vecs)


def test_compress_request_dim_too_small():
    """dim < 64 c'est trop petit"""
    vecs = [[0.0] * 10]
    with pytest.raises(ValidationError):
        CompressRequest(vectors=vecs)


def test_compression_ratio():
    """compression_ratio = 32.0 / bits_per_dim"""
    for b in [2, 3, 4]:
        resp = CompressResponse(
            compressed={},
            original_dim=1536,
            n_vectors=10,
            compressed_bits_per_dim=float(b),
            compression_ratio=32.0 / b,
        )
        assert resp.compression_ratio == 32.0 / b
