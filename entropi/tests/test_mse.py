import numpy as np
from entropi.core.turboquant_mse import TurboQuantMSE


def _compute_dmse(bit_width: int, dim: int = 1536, n_vectors: int = 10000) -> float:
    """calcule la distortion MSE comme dans le papier.
    on genere des vecteurs random, on compresse, on decompresse,
    et on mesure l'erreur sur les vecteurs normalises."""
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)

    tq = TurboQuantMSE(dim, bit_width)
    compressed = tq.quantize(vectors)
    reconstructed = tq.dequantize(compressed)

    # normalise les deux pour comparer sur la sphere unitaire
    orig_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    recon_norm = reconstructed / (np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8)

    dmse = np.mean(np.sum((orig_norm - recon_norm) ** 2, axis=1))
    return float(dmse)


def test_mse_distortion_b1():
    """Theorem 1 : b=1 -> Dmse ~ 0.36"""
    dmse = _compute_dmse(1)
    expected = 0.36
    print(f"b=1 : Dmse = {dmse:.4f} (attendu ~{expected})")
    assert abs(dmse - expected) / expected < 0.2, f"Dmse={dmse:.4f}, trop loin de {expected}"


def test_mse_distortion_b2():
    """Theorem 1 : b=2 -> Dmse ~ 0.117"""
    dmse = _compute_dmse(2)
    expected = 0.117
    print(f"b=2 : Dmse = {dmse:.4f} (attendu ~{expected})")
    assert abs(dmse - expected) / expected < 0.2, f"Dmse={dmse:.4f}, trop loin de {expected}"


def test_mse_distortion_b3():
    """Theorem 1 : b=3 -> Dmse ~ 0.03"""
    dmse = _compute_dmse(3)
    expected = 0.03
    print(f"b=3 : Dmse = {dmse:.4f} (attendu ~{expected})")
    assert abs(dmse - expected) / expected < 0.2, f"Dmse={dmse:.4f}, trop loin de {expected}"


def test_mse_distortion_b4():
    """Theorem 1 : b=4 -> Dmse ~ 0.009"""
    dmse = _compute_dmse(4)
    expected = 0.009
    print(f"b=4 : Dmse = {dmse:.4f} (attendu ~{expected})")
    assert abs(dmse - expected) / expected < 0.2, f"Dmse={dmse:.4f}, trop loin de {expected}"


def test_shape_preserved():
    """output doit avoir la meme shape que input"""
    vecs = np.random.randn(5, 256)
    tq = TurboQuantMSE(256, bit_width=2)
    compressed = tq.quantize(vecs)
    reconstructed = tq.dequantize(compressed)
    assert reconstructed.shape == (5, 256), f"shape = {reconstructed.shape}, attendu (5, 256)"


def test_single_vector():
    """doit marcher avec un seul vecteur shape (d,)"""
    vec = np.random.randn(128)
    tq = TurboQuantMSE(128, bit_width=3)
    compressed = tq.quantize(vec)
    reconstructed = tq.dequantize(compressed)
    assert reconstructed.shape == (128,), f"shape = {reconstructed.shape}, attendu (128,)"


def test_batch_vectors():
    """doit marcher avec un batch shape (n, d)"""
    vecs = np.random.randn(20, 512)
    tq = TurboQuantMSE(512, bit_width=2)
    compressed = tq.quantize(vecs)
    reconstructed = tq.dequantize(compressed)
    assert reconstructed.shape == (20, 512), f"shape = {reconstructed.shape}, attendu (20, 512)"
