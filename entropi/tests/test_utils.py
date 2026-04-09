import numpy as np
from entropi.core.utils import normalize, find_nearest, ensure_batch


def test_normalize_unit_norm():
    """apres normalisation chaque vecteur doit avoir norme ~1"""
    vecs = np.random.randn(10, 128)
    normed, norms = normalize(vecs)
    result_norms = np.linalg.norm(normed, axis=1)
    assert np.allclose(result_norms, 1.0, atol=1e-6)


def test_normalize_preserves_direction():
    """la direction doit pas changer, juste la longueur"""
    vec = np.array([3.0, 4.0])
    normed, _ = normalize(vec)
    # direction = [0.6, 0.8]
    assert np.allclose(normed, [0.6, 0.8], atol=1e-6)


def test_normalize_recovers_original():
    """x_norm * norm doit redonner x original"""
    vecs = np.random.randn(5, 64)
    normed, norms = normalize(vecs)
    recovered = normed * norms[:, np.newaxis]
    assert np.allclose(recovered, vecs, atol=1e-5)


def test_normalize_single_vector():
    """doit marcher avec un seul vecteur (1D)"""
    vec = np.random.randn(64)
    normed, norm_val = normalize(vec)
    assert normed.ndim == 1
    assert np.isscalar(norm_val) or norm_val.ndim == 0
    assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-6)


def test_find_nearest_shape():
    """le output doit avoir la meme shape que l'input (n, d)"""
    rotated = np.random.randn(10, 128)
    codebook = np.array([-1.5, -0.45, 0.45, 1.5])
    indices = find_nearest(rotated, codebook)
    assert indices.shape == (10, 128)
    assert indices.dtype == np.uint8


def test_find_nearest_correctness():
    """test simple : codebook [-1, 0, 1]
    -0.8 -> index 0 (-1), 0.1 -> index 1 (0), 0.9 -> index 2 (1)"""
    codebook = np.array([-1.0, 0.0, 1.0])
    rotated = np.array([[-0.8, 0.1, 0.9]])
    indices = find_nearest(rotated, codebook)
    assert indices[0, 0] == 0  # -0.8 plus proche de -1
    assert indices[0, 1] == 1  # 0.1 plus proche de 0
    assert indices[0, 2] == 2  # 0.9 plus proche de 1


def test_ensure_batch():
    """single vec -> batch (1, d) avec flag True, et batch reste batch"""
    vec = np.random.randn(64)
    batched, was_single = ensure_batch(vec)
    assert batched.shape == (1, 64)
    assert was_single is True

    vecs = np.random.randn(5, 64)
    batched, was_single = ensure_batch(vecs)
    assert batched.shape == (5, 64)
    assert was_single is False
