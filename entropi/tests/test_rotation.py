import numpy as np
from entropi.core.rotation import RandomRotation, generate_rotation


def test_orthogonal():
    """on check que Q @ Q.T donne bien l'identite
    sinon c'est pas une rotation et on a un probleme lol"""
    rot = RandomRotation(dim=64, seed=123)
    Q = rot.matrix
    produit = Q @ Q.T
    identite = np.eye(64)
    assert np.allclose(produit, identite, atol=1e-5), "Q @ Q.T != I, c'est casse"


def test_reproducible():
    """meme seed = meme matrice, basique"""
    r1 = RandomRotation(dim=64, seed=999)
    r2 = RandomRotation(dim=64, seed=999)
    assert np.array_equal(r1.matrix, r2.matrix), "meme seed mais matrices differentes ??"


def test_different_seeds():
    """seeds differents = matrices differentes, logique"""
    r1 = RandomRotation(dim=64, seed=1)
    r2 = RandomRotation(dim=64, seed=2)
    assert not np.array_equal(r1.matrix, r2.matrix), "seeds differents mais meme matrice wtf"


def test_determinant():
    """le determinant doit etre +1 pour une vraie rotation
    (-1 ca serait une reflexion et on veut pas ca)"""
    rot = RandomRotation(dim=64, seed=42)
    det = np.linalg.det(rot.matrix)
    assert np.isclose(det, 1.0, atol=1e-5), f"det = {det}, on voulait +1"


def test_factory():
    """juste pour verifier que generate_rotation marche"""
    rot = generate_rotation(dim=32, seed=77)
    assert isinstance(rot, RandomRotation)
    assert rot.dim == 32
    assert rot.seed == 77
    assert rot.matrix.shape == (32, 32)


def test_lazy_loading():
    """la matrice doit pas etre generee dans __init__
    sinon c'est du gaspillage pour rien"""
    rot = RandomRotation(dim=64, seed=10)
    assert rot._matrix is None, "la matrice a ete generee trop tot"
    _ = rot.matrix
    assert rot._matrix is not None, "la matrice a pas ete generee apres .matrix"
