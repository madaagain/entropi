import numpy as np


class RandomRotation:
    """Matrice de rotation orthogonale reproductible via seed.
    On fait QR decomposition sur une matrice random N(0,1)."""

    def __init__(self, dim: int, seed: int = None):
        self.dim = dim
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)
        self._matrix = None

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is not None:
            return self._matrix

        rng = np.random.RandomState(self.seed)
        random_matrix = rng.randn(self.dim, self.dim)

        Q, R = np.linalg.qr(random_matrix)

        # correction de signe pour rendre le resultat unique
        Q = Q * np.sign(np.diag(R))

        # on veut une vraie rotation (det = +1), pas une reflexion
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1

        self._matrix = Q
        return self._matrix


def generate_rotation(dim: int, seed: int = None) -> RandomRotation:
    """Factory function, rien de fou."""
    return RandomRotation(dim, seed)
