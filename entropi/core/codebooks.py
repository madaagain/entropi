import numpy as np
from scipy.stats import norm


def _lloyd_max(n_levels: int, max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Lloyd-Max iteratif pour N(0,1).
    On part de centroids uniformes dans [-3, +3] et on converge
    vers les centroids optimaux qui minimisent la distortion."""

    centroids = np.linspace(-3, 3, n_levels)

    for _ in range(max_iter):
        # boundaries = midpoints entre centroids consecutifs
        boundaries = np.concatenate([
            [-np.inf],
            (centroids[:-1] + centroids[1:]) / 2,
            [np.inf],
        ])

        new_centroids = np.empty(n_levels)
        for k in range(n_levels):
            lo, hi = boundaries[k], boundaries[k + 1]
            # E[X | lo < X < hi] pour X ~ N(0,1)
            # = (phi(lo) - phi(hi)) / (Phi(hi) - Phi(lo))
            prob = norm.cdf(hi) - norm.cdf(lo)
            if prob < 1e-15:
                new_centroids[k] = centroids[k]
            else:
                new_centroids[k] = (norm.pdf(lo) - norm.pdf(hi)) / prob

        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    centroids.sort()
    return centroids


# centroids precalcules pour b=1,2,3,4 (normalises pour N(0,1))
CODEBOOK_DATA = {
    1: _lloyd_max(2),
    2: _lloyd_max(4),
    3: _lloyd_max(8),
    4: _lloyd_max(16),
}


def get_codebook(bit_width: int, dim: int) -> np.ndarray:
    """Retourne le codebook adapte a la dimension.
    On divise par sqrt(dim) parce que pour dim eleve,
    chaque coordonnee du vecteur rote suit N(0, 1/dim)."""

    if bit_width not in CODEBOOK_DATA:
        raise ValueError(f"bit_width must be in {{1, 2, 3, 4}}, got {bit_width}")

    centroids = CODEBOOK_DATA[bit_width].copy()
    return centroids / np.sqrt(dim)
