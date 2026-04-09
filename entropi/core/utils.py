import numpy as np


def normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """normalise chaque vecteur sur la sphere unitaire.
    retourne (x_normalized, norms)."""
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]

    norms = np.linalg.norm(x, axis=1)
    x_normalized = x / (norms[:, np.newaxis] + 1e-8)

    if single:
        return x_normalized[0], norms[0]
    return x_normalized, norms


def find_nearest(rotated: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """pour chaque coordonnee, trouve l'index du centroid le plus proche.
    tout vectorise, pas de boucle python."""
    # rotated: (n, d), codebook: (c,)
    # diffs: (n, d, c) — distance entre chaque coord et chaque centroid
    diffs = np.abs(rotated[:, :, np.newaxis] - codebook[np.newaxis, np.newaxis, :])
    indices = np.argmin(diffs, axis=2).astype(np.uint8)
    return indices


def ensure_batch(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """transforme un vecteur solo en batch si besoin.
    retourne (x_batch, was_single)."""
    if x.ndim == 1:
        return x[np.newaxis, :], True
    return x, False
