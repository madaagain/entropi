import numpy as np

from entropi.core.rotation import generate_rotation
from entropi.core.codebooks import get_codebook
from entropi.core.utils import normalize, find_nearest, ensure_batch


class TurboQuantMSE:
    """Algorithme 1 du papier TurboQuant.
    Quantization qui minimise l'erreur MSE."""

    def __init__(self, dim: int, bit_width: int = 3, seed: int = None):
        self.dim = dim
        self.bit_width = bit_width
        self.rotation = generate_rotation(dim, seed)
        self.codebook = get_codebook(bit_width, dim)

    def quantize(self, x: np.ndarray) -> dict:
        x, single = ensure_batch(x)
        x_normalized, norms = normalize(x)

        rotated = x_normalized @ self.rotation.matrix.T
        indices = find_nearest(rotated, self.codebook)

        return {
            "indices": indices,
            "norms": norms,
            "rotation_seed": self.rotation.seed,
            "bit_width": self.bit_width,
            "dim": self.dim,
            "single": single,
        }

    def dequantize(self, compressed: dict) -> np.ndarray:
        centroids = self.codebook[compressed["indices"]]
        reconstructed = centroids @ self.rotation.matrix
        norms = compressed["norms"]
        if norms.ndim == 0:
            norms = norms[np.newaxis]
        reconstructed = reconstructed * norms[:, np.newaxis]

        if compressed["single"]:
            return reconstructed[0]
        return reconstructed
