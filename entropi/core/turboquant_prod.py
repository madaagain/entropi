import numpy as np

from entropi.core.turboquant_mse import TurboQuantMSE
from entropi.core.utils import normalize, ensure_batch


class TurboQuantProd:
    """Algorithme 2 du papier TurboQuant.
    TurboQuantMSE(b-1) + QJL sur le residu.
    Estimateur non biaise pour les inner products."""

    def __init__(self, dim: int, bit_width: int = 3, seed: int = None):
        self.dim = dim
        self.bit_width = bit_width
        self.mse = TurboQuantMSE(dim, bit_width - 1, seed)

        # matrice de projection QJL, seed+1 pour pas collisionner avec la rotation
        qjl_seed = self.mse.rotation.seed + 1
        rng = np.random.RandomState(qjl_seed)
        self.S = rng.randn(dim, dim)

    def quantize(self, x: np.ndarray) -> dict:
        x, single = ensure_batch(x)
        x_normalized, norms = normalize(x)

        # quantize MSE avec b-1 bits
        mse_compressed = self.mse.quantize(x_normalized)
        x_mse_reconstructed = self.mse.dequantize(mse_compressed)
        if x_mse_reconstructed.ndim == 1:
            x_mse_reconstructed = x_mse_reconstructed[np.newaxis, :]

        # residu et QJL
        residual = x_normalized - x_mse_reconstructed
        residual_norm = np.linalg.norm(residual, axis=-1)
        qjl = np.sign(residual @ self.S.T)

        # sign(0) = 0 en numpy, on veut que des -1 et +1
        qjl[qjl == 0] = 1.0

        return {
            "mse_indices": mse_compressed["indices"],
            "qjl": qjl.astype(np.int8),
            "residual_norm": residual_norm,
            "norms": norms,
            "rotation_seed": self.mse.rotation.seed,
            "bit_width": self.bit_width,
            "dim": self.dim,
            "single": single,
        }

    def dequantize(self, compressed: dict) -> np.ndarray:
        n = compressed["mse_indices"].shape[0]

        # reconstruire la partie MSE
        mse_fake_compressed = {
            "indices": compressed["mse_indices"],
            "norms": np.ones(n),
            "rotation_seed": compressed["rotation_seed"],
            "bit_width": self.bit_width - 1,
            "dim": compressed["dim"],
            "single": False,
        }
        x_mse = self.mse.dequantize(mse_fake_compressed)

        # reconstruire la correction QJL (papier algo 2 ligne 11)
        residual_norm = compressed["residual_norm"]
        qjl = compressed["qjl"].astype(np.float64)
        x_qjl = (np.sqrt(np.pi / 2) / self.dim) * (qjl @ self.S) * residual_norm[:, np.newaxis]

        # combiner et rescaler
        reconstructed = (x_mse + x_qjl) * compressed["norms"][:, np.newaxis]

        if compressed["single"]:
            return reconstructed[0]
        return reconstructed
