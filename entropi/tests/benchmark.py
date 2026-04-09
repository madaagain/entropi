import numpy as np
import time
from entropi.core.turboquant_mse import TurboQuantMSE
from entropi.core.turboquant_prod import TurboQuantProd


def generate_fake_openai_embeddings(n: int, dim: int = 1536) -> np.ndarray:
    """
    Generates embeddings that look like OpenAI's.
    N(0,1) normalized — good approximation.
    """
    vectors = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def benchmark_compression(n_vectors: int = 10000, dim: int = 1536):
    print(f"\n{'='*60}")
    print(f"Entropi Benchmark — {n_vectors} vectors, dim={dim}")
    print(f"{'='*60}")

    vectors = generate_fake_openai_embeddings(n_vectors, dim)

    for bit_width in [2, 3, 4]:
        tq = TurboQuantProd(dim, bit_width)

        # Compression
        t0 = time.perf_counter()
        compressed = tq.quantize(vectors)
        t_compress = time.perf_counter() - t0

        # Decompression
        t0 = time.perf_counter()
        reconstructed = tq.dequantize(compressed)
        t_decompress = time.perf_counter() - t0

        # Metrics
        # 1. MSE
        dmse = np.mean(np.sum((vectors - reconstructed)**2, axis=1))

        # 2. Inner product preservation (1000 pairs)
        idx = np.random.choice(n_vectors, size=(1000, 2))
        true_ips = np.sum(vectors[idx[:, 0]] * vectors[idx[:, 1]], axis=1)
        recon_ips = np.sum(reconstructed[idx[:, 0]] * reconstructed[idx[:, 1]], axis=1)
        ip_error = np.mean(np.abs(true_ips - recon_ips))

        # 3. Compression ratio
        original_bytes = vectors.nbytes
        compressed_bytes = n_vectors * dim * bit_width / 8
        ratio = original_bytes / compressed_bytes

        print(f"\n  bit_width={bit_width}")
        print(f"  Compression ratio : {ratio:.1f}x")
        print(f"  MSE distortion    : {dmse:.4f}")
        print(f"  IP error (mean)   : {ip_error:.4f}")
        print(f"  Compress time     : {t_compress*1000:.1f}ms ({n_vectors/t_compress:.0f} vec/s)")
        print(f"  Decompress time   : {t_decompress*1000:.1f}ms")

    print(f"\n{'='*60}")
    print("Expected from paper (dim=1536) :")
    print("  b=2 : MSE ~ 0.117, ratio ~ 16x")
    print("  b=3 : MSE ~ 0.030, ratio ~ 10.7x")
    print("  b=4 : MSE ~ 0.009, ratio ~ 8x")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    benchmark_compression(n_vectors=10000, dim=1536)
