import numpy as np
from entropi.core.turboquant_prod import TurboQuantProd


def test_inner_product_unbiased():
    """le test le plus important : l'estimateur doit etre non biaise
    E[<y, decompress(compress(x))>] = <y, x>
    on fait plein de trials avec des seeds differents et on moyenne"""
    dim = 1536
    n_trials = 500

    x = np.random.randn(dim)
    x = x / np.linalg.norm(x)
    y = np.random.randn(dim)

    true_ip = np.dot(y, x)
    estimated_ips = []

    for trial in range(n_trials):
        tq = TurboQuantProd(dim, bit_width=3, seed=trial)
        compressed = tq.quantize(x)
        reconstructed = tq.dequantize(compressed)
        estimated_ips.append(np.dot(y, reconstructed))

    mean_estimated = np.mean(estimated_ips)
    relative_bias = abs(mean_estimated - true_ip) / (abs(true_ip) + 1e-8)
    print(f"true IP = {true_ip:.4f}, mean estimated = {mean_estimated:.4f}, biais relatif = {relative_bias:.4f}")
    assert relative_bias < 0.02, \
        f"biais relatif = {relative_bias:.4f}, c'est trop (max 2%)"


def test_inner_product_distortion_b3():
    """Theorem 2 du papier : Dprod ~ 0.18/dim pour b=3
    on le verifie empiriquement"""
    dim = 1536
    n_vectors = 2000

    x = np.random.randn(n_vectors, dim)
    norms_x = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / norms_x

    y = np.random.randn(dim)

    true_ips = x_norm @ y
    estimated_ips = np.empty(n_vectors)

    tq = TurboQuantProd(dim, bit_width=3)
    compressed = tq.quantize(x_norm)
    reconstructed = tq.dequantize(compressed)
    estimated_ips = reconstructed @ y

    # Dprod = E[(true_ip - estimated_ip)^2] / ||y||^2
    dprod = np.mean((true_ips - estimated_ips) ** 2) / np.dot(y, y)
    expected = 0.18 / dim
    ratio = dprod / expected

    print(f"Dprod = {dprod:.6f}, attendu ~{expected:.6f}, ratio = {ratio:.2f}")
    assert 0.3 < ratio < 3.0, \
        f"Dprod = {dprod:.6f}, trop loin de {expected:.6f} (ratio={ratio:.2f})"


def test_shape_preserved():
    """le output doit avoir la meme shape que l'input"""
    vecs = np.random.randn(5, 256)
    tq = TurboQuantProd(256, bit_width=3)
    compressed = tq.quantize(vecs)
    reconstructed = tq.dequantize(compressed)
    assert reconstructed.shape == (5, 256), f"shape = {reconstructed.shape}"


def test_single_vector():
    """doit marcher avec un seul vecteur shape (d,)"""
    vec = np.random.randn(128)
    tq = TurboQuantProd(128, bit_width=3)
    compressed = tq.quantize(vec)
    reconstructed = tq.dequantize(compressed)
    assert reconstructed.shape == (128,), f"shape = {reconstructed.shape}"


def test_qjl_signs():
    """qjl doit contenir que des -1 et +1, rien d'autre"""
    vecs = np.random.randn(10, 256)
    tq = TurboQuantProd(256, bit_width=3)
    compressed = tq.quantize(vecs)
    qjl = compressed["qjl"]
    unique_vals = set(np.unique(qjl))
    assert unique_vals <= {-1, 1}, f"qjl contient {unique_vals}, on veut que {{-1, +1}}"
