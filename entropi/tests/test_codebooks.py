import numpy as np
from entropi.core.codebooks import get_codebook, CODEBOOK_DATA


def test_codebook_size():
    """chaque codebook doit avoir 2^b niveaux, sinon on quantize mal"""
    for b in [1, 2, 3, 4]:
        cb = get_codebook(b, dim=1536)
        assert len(cb) == 2**b, f"b={b} : attendu {2**b} niveaux, got {len(cb)}"


def test_codebook_symmetric():
    """les codebooks doivent etre symetriques autour de 0
    vu que N(0,1) est symetrique ca fait sens"""
    for b in [1, 2, 3, 4]:
        cb = get_codebook(b, dim=1536)
        # si on flip et on negatie on doit retomber sur les memes valeurs
        assert np.allclose(cb, -cb[::-1], atol=1e-8), f"b={b} : pas symetrique"


def test_codebook_sorted():
    """les centroids doivent etre tries, sinon find_nearest va galerer"""
    for b in [1, 2, 3, 4]:
        cb = get_codebook(b, dim=1536)
        assert np.all(cb[:-1] < cb[1:]), f"b={b} : pas trie"


def test_codebook_scales_with_dim():
    """plus la dim est grande, plus les valeurs sont petites
    parce qu'on divise par sqrt(dim)"""
    for b in [1, 2, 3, 4]:
        cb_small = get_codebook(b, dim=100)
        cb_big = get_codebook(b, dim=1536)
        assert np.max(np.abs(cb_big)) < np.max(np.abs(cb_small)), \
            f"b={b} : dim=1536 devrait avoir des valeurs plus petites que dim=100"


def test_invalid_bit_width():
    """on doit lever une erreur si bit_width est pas dans {1,2,3,4}"""
    try:
        get_codebook(5, dim=1536)
        assert False, "aurait du lever ValueError"
    except ValueError:
        pass


def test_codebook_values_sane():
    """juste un sanity check que les valeurs brutes sont pas n'importe quoi
    pour b=1 on devrait avoir environ [-0.798, +0.798] (valeurs du papier)"""
    cb = CODEBOOK_DATA[1]
    assert np.allclose(np.abs(cb), 0.7979, atol=0.01), \
        f"b=1 centroids = {cb}, attendu ~[-0.798, +0.798]"
