# Entropi — CLAUDE.md

## Ce qu'on construit

Entropi est une API de compression vectorielle basée sur TurboQuant (Google Research, ICLR 2026).
Elle compresse des embeddings haute dimension (ex: OpenAI 1536 dims) par 4-6x sans perte de précision mesurable sur les dot products.

Le client envoie des vecteurs float32. On retourne des vecteurs compressés à 2-4 bits par dimension.
On ne touche jamais les prompts, les données utilisateur, ou le contenu des requêtes LLM — uniquement des vecteurs mathématiques.

---

## Stack technique

- **Language** : Python 3.11
- **API** : FastAPI + Uvicorn
- **Maths** : NumPy uniquement (pas de PyTorch, pas de TensorFlow)
- **Base de données** : PostgreSQL (API keys + logs d'usage uniquement)
- **Deploy** : Railway via Dockerfile
- **SDK client** : Python pur, zéro dépendance lourde

---

## Architecture du repo

```
entropi/
├── core/
│   ├── __init__.py
│   ├── turboquant_mse.py      ← TurboQuantMSE  — Algorithme 1 du papier
│   ├── turboquant_prod.py     ← TurboQuantProd — Algorithme 2 du papier
│   ├── rotation.py            ← RandomRotation reproductible par seed
│   ├── codebooks.py           ← Codebooks Lloyd-Max précalculés b=1,2,3,4
│   └── utils.py               ← normalisation, find_nearest, helpers
├── api/
│   ├── __init__.py
│   ├── main.py                ← FastAPI app entry point
│   ├── routes/
│   │   ├── compress.py        ← POST /v1/compress
│   │   └── decompress.py      ← POST /v1/decompress
│   ├── models/
│   │   ├── requests.py        ← Pydantic input schemas
│   │   └── responses.py       ← Pydantic output schemas
│   ├── auth.py                ← API key validation
│   └── errors.py              ← error handlers
├── db/
│   ├── database.py            ← connexion PostgreSQL
│   └── api_keys.py            ← CRUD api keys
├── sdk/
│   └── client.py              ← pip install entropi
├── tests/
│   ├── test_mse.py            ← valide chiffres du papier
│   ├── test_prod.py           ← valide unbiased inner product
│   ├── test_api.py            ← tests endpoints HTTP
│   └── benchmark.py           ← perf sur embeddings réels
├── scripts/
│   └── generate_api_key.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## L'algorithme TurboQuant — ce qu'il faut comprendre avant de coder

### TurboQuantMSE (Algorithme 1)

Optimisé pour minimiser l'erreur MSE entre vecteur original et reconstruit.

**Quantization :**
1. Normaliser le vecteur sur la sphère unitaire (stocker la norme)
2. Multiplier par la matrice de rotation Π (générée par QR decomposition sur matrice random)
3. Pour chaque coordonnée du vecteur rotaté, trouver le centroïde le plus proche dans le codebook
4. Stocker les indices (b bits par dimension)

**Dequantization :**
1. Récupérer les centroids correspondant aux indices
2. Multiplier par Π^T (rotation inverse)
3. Rescaler par la norme originale

**Garantie du papier :**
- b=1 : Dmse ≈ 0.36
- b=2 : Dmse ≈ 0.117
- b=3 : Dmse ≈ 0.03
- b=4 : Dmse ≈ 0.009

### TurboQuantProd (Algorithme 2)

Optimisé pour préserver les inner products (dot products). Non biaisé.
Utilise TurboQuantMSE avec b-1 bits + QJL sur le résidu.

**Quantization :**
1. Normaliser le vecteur
2. Appliquer TurboQuantMSE avec bit_width - 1
3. Calculer le résidu : r = x_normalized - dequantize(mse_result)
4. Stocker la norme du résidu ||r||
5. Appliquer QJL sur le résidu : qjl = sign(S @ r) où S ~ N(0,1)
6. Output : (mse_indices, qjl_signs, residual_norm)

**Dequantization :**
1. Reconstruire la partie MSE
2. Reconstruire la correction QJL : sqrt(π/2) / d * S^T @ qjl * residual_norm
3. Additionner les deux
4. Rescaler par la norme originale

**Propriété clé :**
E[<y, decompress(compress(x))>] = <y, x>   ← estimateur non biaisé

### Les codebooks Lloyd-Max

Les codebooks sont précalculés une fois pour la distribution Beta/Gaussienne.
Pour dim élevé (>100), la distribution Beta converge vers N(0, 1/d).

Valeurs du papier (normalisées pour dim=1, à scaler par 1/sqrt(dim)) :
- b=1 : [-0.7979, +0.7979]
- b=2 : [-1.510, -0.453, +0.453, +1.510]
- b=3 : 8 valeurs symétriques
- b=4 : 16 valeurs symétriques

### La matrice de rotation

- Générée par QR decomposition sur matrice aléatoire N(0,1)
- Déterministe via un seed entier
- Le seed est stocké avec les données compressées pour la décompression
- Même seed = même rotation = décompression correcte

---

## Règles de code importantes

### NumPy — vectorisation obligatoire

Toujours traiter les batches de vecteurs, jamais vecteur par vecteur.

```python
# INTERDIT — trop lent
for vec in vectors:
    process(vec)

# OBLIGATOIRE — vectorisé
process_batch(vectors)  # shape (n_vectors, dim)
```

### Shapes NumPy à respecter

```
vectors input    : (n_vectors, dim) ou (dim,) pour un seul vecteur
norms            : (n_vectors,)
rotated          : (n_vectors, dim)
indices          : (n_vectors, dim) dtype uint8
codebook         : (2**bit_width,)
qjl signs        : (n_vectors, dim) dtype int8 — valeurs {-1, +1}
residual_norm    : (n_vectors,)
```

### Gestion single vector vs batch

Toujours supporter les deux :
```python
single = x.ndim == 1
if single:
    x = x[np.newaxis, :]
# ... traitement batch ...
if single:
    result = result[0]
```

### Normalisation

Toujours ajouter epsilon pour éviter division par zéro :
```python
norms = np.linalg.norm(x, axis=-1, keepdims=True)
x_normalized = x / (norms + 1e-8)
```

---

## API — comportement attendu

### POST /v1/compress

**Request :**
```json
{
  "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "bit_width": 3,
  "mode": "prod"
}
```

**Response :**
```json
{
  "compressed": {
    "indices": [[2, 5, 1, ...], ...],
    "qjl": [[1, -1, 1, ...], ...],
    "residual_norm": [0.023, 0.031],
    "rotation_seed": 42891,
    "bit_width": 3,
    "mode": "prod",
    "dim": 1536
  },
  "original_dim": 1536,
  "compressed_bits_per_dim": 3.0,
  "compression_ratio": 5.33
}
```

### POST /v1/decompress

**Request :** le champ `compressed` de la response ci-dessus

**Response :**
```json
{
  "vectors": [[0.099, 0.201, ...], ...]
}
```

### Auth

Header : `X-API-Key: ent_live_xxxxx`
Toutes les routes nécessitent une API key valide sauf `/health`

### Endpoints publics

- `GET /health` — status check, pas d'auth
- `GET /` — info API, pas d'auth

---

## Base de données

PostgreSQL minimal. Deux tables seulement.

```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID REFERENCES api_keys(id),
    endpoint VARCHAR(50),
    n_vectors INTEGER,
    dim INTEGER,
    bit_width INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

Ne jamais stocker les vecteurs en base. Uniquement les métadonnées d'usage.

---

## Variables d'environnement

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/entropi
API_ENV=development   # ou production
LOG_LEVEL=INFO
```

---

## Tests — chiffres à valider absolument

Ces tests doivent passer avant tout déploiement.

### test_mse.py

```python
def test_mse_distortion():
    """
    Valide Theorem 1 du papier.
    Dmse doit être proche des valeurs théoriques.
    """
    dim = 1536
    n_vectors = 10000
    
    # Tolérance : 20% autour des valeurs du papier
    expected = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    
    for bit_width, expected_dmse in expected.items():
        vectors = np.random.randn(n_vectors, dim)
        tq = TurboQuantMSE(dim, bit_width)
        compressed = tq.quantize(vectors)
        reconstructed = tq.dequantize(compressed)
        dmse = np.mean(np.sum((vectors/np.linalg.norm(vectors, axis=1, keepdims=True) - reconstructed/np.linalg.norm(reconstructed, axis=1, keepdims=True))**2, axis=1))
        assert abs(dmse - expected_dmse) / expected_dmse < 0.2
```

### test_prod.py

```python
def test_inner_product_unbiased():
    """
    Valide que TurboQuantProd est non biaisé.
    E[<y, decompress(compress(x))>] = <y, x>
    """
    dim = 1536
    n_trials = 1000
    
    x = np.random.randn(dim)
    x = x / np.linalg.norm(x)
    y = np.random.randn(dim)
    
    true_ip = np.dot(y, x)
    estimated_ips = []
    
    for _ in range(n_trials):
        tq = TurboQuantProd(dim, bit_width=3)
        compressed = tq.quantize(x)
        reconstructed = tq.dequantize(compressed)
        estimated_ips.append(np.dot(y, reconstructed))
    
    mean_estimated = np.mean(estimated_ips)
    # Biais doit être < 1% de la vraie valeur
    assert abs(mean_estimated - true_ip) / (abs(true_ip) + 1e-8) < 0.01
```

---

## Deploy Railway

Le Dockerfile expose le port 8000.
Railway détecte automatiquement le Dockerfile et déploie.
Variables d'environnement à configurer dans Railway dashboard.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Ce qu'on ne fait PAS en V1

- Pas de dashboard frontend
- Pas de billing automatique
- Pas de rate limiting complexe
- Pas de multi-tenancy
- Pas d'optimisations SIMD ou AVX2 (Python suffit pour la V1)
- Pas de support Anthropic ou Cohere (OpenAI embeddings d'abord)
- Pas de mémoire persistante cross-session (c'est la V2)

---

## Ordre de développement recommandé

1. `core/rotation.py` — RandomRotation avec seed
2. `core/codebooks.py` — Lloyd-Max b=1,2,3,4
3. `core/utils.py` — normalise, find_nearest
4. `core/turboquant_mse.py` — Algorithme 1
5. `tests/test_mse.py` — VALIDER les chiffres du papier avant de continuer
6. `core/turboquant_prod.py` — Algorithme 2
7. `tests/test_prod.py` — VALIDER le non-biais
8. `api/models/` — Pydantic schemas
9. `api/auth.py` — API key validation
10. `api/routes/compress.py` — POST /v1/compress
11. `api/routes/decompress.py` — POST /v1/decompress
12. `api/main.py` — FastAPI app
13. `db/` — PostgreSQL
14. `sdk/client.py` — SDK Python
15. `Dockerfile` + `docker-compose.yml`
16. `tests/benchmark.py` — benchmark final

---

## Contexte business

Entropi est une API de compression vectorielle.
Le client envoie ses embeddings, on les compresse, il les stocke 6x moins cher.

**Cible** : startups YC et AI builders qui utilisent des vector databases (Pinecone, Weaviate, pgvector).

**Pitch** : "Compresse tes embeddings par 6 avec zéro perte de précision. Une API, trois lignes de code."


## Référence papier

Le papier TurboQuant est disponible dans paper/turbo-quant-algo.pdf
Consulte le pour les algorithmes exacts, les pseudocodes, et les valeurs
numériques des theoremes 1, 2 et 3.

Points clés à retenir du papier :
- Algorithme 1 page 10 → TurboQuantMSE
- Algorithme 2 page 12 → TurboQuantProd
- Table 1 page 20 → benchmarks LongBench
- Table 2 page 20 → temps d'indexation
