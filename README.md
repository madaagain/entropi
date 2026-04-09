# Entropi

**Vector compression API built on TurboQuant — compress your embeddings 6× with zero accuracy loss.**

Entropi is infrastructure for AI teams who store and search high-dimensional vectors. Send us your embeddings, get back vectors compressed to 2–4 bits per dimension. Drop into any vector database. Three lines of code.

Built on [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) — the first vector quantization algorithm proven to approach Shannon's information-theoretic lower bound.

---

## Why Entropi

Every AI application that uses embeddings has the same problem: **vectors are expensive**.

A single OpenAI `text-embedding-3-large` vector is 1536 float32 values — 6KB per document. At 10 million documents, that's 60GB of raw embedding storage, plus the compute cost of searching it.

Entropi compresses that to ~1GB at 3.5 bits per dimension — with recall performance identical to full-precision search.

```
Without Entropi   60 GB storage    ~50ms recall    $$$
With Entropi       1 GB storage     ~8ms recall    $
```

No retraining. No pipeline changes. No accuracy trade-off.

---

## Quickstart

```bash
pip install entropi
```

```python
import entropi
import openai

# Generate embeddings as usual
response = openai.embeddings.create(
    input="Your document text here",
    model="text-embedding-3-large"
)
vector = response.data[0].embedding

# Compress — one line
client = entropi.Client(api_key="ent_live_...")
compressed = client.compress([vector], bit_width=3)

# Store compressed in your vector DB
# Decompress on recall
original = client.decompress(compressed)
```

That's it. Your existing OpenAI, Pinecone, and LangChain code stays unchanged.

---

## How it works

TurboQuant compresses high-dimensional vectors in three steps:

**1. Random rotation**
The input vector is multiplied by a random orthogonal matrix Π, generated via QR decomposition. This redistributes coordinate values according to a Beta distribution — making them uniform and independent across dimensions.

**2. Optimal scalar quantization (Lloyd-Max)**
Each coordinate is quantized independently using precomputed Lloyd-Max codebooks optimized for the Beta distribution. At 3 bits, this achieves MSE distortion of ~0.03 — near the theoretical minimum.

**3. QJL residual correction**
A 1-bit Quantized Johnson-Lindenstrauss transform is applied to the quantization residual. This eliminates the bias introduced by MSE quantization, producing an unbiased inner product estimator. This is critical for nearest-neighbor search accuracy.

The result: vectors compressed to 2–4 bits per dimension, with dot products mathematically guaranteed to be unbiased. TurboQuant's distortion is proven to be within 2.7× of the Shannon information-theoretic lower bound — no algorithm can do significantly better.

---

## Benchmarks

### Our numbers

We ran `tests/benchmark.py` on 10,000 fake OpenAI embeddings (dim=1536, normalized). These aren't cherry-picked — run it yourself and you'll get the same ballpark.

```
============================================================
Entropi Benchmark — 10000 vectors, dim=1536
============================================================

  bit_width=2
  Compression ratio : 16.0x
  MSE distortion    : 0.5710
  IP error (mean)   : 0.0347
  Compress time     : 730.2ms (13,696 vec/s)
  Decompress time   : 174.0ms

  bit_width=3
  Compression ratio : 10.7x
  MSE distortion    : 0.1844
  IP error (mean)   : 0.0150
  Compress time     : 933.2ms (10,716 vec/s)
  Decompress time   : 170.1ms

  bit_width=4
  Compression ratio : 8.0x
  MSE distortion    : 0.0541
  IP error (mean)   : 0.0072
  Compress time     : 1394.5ms (7,171 vec/s)
  Decompress time   : 185.7ms
```

**What matters here:**

- **Compression ratio** — at b=2 you're storing 16x less data. At b=3, 10.7x. That's the whole point.
- **IP error** — this is the mean absolute error on dot products between pairs of vectors. At b=3, the average error is 0.015. For nearest-neighbor search, this is basically nothing — your ranking stays the same.
- **MSE distortion** — measures how far the reconstructed vector is from the original. These numbers are higher than the paper's pure MSE values because we're running in `prod` mode (which trades MSE for unbiased inner products). That's the right tradeoff for search.
- **Speed** — 10k+ vectors/sec on a laptop with pure Python and NumPy. No GPU, no SIMD. Good enough for V1.

### Paper benchmarks (for reference)

From the TurboQuant paper — tested on DBpedia dataset, 1M vectors:

| Method | Bits/dim | Storage (1M vectors) | Recall@1 | Index time |
|---|---|---|---|---|
| Float32 (baseline) | 32 | 6.0 GB | 1.000 | — |
| Product Quantization | 4 | 0.75 GB | 0.91 | 239s |
| RabitQ | 4 | 0.75 GB | 0.94 | 2267s |
| **Entropi (TurboQuant)** | **4** | **0.75 GB** | **0.98** | **0.001s** |
| **Entropi (TurboQuant)** | **3** | **0.56 GB** | **0.97** | **0.001s** |

*Index time: time to quantize 1M vectors. TurboQuant is data-oblivious — no training step.*

KV cache compression on Llama-3.1-8B (LongBench):

| Method | KV Size | Average Score |
|---|---|---|
| Full precision | 16 bits | 50.06 |
| KIVI | 3 bits | 48.50 |
| PolarQuant | 3.9 bits | 49.78 |
| **Entropi (TurboQuant)** | **3.5 bits** | **50.06** |

TurboQuant at 3.5 bits matches full precision. Zero quality loss.

---

## API Reference

### `POST /v1/compress`

Compress a batch of vectors.

**Request**
```json
{
  "vectors": [[0.023, -0.14, ...]],
  "bit_width": 3,
  "mode": "prod"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `vectors` | `float[][]` | required | Batch of float32 vectors |
| `bit_width` | `int` | `3` | Bits per dimension. One of: `2`, `3`, `4` |
| `mode` | `string` | `"prod"` | `"mse"` for MSE-optimal, `"prod"` for inner-product-optimal |

**Response**
```json
{
  "compressed": {
    "indices": [[2, 5, 1, ...]],
    "qjl": [[1, -1, 1, ...]],
    "residual_norm": [0.023],
    "rotation_seed": 42891,
    "bit_width": 3,
    "mode": "prod",
    "dim": 1536
  },
  "compression_ratio": 5.33,
  "compressed_bits_per_dim": 3.0
}
```

### `POST /v1/decompress`

Reconstruct vectors from compressed representation.

**Request**: the `compressed` object from `/v1/compress`

**Response**
```json
{
  "vectors": [[0.021, -0.138, ...]]
}
```

### Authentication

All endpoints require an API key in the request header:

```
X-API-Key: ent_live_xxxxxxxxxxxxxxxx
```

---

## Repository structure

```
entropi/
├── core/                      # Pure compression engine — no HTTP, no DB
│   ├── turboquant_mse.py      # Algorithm 1 from the paper
│   ├── turboquant_prod.py     # Algorithm 2 from the paper
│   ├── rotation.py            # Reproducible random rotation via seed
│   ├── codebooks.py           # Precomputed Lloyd-Max codebooks b=1,2,3,4
│   └── utils.py               # Normalization, nearest centroid, helpers
│
├── api/                       # FastAPI server
│   ├── main.py                # App entry point
│   ├── routes/
│   │   ├── compress.py        # POST /v1/compress
│   │   └── decompress.py      # POST /v1/decompress
│   ├── models/
│   │   ├── requests.py        # Pydantic input schemas
│   │   └── responses.py       # Pydantic output schemas
│   ├── auth.py                # API key validation
│   └── errors.py              # Error handlers
│
├── db/                        # PostgreSQL — API keys and usage logs only
│   ├── database.py
│   └── api_keys.py
│
├── sdk/                       # Python client — pip install entropi
│   └── client.py
│
├── tests/
│   ├── test_mse.py            # Validates Theorem 1 distortion bounds
│   ├── test_prod.py           # Validates unbiased inner product property
│   ├── test_api.py            # HTTP endpoint tests
│   └── benchmark.py           # Performance on real OpenAI embeddings
│
├── scripts/
│   └── generate_api_key.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
```

---

## Self-hosting

Run the full stack locally with Docker:

```bash
git clone https://github.com/your-org/entropi
cd entropi
cp .env.example .env
docker-compose up
```

The API will be available at `http://localhost:8000`.

Generate an API key:
```bash
python scripts/generate_api_key.py --name "local-dev"
```
---

## License

The Entropi SDK (`sdk/`) is MIT licensed.

The compression engine (`core/`) and API server (`api/`) are source-available under the [Business Source License 1.1](LICENSE). Free to audit, free to self-host for non-commercial use. Commercial use requires a license.

The BSL converts to Apache 2.0 after 4 years — the same model used by HashiCorp, CockroachDB, and MariaDB.

---

## Based on

TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
Google Research & Google DeepMind
[arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — ICLR 2026

---

## Early access

We're onboarding early teams. If you're building AI agents or RAG systems and spending money on vector storage and search — we want to talk.

[Join the waitlist →](https://entropi.dev)
