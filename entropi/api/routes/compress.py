import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from entropi.api.auth import verify_api_key
from entropi.api.models.requests import CompressRequest
from entropi.api.models.responses import CompressResponse
from entropi.core.turboquant_mse import TurboQuantMSE
from entropi.core.turboquant_prod import TurboQuantProd

router = APIRouter()


@router.post("/v1/compress", response_model=CompressResponse)
async def compress(
    request: CompressRequest,
    api_key: str = Depends(verify_api_key),
):
    try:
        vectors = np.array(request.vectors, dtype=np.float32)
        dim = vectors.shape[-1]

        if request.mode == "prod":
            tq = TurboQuantProd(dim, request.bit_width)
        else:
            tq = TurboQuantMSE(dim, request.bit_width)

        compressed = tq.quantize(vectors)

        # numpy -> json-serializable
        serialized = {
            "rotation_seed": int(compressed["rotation_seed"]),
            "bit_width": int(compressed["bit_width"]),
            "dim": int(compressed["dim"]),
            "mode": request.mode,
            "norms": compressed["norms"].tolist(),
        }

        if request.mode == "prod":
            serialized["indices"] = compressed["mse_indices"].tolist()
            serialized["qjl"] = compressed["qjl"].tolist()
            serialized["residual_norm"] = compressed["residual_norm"].tolist()
        else:
            serialized["indices"] = compressed["indices"].tolist()

        return CompressResponse(
            compressed=serialized,
            original_dim=dim,
            n_vectors=len(request.vectors),
            compressed_bits_per_dim=float(request.bit_width),
            compression_ratio=32.0 / request.bit_width,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
