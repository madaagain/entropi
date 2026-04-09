import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from entropi.api.auth import verify_api_key
from entropi.api.models.requests import DecompressRequest
from entropi.api.models.responses import DecompressResponse
from entropi.core.turboquant_mse import TurboQuantMSE
from entropi.core.turboquant_prod import TurboQuantProd

router = APIRouter()


@router.post("/v1/decompress", response_model=DecompressResponse)
async def decompress(
    request: DecompressRequest,
    api_key: str = Depends(verify_api_key),
):
    try:
        data = request.compressed
        dim = data["dim"]
        bit_width = data["bit_width"]
        mode = data.get("mode", "prod")
        seed = data["rotation_seed"]

        if mode == "prod":
            tq = TurboQuantProd(dim, bit_width, seed=seed)
            compressed_numpy = {
                "mse_indices": np.array(data["indices"], dtype=np.uint8),
                "qjl": np.array(data["qjl"], dtype=np.int8),
                "residual_norm": np.array(data["residual_norm"], dtype=np.float64),
                "norms": np.array(data["norms"], dtype=np.float64),
                "rotation_seed": seed,
                "bit_width": bit_width,
                "dim": dim,
                "single": False,
            }
        else:
            tq = TurboQuantMSE(dim, bit_width, seed=seed)
            compressed_numpy = {
                "indices": np.array(data["indices"], dtype=np.uint8),
                "norms": np.array(data["norms"], dtype=np.float64),
                "rotation_seed": seed,
                "bit_width": bit_width,
                "dim": dim,
                "single": False,
            }

        vectors = tq.dequantize(compressed_numpy)

        return DecompressResponse(
            vectors=vectors.tolist(),
            n_vectors=len(vectors),
            dim=dim,
        )
    except KeyError as e:
        raise HTTPException(
            status_code=400, detail=f"Missing field in compressed data: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
