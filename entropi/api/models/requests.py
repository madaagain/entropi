from pydantic import BaseModel, field_validator, model_validator


class CompressRequest(BaseModel):
    vectors: list[list[float]]
    bit_width: int = 3
    mode: str = "prod"

    @field_validator("bit_width")
    @classmethod
    def check_bit_width(cls, v):
        if v not in {2, 3, 4}:
            raise ValueError("bit_width must be 2, 3 or 4")
        return v

    @field_validator("mode")
    @classmethod
    def check_mode(cls, v):
        if v not in {"mse", "prod"}:
            raise ValueError("mode must be 'mse' or 'prod'")
        return v

    @field_validator("vectors")
    @classmethod
    def check_vectors_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("vectors must not be empty")
        return v

    @model_validator(mode="after")
    def check_consistent_dims(self):
        if not self.vectors:
            return self
        dim = len(self.vectors[0])
        if dim < 64:
            raise ValueError(f"dimension must be >= 64, got {dim}")
        for i, vec in enumerate(self.vectors):
            if len(vec) != dim:
                raise ValueError(
                    f"vector {i} has dim {len(vec)}, expected {dim}"
                )
        return self


class DecompressRequest(BaseModel):
    compressed: dict
