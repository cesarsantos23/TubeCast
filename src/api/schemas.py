import math
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    lags_14_dias: list[float] = Field(
        ...,
        min_length=14,
        max_length=14,
        description="Lista ordenada de vistas diarias de los últimos 14 días (día t-14 al día t-1).",
        examples=[[
            3529320.0, 3816846.0, 1660823.0, 3210395.0, 3791115.0,
            2218409.0, 1034500.0, 3077739.0, 3434652.0, 838785.0,
            1615544.0, 1548594.0, 2100450.0, 1950300.0
        ]],
    )

    @field_validator("lags_14_dias")
    @classmethod
    def validar_vistas_no_negativas(cls, v: list[float]) -> list[float]:
        for idx, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"El valor en la posición {idx} no es un número válido.")
            if val < 0:
                raise ValueError(f"Las vistas no pueden ser negativas (valor recibido: {val} en posición {idx}).")
        return v


class PredictionResponse(BaseModel):
    predicted_views: float
    predicted_views_formatted: str
    interpretation: str
    model_name: str
    model_version: str
    model_family: str
    model_source: str