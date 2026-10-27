import os
import json
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from src.api.schemas import PredictionRequest, PredictionResponse
from src.entrenamiento.datos import LOOKBACK

load_dotenv()

ARTIFACTS_DIR = Path("artifacts")
LOCAL_MODEL_PATH = ARTIFACTS_DIR / "champion_model.pkl"
LOCAL_METADATA_PATH = ARTIFACTS_DIR / "champion_metadata.json"

# Mismo orden de columnas que crear_ventanas_tabulares: lag_14 ... lag_1
FEATURE_NAMES = [f"lag_{i}" for i in range(LOOKBACK, 0, -1)]

estado = {"modelo": None, "metadata": {}, "source": None, "error": None}


def cargar_metadata_local() -> dict:
    if LOCAL_METADATA_PATH.exists():
        with open(LOCAL_METADATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def cargar_modelo() -> None:
    """
    Carga el modelo campeón generado por el flow de Prefect.
    Primero intenta el Model Registry de MLflow (alias @champion) y,
    si falla, usa el artefacto local artifacts/champion_model.pkl.
    """
    metadata = cargar_metadata_local()
    uc_model_name = os.getenv("UC_MODEL_NAME", metadata.get("model_name", "workspace.default.pcd_otono_2026_tubecast"))
    alias_name = os.getenv("MODEL_ALIAS", metadata.get("alias", "champion"))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        model_uri = f"models:/{uc_model_name}@{alias_name}"
        estado["modelo"] = mlflow.sklearn.load_model(model_uri)
        mv = mlflow.MlflowClient().get_model_version_by_alias(uc_model_name, alias_name)
        estado["metadata"] = {
            **metadata,
            "model_name": uc_model_name,
            "version": str(mv.version),
            "familia_ganadora": mv.tags.get("familia", metadata.get("familia_ganadora")),
        }
        estado["source"] = f"mlflow-registry ({model_uri})"
        estado["error"] = None
        return
    except Exception as e:
        estado["error"] = f"Registry no disponible: {e}"

    if LOCAL_MODEL_PATH.exists():
        estado["modelo"] = joblib.load(LOCAL_MODEL_PATH)
        estado["metadata"] = metadata
        estado["source"] = f"local ({LOCAL_MODEL_PATH})"
    else:
        estado["error"] = (
            f"{estado['error']}. Tampoco existe {LOCAL_MODEL_PATH}: "
            "ejecuta primero el flow de entrenamiento (python -m src.entrenamiento.flow)."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()
    yield


app = FastAPI(
    title="TubeCast API",
    description="Pronóstico diario de visualizaciones del canal MLB.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if estado["modelo"] is not None else "sin_modelo",
        "model_loaded": estado["modelo"] is not None,
        "model_source": estado["source"],
        "model_version": estado["metadata"].get("version"),
        "model_family": estado["metadata"].get("familia_ganadora"),
        "detail": estado["error"],
    }


@app.post("/reload")
def reload_model() -> dict:
    """Recarga el modelo tras una nueva ejecución del flow de Prefect."""
    cargar_modelo()
    return health()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if estado["modelo"] is None:
        raise HTTPException(status_code=503, detail=estado["error"] or "Modelo no cargado.")

    X = pd.DataFrame([request.lags_14_dias], columns=FEATURE_NAMES, dtype="float64")
    pred = max(float(estado["modelo"].predict(X)[0]), 0.0)

    promedio = sum(request.lags_14_dias) / len(request.lags_14_dias)
    cambio = (pred - promedio) / promedio * 100 if promedio > 0 else 0.0
    tendencia = "por encima" if cambio >= 0 else "por debajo"

    metadata = estado["metadata"]
    return PredictionResponse(
        predicted_views=pred,
        predicted_views_formatted=f"{pred:,.0f}",
        interpretation=(
            f"Se estima que los videos publicados el día t acumularán {pred:,.0f} vistas en total, "
            f"un {abs(cambio):.1f}% {tendencia} del promedio de los últimos {LOOKBACK} días ({promedio:,.0f})."
        ),
        model_name=metadata.get("model_name", "desconocido"),
        model_version=str(metadata.get("version", "desconocida")),
        model_family=metadata.get("familia_ganadora", "desconocida"),
        model_source=estado["source"],
    )
