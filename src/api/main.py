import os
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from src.api.schemas import PredictionRequest, PredictionResponse

# override=False: si Docker ya entregó las variables con env_file, no se reemplazan.
load_dotenv(override=False)

logger = logging.getLogger("uvicorn.error")

# Debe coincidir con LOOKBACK de src/entrenamiento/datos.py (mismo procesamiento
# que en entrenamiento: columnas lag_14 ... lag_1, en ese orden).
LOOKBACK = 14
FEATURE_NAMES = [f"lag_{i}" for i in range(LOOKBACK, 0, -1)]

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
LOCAL_MODEL_PATH = ARTIFACTS_DIR / "champion_model.pkl"
LOCAL_METADATA_PATH = ARTIFACTS_DIR / "champion_metadata.json"

estado = {"modelo": None, "metadata": {}, "source": None, "error": None}


def cargar_desde_registry() -> None:
    """Resuelve el alias a una versión concreta y carga exactamente esa versión."""
    uc_model_name = os.getenv("UC_MODEL_NAME", "workspace.default.pcd_otono_2026_tubecast")
    alias_name = os.getenv("MODEL_ALIAS", "champion")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = mlflow.MlflowClient()

    mv = client.get_model_version_by_alias(uc_model_name, alias_name)
    model_uri = f"models:/{uc_model_name}/{mv.version}"
    estado["modelo"] = mlflow.sklearn.load_model(model_uri)
    estado["metadata"] = {
        "model_name": uc_model_name,
        "alias": alias_name,
        "version": str(mv.version),
        "familia_ganadora": mv.tags.get("familia", "desconocida"),
        "run_id": mv.run_id,
    }
    estado["source"] = f"registry {registry_uri}: {uc_model_name}@{alias_name} -> versión {mv.version}"


def cargar_desde_artefacto_local() -> None:
    """
    Respaldo explícito: un artefacto propio transferido de forma privada.
    La versión que se reporta es la del archivo de metadatos que lo acompaña.
    """
    if not LOCAL_MODEL_PATH.exists() or not LOCAL_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Faltan {LOCAL_MODEL_PATH} o {LOCAL_METADATA_PATH}; genera ambos con el flow."
        )
    with open(LOCAL_METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    estado["modelo"] = joblib.load(LOCAL_MODEL_PATH)
    estado["metadata"] = metadata
    estado["source"] = (
        f"artefacto local {LOCAL_MODEL_PATH} (copia de {metadata.get('model_name')} "
        f"versión {metadata.get('version')})"
    )


def cargar_modelo() -> None:
    """
    MODEL_SOURCE=registry (predeterminado): usa el Model Registry; si falla, la API
    queda sin modelo y lo dice en /health, sin cambiar en silencio a otra fuente.
    MODEL_SOURCE=local: usa el artefacto de artifacts/ y lo identifica como tal.
    """
    fuente = os.getenv("MODEL_SOURCE", "registry").strip().lower()
    estado.update({"modelo": None, "metadata": {}, "source": None, "error": None})
    try:
        if fuente == "local":
            cargar_desde_artefacto_local()
        else:
            cargar_desde_registry()
        logger.info(
            "Modelo cargado: %s | familia=%s",
            estado["source"],
            estado["metadata"].get("familia_ganadora"),
        )
    except Exception as e:
        estado["error"] = f"No se pudo cargar el modelo ({fuente}): {e}"
        logger.error(estado["error"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()
    yield


app = FastAPI(
    title="TubeCast API",
    description=(
        "Estimación retrospectiva de las vistas acumuladas por la cohorte de videos "
        "publicados en un día del canal MLB, a partir de las 14 cohortes previas."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if estado["modelo"] is not None else "sin_modelo",
        "model_loaded": estado["modelo"] is not None,
        "model_source": estado["source"],
        "model_name": estado["metadata"].get("model_name"),
        "model_version": estado["metadata"].get("version"),
        "model_family": estado["metadata"].get("familia_ganadora"),
        "detail": estado["error"],
    }


@app.post("/reload")
def reload_model() -> dict:
    """Vuelve a resolver el alias tras una nueva ejecución del flow."""
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
    logger.info(
        "Predicción: %.0f vistas | versión=%s | promedio_entrada=%.0f",
        pred, metadata.get("version"), promedio,
    )
    return PredictionResponse(
        predicted_views=pred,
        predicted_views_formatted=f"{pred:,.0f}",
        interpretation=(
            f"Se estima que la cohorte del día t (videos publicados ese día) acumula {pred:,.0f} vistas "
            f"en la misma instantánea que las entradas; es {abs(cambio):.1f}% {tendencia} del promedio "
            f"de las {LOOKBACK} cohortes previas ({promedio:,.0f}). No son las vistas recibidas el día t."
        ),
        model_name=metadata.get("model_name", "desconocido"),
        model_version=str(metadata.get("version", "desconocida")),
        model_family=metadata.get("familia_ganadora", "desconocida"),
        model_source=estado["source"],
    )
