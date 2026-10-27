import os
import json
from pathlib import Path


import joblib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task, get_run_logger
from dotenv import load_dotenv

from src.entrenamiento.datos import (
    descargar_datos_mlb,
    construir_serie_diaria,
    crear_ventanas_tabulares,
    particionar_serie,
    LOOKBACK,
)
from src.entrenamiento.hpo import FAMILIAS, optimizar_y_registrar_familia

load_dotenv()

ARTIFACTS_DIR = Path("artifacts")


@task(name="1_cargar_y_validar_datos")
def task_cargar_datos() -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Cargando y validando datos de la MLB...")
    df_raw = descargar_datos_mlb()
    if df_raw.empty or "views" not in df_raw.columns:
        raise ValueError("El dataset crudo está vacío o no contiene la columna 'views'.")
    return df_raw


@task(name="2_preparar_serie_y_particiones")
def task_preparar_datos(df_raw: pd.DataFrame) -> dict:
    logger = get_run_logger()
    logger.info("Construyendo serie diaria y ventanas deslizantes...")
    daily = construir_serie_diaria(df_raw)
    X, y, dates = crear_ventanas_tabulares(daily, lookback=LOOKBACK)
    return particionar_serie(X, y, dates)


@task(name="3_ejecutar_hpo_familia")
def task_hpo_familia(familia: str, particiones: dict, n_trials: int) -> dict:
    logger = get_run_logger()
    logger.info(f"Iniciando HPO con Optuna para la familia: {familia}")
    return optimizar_y_registrar_familia(familia=familia, particiones=particiones, n_trials=n_trials)


@task(name="4_evaluar_y_registrar_champion")
def task_registrar_modelos(resultados: list[dict]) -> dict:
    logger = get_run_logger()
    logger.info("Evaluando modelos y registrando el campeón...")
    
    df_comp = pd.DataFrame([
        {
            "familia": r["familia"],
            "final_run_id": r["final_run_id"],
            "val_rmse": r["val_rmse"],
            "test_rmse": r["test_rmse"],
            "test_mae": r["test_mae"],
            "test_mape": r["test_mape"],
            "best_params": json.dumps(r["best_params"]),
        }
        for r in resultados
    ]).sort_values("test_rmse")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)

    ganador = min(resultados, key=lambda x: x["test_rmse"])

    uc_model_name = os.getenv("UC_MODEL_NAME", "workspace.default.pcd_otono_2026_tubecast")
    alias_name = os.getenv("MODEL_ALIAS", "champion")
    tracking_uri = mlflow.get_tracking_uri()

    version_ganadora = "local-v1"
    client = MlflowClient()

    try:
        versiones_registradas = {}
        for r in resultados:
            mv = mlflow.register_model(model_uri=r["model_uri"], name=uc_model_name)
            versiones_registradas[r["familia"]] = mv.version

        version_ganadora = str(versiones_registradas[ganador["familia"]])
        client.set_registered_model_alias(
            name=uc_model_name,
            alias=alias_name,
            version=version_ganadora,
        )
    except Exception as e:
        logger.warning(f"Aviso Registry ({tracking_uri}): {e}. Usando artefacto local de respaldo.")

    joblib.dump(ganador["pipeline"], ARTIFACTS_DIR / "champion_model.pkl")
    metadata = {
        "model_name": uc_model_name,
        "alias": alias_name,
        "version": version_ganadora,
        "familia_ganadora": ganador["familia"],
        "final_run_id": ganador["final_run_id"],
        "test_rmse": ganador["test_rmse"],
        "test_mae": ganador["test_mae"],
        "test_mape": ganador["test_mape"],
    }
    with open(ARTIFACTS_DIR / "champion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


@flow(name="tubecast-training-pipeline", log_prints=True)
def ejecutar_pipeline_entrenamiento(n_trials: int = 5) -> dict:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/pcd-otono-2026-proyecto-tubecast")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(experiment_name)

    df_raw = task_cargar_datos()
    particiones = task_preparar_datos(df_raw)

    resultados = []
    for familia in FAMILIAS:
        res = task_hpo_familia(familia=familia, particiones=particiones, n_trials=n_trials)
        resultados.append(res)

    resumen = task_registrar_modelos(resultados)
    print("Pipeline finalizado con éxito. Resumen del modelo campeón:", resumen)
    return resumen


if __name__ == "__main__":
    ejecutar_pipeline_entrenamiento(n_trials=5)