import os
import sys
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
MIN_DIAS_SERIE = 180


@task(name="1_cargar_y_validar_datos")
def task_cargar_datos() -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Cargando y validando datos de la MLB...")
    df_raw = descargar_datos_mlb()
    if df_raw.empty or "views" not in df_raw.columns:
        raise ValueError("El dataset crudo está vacío o no contiene la columna 'views'.")
    if df_raw["views"].isna().any() or (df_raw["views"] < 0).any():
        raise ValueError("La columna 'views' contiene nulos o valores negativos.")
    if df_raw["video_id"].duplicated().any():
        logger.warning("Se encontraron video_id duplicados; se conserva la primera aparición.")
        df_raw = df_raw.drop_duplicates(subset="video_id", keep="first")
    logger.info(f"{len(df_raw)} videos entre {df_raw['date'].min():%Y-%m-%d} y {df_raw['date'].max():%Y-%m-%d}")
    return df_raw


@task(name="2_preparar_serie_y_particiones")
def task_preparar_datos(df_raw: pd.DataFrame) -> dict:
    logger = get_run_logger()
    logger.info("Construyendo serie diaria y ventanas deslizantes...")
    daily = construir_serie_diaria(df_raw)
    if len(daily) < MIN_DIAS_SERIE:
        raise ValueError(f"La serie diaria tiene {len(daily)} días; se requieren al menos {MIN_DIAS_SERIE}.")
    X, y, dates = crear_ventanas_tabulares(daily, lookback=LOOKBACK)
    particiones = particionar_serie(X, y, dates)
    for nombre in ("train", "val", "test"):
        fechas = particiones[f"dates_{nombre}"]
        logger.info(f"{nombre}: {len(fechas)} filas ({fechas.min():%Y-%m-%d} a {fechas.max():%Y-%m-%d})")
    return particiones


@task(name="3_hpo_y_modelo_final_por_familia")
def task_hpo_familia(familia: str, particiones: dict, n_trials: int) -> dict:
    logger = get_run_logger()
    logger.info(f"Iniciando HPO con Optuna para la familia: {familia} ({n_trials} trials)")
    resultado = optimizar_y_registrar_familia(familia=familia, particiones=particiones, n_trials=n_trials)
    logger.info(f"{familia}: best_val_rmse={resultado['val_rmse']:,.0f} test_rmse={resultado['test_rmse']:,.0f}")
    return resultado


@task(name="4_comparar_y_seleccionar")
def task_comparar_modelos(resultados: list[dict]) -> dict:
    logger = get_run_logger()
    logger.info("Comparando el mejor modelo de cada familia en el conjunto de prueba...")

    df_comp = pd.DataFrame([
        {
            "familia": r["familia"],
            "final_run_id": r["final_run_id"],
            "n_trials": r["n_trials"],
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
    logger.info("\n" + df_comp.drop(columns=["best_params"]).to_string(index=False))

    ganador = min(resultados, key=lambda x: x["test_rmse"])
    logger.info(f"Familia seleccionada: {ganador['familia']} (menor test_rmse)")
    return ganador


@task(name="5_registrar_en_registry_y_asignar_champion")
def task_registrar_modelos(resultados: list[dict], ganador: dict) -> dict:
    logger = get_run_logger()

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
            etiquetas = {
                "familia": r["familia"],
                "final_run_id": r["final_run_id"],
                "parent_run_id": r["parent_run_id"],
                "val_rmse": f"{r['val_rmse']:.2f}",
                "test_rmse": f"{r['test_rmse']:.2f}",
                "test_mae": f"{r['test_mae']:.2f}",
                "test_mape": f"{r['test_mape']:.2f}",
            }
            for clave, valor in etiquetas.items():
                client.set_model_version_tag(uc_model_name, mv.version, clave, valor)
            client.update_model_version(
                name=uc_model_name,
                version=mv.version,
                description=(
                    f"Familia {r['familia']} (run {r['final_run_id']}). "
                    f"test_rmse={r['test_rmse']:,.0f}, test_mae={r['test_mae']:,.0f}."
                ),
            )
            logger.info(f"Registrada versión {mv.version} ({r['familia']}) en {uc_model_name}")

        version_ganadora = str(versiones_registradas[ganador["familia"]])
        client.set_registered_model_alias(
            name=uc_model_name,
            alias=alias_name,
            version=version_ganadora,
        )
        logger.info(f"Alias @{alias_name} asignado a la versión {version_ganadora} ({ganador['familia']})")
    except Exception as e:
        logger.warning(f"Aviso Registry ({tracking_uri}): {e}. Usando artefacto local de respaldo.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ganador["pipeline"], ARTIFACTS_DIR / "champion_model.pkl")
    metadata = {
        "model_name": uc_model_name,
        "alias": alias_name,
        "version": version_ganadora,
        "familia_ganadora": ganador["familia"],
        "final_run_id": ganador["final_run_id"],
        "tracking_uri": tracking_uri,
        "test_rmse": ganador["test_rmse"],
        "test_mae": ganador["test_mae"],
        "test_mape": ganador["test_mape"],
    }
    with open(ARTIFACTS_DIR / "champion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


@flow(name="tubecast-training-pipeline", log_prints=True)
def ejecutar_pipeline_entrenamiento(n_trials: int = 20) -> dict:
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

    ganador = task_comparar_modelos(resultados)
    resumen = task_registrar_modelos(resultados, ganador)
    print("Pipeline finalizado con éxito. Resumen del modelo campeón:", resumen)
    return resumen


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("N_TRIALS", "20"))
    ejecutar_pipeline_entrenamiento(n_trials=n_trials)
