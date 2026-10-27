"""Consultas a MLflow reutilizadas por los notebooks y el informe."""
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()


def configurar_mlflow() -> dict:
    """Aplica la configuración de .env y devuelve las referencias del proyecto."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    return {
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
        "experimento": os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/pcd-otono-2026-proyecto-tubecast"),
        "modelo": os.getenv("UC_MODEL_NAME", "workspace.default.pcd_otono_2026_tubecast"),
        "alias": os.getenv("MODEL_ALIAS", "champion"),
    }


def runs_ultima_ejecucion(experimento: str) -> dict[str, pd.DataFrame]:
    """
    Devuelve los runs de la ejecución más reciente de cada familia:
    el parent hpo-<familia>, sus trials trial-NNN y el run <familia>-final.
    """
    exp = mlflow.get_experiment_by_name(experimento)
    if exp is None:
        raise ValueError(f"No existe el experimento {experimento}. Ejecuta primero el flow.")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=5000)
    nombre = runs["tags.mlflow.runName"].fillna("")

    padres = (
        runs[nombre.str.startswith("hpo-")]
        .sort_values("start_time")
        .groupby("tags.familia")
        .tail(1)
    )
    hijos = runs[runs["tags.mlflow.parentRunId"].isin(padres["run_id"])]
    nombre_hijos = hijos["tags.mlflow.runName"]
    trials = hijos[nombre_hijos.str.startswith("trial-")].copy()
    trials["trial"] = trials["tags.mlflow.runName"].str[-3:].astype(int)
    finales = hijos[nombre_hijos.str.endswith("-final")]
    return {
        "padres": padres,
        "trials": trials.sort_values(["tags.familia", "trial"]),
        "finales": finales,
        "experiment_id": exp.experiment_id,
    }


def columnas_params(df: pd.DataFrame) -> list[str]:
    """Columnas params.* que tienen algún valor en df."""
    return [c for c in df.columns if c.startswith("params.") and df[c].notna().any()]


def tabla_comparacion(finales: pd.DataFrame) -> pd.DataFrame:
    """Mejor modelo de cada familia evaluado en test bajo la misma partición."""
    filas = []
    for _, r in finales.iterrows():
        params = {c.removeprefix("params."): r[c] for c in columnas_params(finales) if pd.notna(r[c])}
        filas.append({
            "familia": r["tags.familia"],
            "run": r["tags.mlflow.runName"],
            "val_rmse": r["metrics.val_rmse"],
            "test_rmse": r["metrics.test_rmse"],
            "test_mae": r["metrics.test_mae"],
            "test_mape": r["metrics.test_mape"],
            "mejores_params": params,
            "run_id": r["run_id"],
        })
    return pd.DataFrame(filas).sort_values("test_rmse").reset_index(drop=True)


def versiones_registradas(modelo: str) -> pd.DataFrame:
    """Versiones del modelo registrado con su familia, run de procedencia, métricas y alias."""
    client = MlflowClient()
    filas = []
    for mv in client.search_model_versions(f"name='{modelo}'"):
        detalle = client.get_model_version(modelo, mv.version)
        filas.append({
            "version": int(detalle.version),
            "familia": detalle.tags.get("familia"),
            "run_id": detalle.run_id or detalle.tags.get("final_run_id"),
            "test_rmse": float(detalle.tags["test_rmse"]) if "test_rmse" in detalle.tags else None,
            "test_mae": float(detalle.tags["test_mae"]) if "test_mae" in detalle.tags else None,
            "aliases": ", ".join(detalle.aliases),
            "creada": pd.to_datetime(detalle.creation_timestamp, unit="ms"),
        })
    return pd.DataFrame(filas).sort_values("version").reset_index(drop=True)
