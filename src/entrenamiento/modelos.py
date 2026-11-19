import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_SEED = 42


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calcula RMSE, MAE y MAPE truncando predicciones negativas a 0.
    El MAPE se calcula solo sobre los días con vistas > 0: dividir entre un cero
    (días sin publicaciones) produce porcentajes sin sentido.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_clean = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred_clean)))
    mae_val = float(mean_absolute_error(y_true, y_pred_clean))
    con_vistas = y_true > 0
    if con_vistas.any():
        mape_val = float(np.mean(np.abs((y_true[con_vistas] - y_pred_clean[con_vistas]) / y_true[con_vistas])) * 100.0)
    else:
        mape_val = float("nan")
    return {"rmse": rmse_val, "mae": mae_val, "mape": mape_val}


def construir_pipeline(familia: str, params: dict) -> Pipeline:
    """Construye un Pipeline (Preprocesamiento + Modelo) para la familia indicada."""
    if familia == "ridge":
        modelo = Ridge(
            alpha=params.get("alpha", 1.0),
            solver=params.get("solver", "auto"),
            random_state=RANDOM_SEED,
        )
    elif familia == "random-forest":
        modelo = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 10)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif familia == "gradient-boosting":
        modelo = HistGradientBoostingRegressor(
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_iter=int(params.get("max_iter", 200)),
            max_depth=params.get("max_depth"),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            l2_regularization=float(params.get("l2_regularization", 0.0)),
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Familia no soportada: {familia}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", modelo),
    ])