import optuna
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from skops.io import dumps, get_untrusted_types
import pandas as pd
from src.entrenamiento.modelos import construir_pipeline, calcular_metricas

optuna.logging.set_verbosity(optuna.logging.WARNING)

FAMILIAS = ["ridge", "random-forest", "gradient-boosting"]


def sugerir_hiperparametros(trial: optuna.Trial, familia: str) -> dict:
    if familia == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 1000.0, log=True),
            "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"]),
        }
    elif familia == "random-forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
    elif familia == "gradient-boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
    raise ValueError(f"Familia desconocida: {familia}")


def optimizar_y_registrar_familia(
    familia: str,
    particiones: dict,
    n_trials: int = 10,
) -> dict:
    """
    Ejecuta HPO con Optuna en un parent run hpo-<familia>,
    anida trial-NNN y entrena el mejor modelo en <familia>-final.
    Métrica principal de tuning: val_rmse (minimizar).
    """
    X_train, y_train = particiones["X_train"], particiones["y_train"]
    X_val, y_val = particiones["X_val"], particiones["y_val"]
    X_test, y_test = particiones["X_test"], particiones["y_test"]

    parent_run_name = f"hpo-{familia}"

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tag("familia", familia)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("metrica_objetivo", "val_rmse")
        mlflow.log_param("direccion", "minimize")
        mlflow.log_param("sampler", "TPESampler(seed=42)")

        def objective(trial: optuna.Trial) -> float:
            params = sugerir_hiperparametros(trial, familia)
            trial_name = f"trial-{trial.number:03d}"

            with mlflow.start_run(run_name=trial_name, nested=True):
                mlflow.set_tag("familia", familia)
                mlflow.log_params(params)

                pipe = construir_pipeline(familia, params)
                pipe.fit(X_train, y_train)

                preds_val = pipe.predict(X_val)
                metricas_val = calcular_metricas(y_val.values, preds_val)

                mlflow.log_metric("val_rmse", metricas_val["rmse"])
                mlflow.log_metric("val_mae", metricas_val["mae"])
                mlflow.log_metric("val_mape", metricas_val["mape"])

                return metricas_val["rmse"]

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_val_rmse = study.best_value

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_rmse", best_val_rmse)

        # Reentrenar modelo final usando Train + Validación y evaluar en Test
        X_train_full = pd.concat([X_train, X_val], axis=0)
        y_train_full = pd.concat([y_train, y_val], axis=0)

        final_run_name = f"{familia}-final"
        with mlflow.start_run(run_name=final_run_name, nested=True) as final_run:
            mlflow.set_tag("familia", familia)
            mlflow.set_tag("stage", "final_evaluation")
            mlflow.log_params(best_params)

            final_pipe = construir_pipeline(familia, best_params)
            final_pipe.fit(X_train_full, y_train_full)

            preds_test = final_pipe.predict(X_test)
            metricas_test = calcular_metricas(y_test.values, preds_test)

            mlflow.log_metric("val_rmse", best_val_rmse)
            mlflow.log_metric("test_rmse", metricas_test["rmse"])
            mlflow.log_metric("test_mae", metricas_test["mae"])
            mlflow.log_metric("test_mape", metricas_test["mape"])

            input_example = X_test.iloc[:3]
            signature = infer_signature(X_test, preds_test)
            # El pipeline lo entrenamos nosotros: se confía solo en los tipos que contiene.
            tipos_confiables = get_untrusted_types(data=dumps(final_pipe))

            model_info = mlflow.sklearn.log_model(
                sk_model=final_pipe,
                name="model",
                signature=signature,
                input_example=input_example,
                skops_trusted_types=tipos_confiables,
            )

            return {
                "familia": familia,
                "parent_run_id": parent_run.info.run_id,
                "final_run_id": final_run.info.run_id,
                "model_uri": model_info.model_uri,
                "n_trials": n_trials,
                "best_params": best_params,
                "val_rmse": best_val_rmse,
                "test_rmse": metricas_test["rmse"],
                "test_mae": metricas_test["mae"],
                "test_mape": metricas_test["mape"],
                "pipeline": final_pipe,
            }