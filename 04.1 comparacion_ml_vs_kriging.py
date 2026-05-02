#%%
"""Comparacion directa ML vs Kriging usando Leave-One-Out Cross-Validation (LOOCV).

Metodologia equivalente al kriging jack-knife:
- Cada punto se predice usando un modelo entrenado con los demas N-1 puntos.
- Las predicciones LOOCV son imparciales y comparables con el kriging.
- Metricas y graficos: real (recpe_og) vs prediccion ML (LOOCV).
"""

import importlib
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR

import config_figuras_tesina as cfg_fig

cfg_fig = importlib.reload(cfg_fig)
setup_figuras_tesina = cfg_fig.setup_figuras_tesina
plot_real_vs_ml_vs_kriging = cfg_fig.plot_real_vs_ml_vs_kriging

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("xgboost no encontrado. Se omitira ese modelo.")


DATA_DIR = Path("data") / "processed"
# Fuente unica: tiene coords originales, recpe_og y recpe_val (kriging)
COMP_PATH = DATA_DIR / "comparacion_original_vs_validacion.csv"

COORD_COLS = ["Este", "Norte", "Cota"]
TARGET_COL = "recpe_og"
KRIGING_COL = "recpe_val"

PLOTS_DIR = Path("imagenes") / "comparacion_ml_kriging"
METRICS_OUTPUT_PATH = DATA_DIR / "metricas_ml_vs_kriging.xlsx"
SHOW_FIGURES = True


# ─── Transformacion gaussiana ─────────────────────────────────────────────────

def build_transformer(series: pd.Series) -> QuantileTransformer:
    """Ajusta QuantileTransformer sobre recpe_og."""
    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(series)),
        output_distribution="normal",
    )
    transformer.fit(series.to_numpy().reshape(-1, 1))
    return transformer


def inverse_transform(transformer: QuantileTransformer, values: np.ndarray) -> np.ndarray:
    """Invierte la transformacion gaussiana a escala original."""
    return transformer.inverse_transform(values.reshape(-1, 1)).ravel()


# ─── Modelos ──────────────────────────────────────────────────────────────────

def get_models() -> list[tuple[str, object, dict]]:
    """Retorna lista de (nombre, estimador_base, param_grid) para los 5 modelos."""
    models = [
        (
            "Ridge",
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        (
            "KNN",
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
            {"model__n_neighbors": [5, 9, 15], "model__weights": ["uniform", "distance"]},
        ),
        (
            "SVR",
            Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
            {"model__C": [1, 10, 100], "model__epsilon": [0.1, 0.5], "model__kernel": ["rbf"]},
        ),
        (
            "RandomForest",
            RandomForestRegressor(n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 10], "min_samples_leaf": [2, 4]},
        ),
        (
            "GradientBoosting",
            GradientBoostingRegressor(),
            {"n_estimators": [150, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        ),
    ]

    if _HAS_XGB:
        models.append((
            "XGBoost",
            XGBRegressor(
                objective="reg:squarederror",
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
            ),
            {"n_estimators": [150, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        ))

    return models


def tune_hyperparams(
    estimator,
    param_grid: dict,
    x: pd.DataFrame,
    y: pd.Series,
) -> object:
    """Selecciona hiperparametros optimos con GridSearch 5-fold.

    Los hiperparametros se buscan con CV 5-fold para eficiencia.
    Luego se aplica LOOCV con esos parametros fijos para la evaluacion final.

    Args:
        estimator: Estimador base (puede ser Pipeline).
        param_grid: Grilla de hiperparametros.
        x: Features de entrenamiento.
        y: Target de entrenamiento.

    Returns:
        Mejor estimador configurado pero NO reentrenado.
    """
    search = GridSearchCV(estimator, param_grid, scoring="r2", cv=5, n_jobs=-1)
    search.fit(x, y)
    return search.best_estimator_


def loocv_predict(
    estimator,
    x: pd.DataFrame,
    y: pd.Series,
) -> np.ndarray:
    """Genera predicciones Leave-One-Out imparciales.

    Cada punto es predicho por un modelo entrenado con los N-1 puntos restantes.
    Nota: Random Forest, GradientBoosting y SVR pueden tardar varios minutos.

    Args:
        estimator: Estimador ya configurado con los mejores hiperparametros.
        x: Features.
        y: Target en escala gaussiana.

    Returns:
        Array de predicciones LOOCV en escala gaussiana.
    """
    return cross_val_predict(estimator, x, y, cv=LeaveOneOut(), n_jobs=-1)


# ─── Metricas ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcula R2, RMSE y MAPE."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    return {"r2": r2_score(y_true, y_pred), "rmse": rmse, "mape_pct": mape}


# ─── Ejecucion ────────────────────────────────────────────────────────────────

def main() -> None:
    """Evalua 5 modelos ML con LOOCV y compara contra kriging."""
    setup_figuras_tesina()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # — Cargar datos (coords originales + kriging del vecino mas cercano) —
    comp_df = pd.read_csv(COMP_PATH)
    x_all = comp_df[COORD_COLS].copy()
    y_real = comp_df[TARGET_COL].copy()
    y_kriging = comp_df[KRIGING_COL].copy()

    transformer = build_transformer(y_real)
    y_gauss = pd.Series(
        transformer.transform(y_real.to_numpy().reshape(-1, 1)).ravel(),
        index=y_real.index,
    )
    print(f"Puntos: {len(x_all)}  |  Metodo de evaluacion: LOOCV\n")

    # — Entrenar, evaluar con LOOCV y graficar cada modelo —
    all_metrics = []

    for name, estimator, param_grid in get_models():
        print(f"[{name}] Ajustando hiperparametros (GridSearch 5-fold)...")
        best_model = tune_hyperparams(estimator, param_grid, x_all, y_gauss)

        print(f"[{name}] Calculando predicciones LOOCV ({len(x_all)} iteraciones)...")
        pred_gauss_loo = loocv_predict(best_model, x_all, y_gauss)
        pred_og_loo = inverse_transform(transformer, pred_gauss_loo)

        metrics = compute_metrics(y_real, pd.Series(pred_og_loo, index=y_real.index))
        print(f"  R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape_pct']:.2f}%\n")

        all_metrics.append({"modelo": name, **metrics})

        fig = plot_real_vs_ml_vs_kriging(
            y_true_og=y_real,
            y_ml=pd.Series(pred_og_loo, index=y_real.index),
            y_kriging=y_kriging,
            model_name=f"{name} (LOOCV)",
        )
        fig.savefig(PLOTS_DIR / f"{name.lower()}_loocv_vs_kriging.png", dpi=300)
        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)

    # — Tabla resumen —
    metrics_df = (
        pd.DataFrame(all_metrics)
        .sort_values("r2", ascending=False)
        .reset_index(drop=True)
    )
    metrics_df.to_excel(METRICS_OUTPUT_PATH, index=False)

    print("Resumen LOOCV (ordenado por R²  |  metricas: real vs ML):")
    print(
        metrics_df[["modelo", "r2", "rmse", "mape_pct"]]
        .rename(columns={"modelo": "Modelo", "r2": "R²", "rmse": "RMSE", "mape_pct": "MAPE (%)"})
        .assign(**{
            "R²": lambda d: d["R²"].round(4),
            "RMSE": lambda d: d["RMSE"].round(4),
            "MAPE (%)": lambda d: d["MAPE (%)"].round(2),
        })
        .to_string(index=False)
    )
    print(f"\nMetricas: {METRICS_OUTPUT_PATH.resolve()}")
    print(f"Graficos: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()

# %%
