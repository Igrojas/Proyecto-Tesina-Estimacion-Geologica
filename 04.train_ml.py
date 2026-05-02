#%%
"""Entrenamiento rapido de KNN y XGBoost por archivo de cluster."""

import importlib
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import QuantileTransformer

import config_figuras_tesina as cfg_fig

cfg_fig = importlib.reload(cfg_fig)
setup_figuras_tesina = cfg_fig.setup_figuras_tesina
figsize_cm = cfg_fig.figsize_cm
plot_real_vs_pred_subplot = cfg_fig.plot_real_vs_pred_subplot

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "No se encontro xgboost. Instala con: pip install xgboost"
    ) from exc


DATA_DIR = Path("data") / "processed"
ORG_DATA_PATH = DATA_DIR / "df_rec_peso_pnd25.xlsx"
CLUSTER_FILE_PATTERN = "df_rec_peso_pnd25_gauss_*_clusters.xlsx"
TRAIN_TARGET_COL = "recpe_gauss"
TARGET_COL = "recpe_og"
COORD_COLS = ["Este", "Norte", "Cota"]
CLUSTER_COL = "cluster"
REQUIRED_COLS = COORD_COLS + [CLUSTER_COL, TARGET_COL, TRAIN_TARGET_COL]
TEST_SIZE = 0.20
SHOW_FIGURES = True

PLOTS_DIR = Path("imagenes") / "train_ml"
METRICS_OUTPUT_PATH = DATA_DIR / "metricas_train_ml_clusters.xlsx"


def load_cluster_files(data_dir: Path, pattern: str) -> list[Path]:
    """Obtiene archivos clusterizados a procesar."""
    candidate_files = sorted(data_dir.glob(pattern))
    valid_files = []
    for file_path in candidate_files:
        try:
            sample_df = pd.read_excel(file_path, nrows=1)
            validate_columns(sample_df, REQUIRED_COLS)
            valid_files.append(file_path)
        except Exception:
            continue

    if not valid_files:
        raise FileNotFoundError(f"No se encontraron archivos con patron: {pattern}")
    return valid_files


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Valida que el DataFrame tenga todas las columnas requeridas."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye features con coordenadas y dummies de cluster."""
    cluster_dummies = pd.get_dummies(df[CLUSTER_COL].astype("category"), prefix="cluster")
    return pd.concat([df[COORD_COLS].copy(), cluster_dummies], axis=1)


def build_inverse_transformer(original_series: pd.Series) -> QuantileTransformer:
    """Ajusta un QuantileTransformer sobre recpe_og para poder invertir predicciones gaussianas.

    Args:
        original_series: Serie con valores originales de recpe_og.

    Returns:
        Transformer ajustado listo para inverse_transform.
    """
    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(original_series)),
        output_distribution="normal",
    )
    transformer.fit(original_series.to_numpy().reshape(-1, 1))
    return transformer


def inverse_transform_predictions(
    transformer: QuantileTransformer,
    pred_gauss: np.ndarray,
) -> np.ndarray:
    """Aplica la transformacion inversa para obtener escala original.

    Args:
        transformer: Transformer ajustado sobre recpe_og.
        pred_gauss: Predicciones en escala gaussiana.

    Returns:
        Array con valores en escala original.
    """
    return transformer.inverse_transform(
        pred_gauss.reshape(-1, 1)
    ).ravel()


def train_knn_with_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[KNeighborsRegressor, dict]:
    """Entrena KNN con busqueda rapida de hiperparametros."""
    param_grid = {
        "n_neighbors": [5, 9, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }
    base_model = KNeighborsRegressor()
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


def train_xgb_with_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[XGBRegressor, dict]:
    """Entrena XGBoost con busqueda rapida de hiperparametros."""
    param_grid = {
        "n_estimators": [150, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    base_model = XGBRegressor(
        objective="reg:squarederror",
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
    )
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcula metricas R2, RMSE y MAPE."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse,
        "mape_pct": mape,
    }


def maybe_show_figure(fig, show_figures: bool) -> None:
    """Muestra la figura o la cierra."""
    if show_figures:
        plt.show()
    else:
        plt.close(fig)


def train_and_evaluate_file(cluster_file: Path) -> list[dict]:
    """Entrena modelos para un archivo de clusters y devuelve metricas.

    Entrena sobre recpe_gauss y grafica en escala original (recpe_og).
    """
    df = pd.read_excel(cluster_file)
    validate_columns(df, REQUIRED_COLS)

    # Ajustar transformer inverso con todos los datos originales disponibles.
    try:
        org_df = pd.read_excel(ORG_DATA_PATH)
        transformer = build_inverse_transformer(org_df[TARGET_COL].dropna())
    except FileNotFoundError:
        transformer = build_inverse_transformer(df[TARGET_COL].dropna())

    x_data = build_features(df)
    y_gauss = df[TRAIN_TARGET_COL].copy()
    y_og = df[TARGET_COL].copy()

    x_train, x_test, y_train, y_test_gauss = train_test_split(
        x_data,
        y_gauss,
        test_size=TEST_SIZE,
    )
    y_test_og = y_og.loc[y_test_gauss.index]

    file_stem = cluster_file.stem
    file_plot_dir = PLOTS_DIR / file_stem
    file_plot_dir.mkdir(parents=True, exist_ok=True)

    # ── KNN ──────────────────────────────────────────────────────────────────
    knn_model, knn_params = train_knn_with_search(x_train, y_train)
    knn_pred_gauss = knn_model.predict(x_test)
    knn_pred_og = inverse_transform_predictions(transformer, knn_pred_gauss)
    knn_metrics = compute_metrics(y_test_og, pd.Series(knn_pred_og, index=y_test_gauss.index))

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_model, xgb_params = train_xgb_with_search(x_train, y_train)
    xgb_pred_gauss = xgb_model.predict(x_test)
    xgb_pred_og = inverse_transform_predictions(transformer, xgb_pred_gauss)
    xgb_metrics = compute_metrics(y_test_og, pd.Series(xgb_pred_og, index=y_test_gauss.index))

    # ── Subplot lado a lado ───────────────────────────────────────────────────
    fig = plot_real_vs_pred_subplot(
        y_true_og=y_test_og,
        pred_knn_og=pd.Series(knn_pred_og, index=y_test_gauss.index),
        pred_xgb_og=pd.Series(xgb_pred_og, index=y_test_gauss.index),
        metrics_knn=knn_metrics,
        metrics_xgb=xgb_metrics,
    )
    fig.savefig(file_plot_dir / "real_vs_pred.png", dpi=300)
    maybe_show_figure(fig, SHOW_FIGURES)

    return [
        {
            "archivo_cluster": file_stem,
            "modelo": "knn",
            "best_params": str(knn_params),
            **knn_metrics,
        },
        {
            "archivo_cluster": file_stem,
            "modelo": "xgboost",
            "best_params": str(xgb_params),
            **xgb_metrics,
        },
    ]


def main() -> None:
    """Ejecuta entrenamiento para todos los archivos de clusters."""
    setup_figuras_tesina()
    cluster_files = load_cluster_files(DATA_DIR, CLUSTER_FILE_PATTERN)

    all_metrics_rows = []
    for cluster_file in cluster_files:
        print(f"Procesando: {cluster_file.name}")
        file_metrics = train_and_evaluate_file(cluster_file)
        all_metrics_rows.extend(file_metrics)

    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df = metrics_df.sort_values(
        by=["r2", "rmse", "mape_pct"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    metrics_df_export = metrics_df.copy()
    metrics_df_export.to_excel(METRICS_OUTPUT_PATH, index=False)

    metrics_df_print = (
        metrics_df[["archivo_cluster", "modelo", "r2", "rmse", "mape_pct"]]
        .copy()
        .rename(columns={
            "archivo_cluster": "Archivo",
            "modelo": "Modelo",
            "r2": "R²",
            "rmse": "RMSE",
            "mape_pct": "MAPE (%)",
        })
    )
    metrics_df_print["R²"] = metrics_df_print["R²"].round(4)
    metrics_df_print["RMSE"] = metrics_df_print["RMSE"].round(4)
    metrics_df_print["MAPE (%)"] = metrics_df_print["MAPE (%)"].round(2)

    print("\nResultados de metricas (ordenados por mejor R²):")
    print(metrics_df_print.to_string(index=False))
    print(f"Archivo de metricas: {METRICS_OUTPUT_PATH.resolve()}")
    print(f"Graficos guardados en: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()

# %%
