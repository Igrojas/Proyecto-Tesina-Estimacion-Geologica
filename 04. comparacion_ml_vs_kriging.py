#%%
"""Comparacion directa ML vs Kriging: prediccion en coordenadas del grid de bloques.

Metodologia:
- Los modelos ML se entrenan con el 100% de los datos de sondajes (coordenadas originales).
- Para predecir, se usan las coordenadas del bloque mas cercano a cada sondaje (guardadas
  en el CSV por 00. coords.py), garantizando que ML y kriging se evaluan en los mismos puntos.
- Hiperparametros: fijos y elegidos para interpolacion espacial. No se usa CV para tunear
  porque el CV aleatorio sobre datos espacialmente correlacionados es optimista: maximizarlo
  tiende a producir modelos mas ajustados a los sondajes pero con menor capacidad de
  generalizacion a coordenadas nuevas (exactamente lo que necesitamos aqui).
- Metricas: R2, RMSE, MAPE comparados contra el valor real del sondaje (recpe_og).
  La tabla final incluye la diferencia en puntos porcentuales de R² respecto al kriging.
- Features: solo coordenadas (simetrico con kriging).
"""

import importlib
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

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
TRAIN_PATH = DATA_DIR / "df_rec_peso_pnd25_gauss.xlsx"
COMP_PATH  = DATA_DIR / "comparacion_original_vs_validacion.csv"

COORD_COLS     = ["Este", "Norte", "Cota"]
VAL_COORD_COLS = ["Este_val", "Norte_val", "Cota_val"]

TARGET_COL       = "recpe_og"
TRAIN_TARGET_COL = "recpe_gauss"
KRIGING_COL      = "recpe_val"

SHOW_FIGURES = True

RESULTS_DIR         = Path("data") / "results"
PREDICTIONS_PATH    = RESULTS_DIR / "predicciones_ml_vs_kriging.csv"
PLOTS_DIR           = Path("imagenes") / "comparacion_ml_kriging"
METRICS_OUTPUT_PATH = DATA_DIR / "Obs" / "metricas_ml_vs_kriging.xlsx"


# ─── Transformacion gaussiana ─────────────────────────────────────────────────

def build_transformer(series: pd.Series) -> QuantileTransformer:
    """Ajusta QuantileTransformer sobre los valores originales de entrenamiento."""
    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(series)),
        output_distribution="normal",
    )
    transformer.fit(series.to_numpy().reshape(-1, 1))
    return transformer


def to_original(transformer: QuantileTransformer, values: np.ndarray) -> np.ndarray:
    """Invierte la transformacion gaussiana a escala original."""
    return transformer.inverse_transform(values.reshape(-1, 1)).ravel()


# ─── Modelos con parametros fijos ────────────────────────────────────────────

def get_models() -> list[tuple[str, object]]:
    """Retorna lista de (nombre, estimador) con parametros fijos para interpolacion espacial.

    Los parametros se eligen para favorecer la generalizacion a coordenadas nuevas:
    - KNN con ponderacion por distancia (equivalente a IDW, natural para interpolacion).
    - Arboles con regularizacion moderada (evita memorizar sondajes exactos).
    - Ridge con alpha conservador (suaviza la prediccion lineal).
    - SVR con margen amplio (kernel RBF captura relaciones espaciales no lineales).
    """
    models: list[tuple[str, object]] = [
        (
            "KNN",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=10, weights="distance", p=2)),
            ]),
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=4, n_jobs=-1, random_state=42
            ),
        ),
        (
            "GradientBoosting",
            GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.1,
                subsample=0.8, min_samples_leaf=4, random_state=42
            ),
        ),
    ]
    if _HAS_XGB:
        models.append((
            "XGBoost",
            XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=4,
                reg_alpha=0.1, objective="reg:squarederror",
                n_jobs=-1, random_state=42,
            ),
        ))
    return models


# ─── Metricas ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcula R2, RMSE y MAPE."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    return {"r2": r2_score(y_true, y_pred), "rmse": rmse, "mape_pct": mape}


# ─── Ejecucion ────────────────────────────────────────────────────────────────

def main() -> None:
    """Entrena modelos ML con optimizacion bayesiana y compara predicciones vs kriging."""
    setup_figuras_tesina()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # — Datos de entrenamiento —
    train_df      = pd.read_excel(TRAIN_PATH)
    x_train       = train_df[COORD_COLS].copy()
    y_train_gauss = train_df[TRAIN_TARGET_COL].copy()
    transformer   = build_transformer(train_df[TARGET_COL].dropna())

    # — Datos de comparacion (generados por 00. coords.py) —
    comp_df   = pd.read_csv(COMP_PATH)

    missing_val_cols = [c for c in VAL_COORD_COLS if c not in comp_df.columns]
    if missing_val_cols:
        raise ValueError(
            f"Columnas {missing_val_cols} no encontradas en {COMP_PATH}. "
            "Volver a ejecutar '00. coords.py' para regenerar el CSV."
        )

    y_real    = comp_df[TARGET_COL].copy()
    y_kriging = comp_df[KRIGING_COL].copy()
    x_val     = comp_df[VAL_COORD_COLS].rename(
        columns=dict(zip(VAL_COORD_COLS, COORD_COLS))
    ).reset_index(drop=True)

    print(f"Puntos de entrenamiento        : {len(x_train)}")
    print(f"Bloques de prediccion          : {len(x_val)}")
    print(f"Distancia media sondaje→bloque : {comp_df['distancia_m'].mean():.1f} m\n")

    # — Metricas kriging (linea de referencia) —
    kriging_metrics = compute_metrics(y_real, y_kriging)
    kriging_r2      = kriging_metrics["r2"]
    print(
        f"[Kriging]  R²={kriging_r2:.4f}  "
        f"RMSE={kriging_metrics['rmse']:.4f}  "
        f"MAPE={kriging_metrics['mape_pct']:.2f}%"
    )

    all_metrics: list[dict] = [{"modelo": "Kriging", "delta_r2_pct": 0.0, **kriging_metrics}]
    ml_predictions: dict[str, pd.Series] = {}

    # — Entrenar y evaluar cada modelo ML —
    for name, model in get_models():
        print(f"[{name}] Entrenando...", end=" ", flush=True)

        model.fit(x_train, y_train_gauss)

        pred_gauss = model.predict(x_val)
        pred_og    = pd.Series(to_original(transformer, pred_gauss), index=y_real.index)

        metrics = compute_metrics(y_real, pred_og)
        delta   = (metrics["r2"] - kriging_r2) * 100.0  # puntos porcentuales de R²

        print(
            f"R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MAPE={metrics['mape_pct']:.2f}%  "
            f"ΔR²={delta:+.1f}%"
        )

        all_metrics.append({"modelo": name, "delta_r2_pct": delta, **metrics})
        ml_predictions[name] = pred_og

        fig = plot_real_vs_ml_vs_kriging(
            y_true_og=y_real,
            y_ml=pred_og,
            y_kriging=y_kriging,
            model_name=name,
        )
        fig.savefig(PLOTS_DIR / f"{name.lower()}_vs_kriging.png", dpi=300)
        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)

    # — Exportar predicciones para analisis geostadistico (05.) —
    pred_export_df = comp_df[VAL_COORD_COLS + [TARGET_COL, KRIGING_COL]].copy()
    for model_name, preds in ml_predictions.items():
        pred_export_df[model_name] = preds.values
    pred_export_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nPredicciones exportadas: {PREDICTIONS_PATH.resolve()}")

    # — Tabla resumen —
    metrics_df = (
        pd.DataFrame(all_metrics)
        .sort_values("r2", ascending=False)
        .reset_index(drop=True)
    )
    metrics_df.to_excel(METRICS_OUTPUT_PATH, index=False)

    print("\nResumen (ordenado por R²):")
    print(
        metrics_df[["modelo", "r2", "rmse", "mape_pct", "delta_r2_pct"]]
        .rename(columns={
            "modelo":       "Modelo",
            "r2":           "R²",
            "rmse":         "RMSE",
            "mape_pct":     "MAPE (%)",
            "delta_r2_pct": "ΔR² vs Kriging (%)",
        })
        .assign(**{
            "R²":                lambda d: d["R²"].round(4),
            "RMSE":              lambda d: d["RMSE"].round(4),
            "MAPE (%)":          lambda d: d["MAPE (%)"].round(2),
            "ΔR² vs Kriging (%)": lambda d: d["ΔR² vs Kriging (%)"].apply(
                lambda v: "—" if v == 0.0 else f"{v:+.1f}%"
            ),
        })
        .to_string(index=False)
    )
    print(f"\nMetricas: {METRICS_OUTPUT_PATH.resolve()}")
    print(f"Graficos: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()

# %%
