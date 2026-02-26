# %%
"""
Predicción de Rec_Peso_PND25_(%)_nscore con KNN y XGBoost.
Features: Este, Norte, Cota (estandarizadas) + get_dummies(cluster_con_nscore).
Misma línea de gráficos que Analisis_EDA / Analisis_cluster.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --- Configuración ---
INPUT_PATH = Path("data/processed/cluster/clusters_df_con_nscore.csv")
INPUT_COMBINED_PATH = Path("data/processed/cluster/puntos_originales_y_relleno.csv")
OUTPUT_DF_RELLENO_PATH = Path("data/processed/df_relleno.csv")
OUTPUT_METRICAS_ML_PATH = Path("data/processed/metricas_ml.csv")
IMAGENES_DIR = Path("imagenes")
COORD_COLS = ["Este", "Norte", "Cota"]
COORD_LABELS = ["Este (m)", "Norte (m)", "Cota (m)"]
CLUSTER_COL = "cluster_con_nscore"
TARGET_COL = "Rec_Peso_PND25_(%)_nscore"
TARGET_LABEL = "Recuperación en peso (%) (normal score)"
ORIGEN_COL = "origen"
TEST_SIZE = 0.2
N_ITER = 20  # iteraciones para obtener distribución de R² y RMSE (histograma)
RANDOM_STATE = 42  # seed para entrenamiento único (tabla y modelo final)


def setup_report_style() -> None:
    """Estilo de figuras (misma línea que Analisis_EDA / Analisis_cluster)."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def build_X(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    dummy_cols_order: list[str] | None = None,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Coords estandarizadas + get_dummies(cluster). Si scaler/dummy_cols_order se pasan, transforma (mismo orden)."""
    coords = df[COORD_COLS].values
    if scaler is None:
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
    else:
        coords_scaled = scaler.transform(coords)
    dummies = pd.get_dummies(df[CLUSTER_COL], prefix="cluster", dtype=float)
    if dummy_cols_order is not None:
        for c in dummy_cols_order:
            if c not in dummies.columns:
                dummies[c] = 0.0
        dummies = dummies[dummy_cols_order]
    dummy_cols = list(dummies.columns)
    X = np.hstack([coords_scaled, dummies.values])
    return X, scaler, dummy_cols


def plot_real_vs_pred(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Gráfico real vs predicho (misma línea que otros scripts)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_real, y_pred, alpha=0.6, s=20, color="#2563eb", edgecolors="white", linewidths=0.3)
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="y = x")
    ax.set_xlabel(f"Real ({TARGET_LABEL})")
    ax.set_ylabel(f"Predicho ({TARGET_LABEL})")
    ax.set_title(f"{model_name} — Real vs predicho")
    ax.legend()
    ax.grid(True, alpha=0.35)
    return ax


def compare_models_distrib(
    r2_knn: list[float],
    rmse_knn: list[float],
    r2_xgb: list[float],
    rmse_xgb: list[float],
    save_path: Path | None = None,
) -> None:
    """Distribución de R² y RMSE por modelo (histogramas)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # R²
    axes[0].hist(r2_knn, bins=15, alpha=0.7, color="#2563eb", edgecolor="white", label="KNN")
    axes[0].hist(r2_xgb, bins=15, alpha=0.7, color="#059669", edgecolor="white", label="XGBoost")
    axes[0].set_xlabel("R²")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución de R²")
    axes[0].legend()
    axes[0].grid(True, alpha=0.35, axis="y")
    # RMSE
    axes[1].hist(rmse_knn, bins=15, alpha=0.7, color="#2563eb", edgecolor="white", label="KNN")
    axes[1].hist(rmse_xgb, bins=15, alpha=0.7, color="#059669", edgecolor="white", label="XGBoost")
    axes[1].set_xlabel("RMSE")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.35, axis="y")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


# --- Ejecución ---
# %%
if __name__ == "__main__":
    setup_report_style()
    IMAGENES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    X, scaler, dummy_cols = build_X(df)
    y = df[TARGET_COL]

    r2_knn, rmse_knn = [], []
    r2_xgb, rmse_xgb = [], []

    for _ in range(N_ITER):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        r2_knn.append(r2_score(y_test, y_pred_knn))
        rmse_knn.append(np.sqrt(mean_squared_error(y_test, y_pred_knn)))

        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        r2_xgb.append(r2_score(y_test, y_pred_xgb))
        rmse_xgb.append(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

    # Resumen distribución (solo para el histograma)
    print("Distribución de métricas (media ± std) — múltiples particiones:")
    print(f"  KNN:     R² = {np.mean(r2_knn):.4f} ± {np.std(r2_knn):.4f}  |  RMSE = {np.mean(rmse_knn):.4f} ± {np.std(rmse_knn):.4f}")
    print(f"  XGBoost: R² = {np.mean(r2_xgb):.4f} ± {np.std(r2_xgb):.4f}  |  RMSE = {np.mean(rmse_xgb):.4f} ± {np.std(rmse_xgb):.4f}")

    # Gráficos: distribución R² y RMSE (iteraciones)
    compare_models_distrib(
        r2_knn, rmse_knn, r2_xgb, rmse_xgb,
        save_path=IMAGENES_DIR / "train_ml_distribucion_R2_RMSE.png",
    )

    # --- Un único modelo con seed (reproducible): métricas para la tabla y modelo final ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    n_test = len(y_test)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    r2_knn_final = r2_score(y_test, y_pred_knn)
    rmse_knn_final = np.sqrt(mean_squared_error(y_test, y_pred_knn))

    xgb = XGBRegressor(random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    r2_xgb_final = r2_score(y_test, y_pred_xgb)
    rmse_xgb_final = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

    print("\nModelo único (seed=%d) — métricas para tabla:" % RANDOM_STATE)
    print(f"  N (test) = {n_test}")
    print(f"  KNN:     R² = {r2_knn_final:.4f}  |  RMSE = {rmse_knn_final:.4f}")
    print(f"  XGBoost: R² = {r2_xgb_final:.4f}  |  RMSE = {rmse_xgb_final:.4f}")

    # Guardar métricas en formato tabla (sin std: un solo modelo)
    df_metricas = pd.DataFrame([
        {"modelo": "KNN", "n": n_test, "R2": r2_knn_final, "RMSE": rmse_knn_final},
        {"modelo": "XGBoost", "n": n_test, "R2": r2_xgb_final, "RMSE": rmse_xgb_final},
    ])
    OUTPUT_METRICAS_ML_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_metricas.to_csv(OUTPUT_METRICAS_ML_PATH, index=False)
    print(f"Métricas guardadas en: {OUTPUT_METRICAS_ML_PATH}")

    # Real vs predicho (modelo único con seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_real_vs_pred(y_test.values, y_pred_knn, "KNN", ax=axes[0])
    plot_real_vs_pred(y_test.values, y_pred_xgb, "XGBoost", ax=axes[1])
    plt.tight_layout()
    plt.savefig(IMAGENES_DIR / "train_ml_real_vs_predicho_KNN_XGBoost.png")
    plt.show()

    # --- Predicción sobre puntos de relleno (modelo ya entrenado arriba) ---
    # %%
    df_full = pd.read_csv(INPUT_COMBINED_PATH)
    df_relleno = df_full[df_full[ORIGEN_COL] == "relleno"].copy()
    
    df_orig = df_full[df_full[ORIGEN_COL] == "original"].copy()

    X_relleno, _, _ = build_X(df_relleno, scaler, dummy_cols)

    y_pred_knn = knn.predict(X_relleno)
    y_pred_xgb = xgb.predict(X_relleno)

    df_relleno["pred_nscore_knn"] = y_pred_knn
    df_relleno["pred_nscore_xgb"] = y_pred_xgb

    OUTPUT_DF_RELLENO_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_relleno.to_csv(OUTPUT_DF_RELLENO_PATH, index=False)
    print(f"DataFrame con predicciones guardado en: {OUTPUT_DF_RELLENO_PATH}")


# %%
