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
COORD_COLS = ["Este", "Norte", "Cota"]
COORD_LABELS = ["Este (m)", "Norte (m)", "Cota (m)"]
CLUSTER_COL = "cluster_con_nscore"
TARGET_COL = "Rec_Peso_PND25_(%)_nscore"
TARGET_LABEL = "Recuperación en peso (%) (normal score)"
ORIGEN_COL = "origen"
TEST_SIZE = 0.2
N_ITER = 50  # iteraciones para obtener distribución de R² y RMSE


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
    plt.show()


def plot_3d_cmap(
    df: pd.DataFrame,
    coord_cols: list[str],
    color_col: str,
    coord_labels: list[str],
    title: str,
    cmap: str = "jet",
) -> None:
    """Scatter 3D coloreado por una variable continua (cmap)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        df[coord_cols[0]], df[coord_cols[1]], df[coord_cols[2]],
        c=df[color_col], cmap=cmap, s=15, alpha=0.7,
    )
    ax.set_xlabel(coord_labels[0])
    ax.set_ylabel(coord_labels[1])
    ax.set_zlabel(coord_labels[2])
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.12, label=color_col)
    plt.tight_layout()
    plt.show()


# --- Ejecución ---
# %%
if __name__ == "__main__":
    setup_report_style()

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

    # Resumen: media ± desv estándar
    print("Distribución de métricas (media ± std):")
    print(f"  KNN:     R² = {np.mean(r2_knn):.4f} ± {np.std(r2_knn):.4f}  |  RMSE = {np.mean(rmse_knn):.4f} ± {np.std(rmse_knn):.4f}")
    print(f"  XGBoost: R² = {np.mean(r2_xgb):.4f} ± {np.std(r2_xgb):.4f}  |  RMSE = {np.mean(rmse_xgb):.4f} ± {np.std(rmse_xgb):.4f}")

    # Gráficos: distribución R² y RMSE
    compare_models_distrib(r2_knn, rmse_knn, r2_xgb, rmse_xgb)

    # Real vs predicho (última iteración)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    xgb = XGBRegressor().fit(X_train, y_train)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_real_vs_pred(y_test.values, knn.predict(X_test), "KNN", ax=axes[0])
    plot_real_vs_pred(y_test.values, xgb.predict(X_test), "XGBoost", ax=axes[1])
    plt.tight_layout()
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

    plot_3d_cmap(df_relleno, COORD_COLS, "pred_nscore_knn", COORD_LABELS, "Predicción KNN")
    plot_3d_cmap(df_relleno, COORD_COLS, "pred_nscore_xgb", COORD_LABELS, "Predicción XGBoost")


    plt.hist(df_relleno["pred_nscore_knn"], bins='auto', alpha=0.7, color="#2563eb", edgecolor="white", label="KNN")
    plt.hist(df_relleno["pred_nscore_xgb"], bins='auto', alpha=0.7, color="#059669", edgecolor="white", label="XGBoost")
    plt.hist(df_orig["Rec_Peso_PND25_(%)_nscore"], bins='auto', alpha=0.7, color="#7c3aed", edgecolor="white", label="Original")
    plt.xlabel("Predicción")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de predicciones")
    plt.legend()
    plt.grid(True, alpha=0.35, axis="y")
    plt.show()


    #%%
    # Calcula la media, mínimo y máximo por cluster de las predicciones de KNN y XGBoost
    agg_pred_por_cluster = df_relleno.groupby("cluster_con_nscore")[["pred_nscore_knn", "pred_nscore_xgb"]].agg(["mean", "min", "max"])

    # Calcula la media, mínimo y máximo reales por cluster de los originales
    agg_real_por_cluster = df_orig.groupby("cluster_con_nscore")["Rec_Peso_PND25_(%)_nscore"].agg(["mean", "min", "max"])

    # --- Gráfico de la media por cluster (igual que antes) ---
    plt.figure(figsize=(8, 5))
    plt.plot(agg_pred_por_cluster.index, agg_pred_por_cluster["pred_nscore_knn"]["mean"], marker='o', label="KNN (relleno)")
    plt.plot(agg_pred_por_cluster.index, agg_pred_por_cluster["pred_nscore_xgb"]["mean"], marker='o', label="XGBoost (relleno)")
    plt.plot(agg_real_por_cluster.index, agg_real_por_cluster["mean"], marker='o', label="Original", linestyle='--', color="#7c3aed")
    plt.xlabel("Cluster")
    plt.ylabel("Predicción media (normal score)")
    plt.title("Media de predicción por cluster")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()

    # --- Gráfico del MINIMO por cluster ---
    plt.figure(figsize=(8, 5))
    plt.plot(agg_pred_por_cluster.index, agg_pred_por_cluster["pred_nscore_knn"]["min"], marker='o', label="KNN (relleno)")
    plt.plot(agg_pred_por_cluster.index, agg_pred_por_cluster["pred_nscore_xgb"]["min"], marker='o', label="XGBoost (relleno)")
    plt.plot(agg_real_por_cluster.index, agg_real_por_cluster["min"], marker='o', label="Original", linestyle='--', color="#7c3aed")
    plt.xlabel("Cluster")
    plt.ylabel("Predicción mínima (normal score)")
    plt.title("Mínimo de predicción por cluster")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()

    # --- Gráfico del MAXIMO por cluster ---
    plt.figure(figsize=(8, 5))
    plt.plot(agg_pred_por_cluster.index, agg_pred_por_cluster["pred_nscore_knn"]["max"], marker='o', label="KNN (relleno)")
    plt.plot(agg_pred_por_cluster.index, agg_pred_por_cluster["pred_nscore_xgb"]["max"], marker='o', label="XGBoost (relleno)")
    plt.plot(agg_real_por_cluster.index, agg_real_por_cluster["max"], marker='o', label="Original", linestyle='--', color="#7c3aed")
    plt.xlabel("Cluster")
    plt.ylabel("Predicción máxima (normal score)")
    plt.title("Máximo de predicción por cluster")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


# %%
