# %%
"""
Visualización y análisis de resultados de predicción (KNN y XGBoost).
Carga el DataFrame df_validacion_predicciones generado por train_ml.py y genera gráficos.
Des-transforma las predicciones (Normal Score -> %) para comparar con 'recpe'.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

# --- Configuración ---
INPUT_DF_VAL_PATH = Path("data/processed/df_validacion_predicciones.csv")
INPUT_TRAIN_DATA_PATH = Path("data/processed/cluster/clusters_df_con_nscore.csv")
INPUT_COMBINED_PATH = Path("data/processed/cluster/puntos_originales_y_validacion.csv")
OUTPUT_TABLA_TESINA_PATH = Path("data/processed/tabla_resultados_ml_tesina.xlsx")
OUTPUT_TABLA_MODELOS_PATH = Path("data/processed/tabla_resultados_modelos_tesina.xlsx")
IMAGENES_DIR = Path("imagenes")
COORD_COLS = ["Este", "Norte", "Cota"]
COORD_LABELS = ["Este (m)", "Norte (m)", "Cota (m)"]
CLUSTER_COL = "cluster_con_nscore"
ORIGEN_COL = "origen"
VALUE_COL_ORIG_REAL = "Rec_Peso_PND25_(%)"
VALUE_COL_ORIG_NSCORE = "Rec_Peso_PND25_(%)_nscore"
VALUE_COL_VAL_REAL = "recpe"  # Columna real en datos de validación
VALUE_LABEL = "Recuperación en peso (%)"


def setup_report_style() -> None:
    """Estilo de figuras (proporcional y limpio)."""
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


def set_proportional_aspect(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Ajusta el aspect ratio del gráfico 3D según el rango real de las coordenadas."""
    x, y, z = df[COORD_COLS[0]], df[COORD_COLS[1]], df[COORD_COLS[2]]
    range_x = x.max() - x.min()
    range_y = y.max() - y.min()
    range_z = z.max() - z.min()
    max_range = max(range_x, range_y, range_z)
    if max_range > 0:
        ax.set_box_aspect((range_x / max_range, range_y / max_range, range_z / max_range))


def drift_by_bins(
    df: pd.DataFrame,
    coord_col: str,
    value_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Promedio de value_col por intervalos (bins) de coord_col."""
    out = df[[coord_col, value_col]].copy()
    out["bin"] = pd.cut(out[coord_col], bins=n_bins)
    agg = out.groupby("bin", observed=True)[value_col].agg(["mean", "count"])
    agg["coord_center"] = agg.index.map(lambda x: x.mid)
    return agg.reset_index()


def plot_drift_comparison(
    df_orig: pd.DataFrame,
    df_val: pd.DataFrame,
    coord_cols: list[str],
    n_bins: int = 20,
    coord_labels: list[str] | None = None,
    save_path: Path | None = None,
) -> None:
    """Deriva: Compara Original vs Validación (Real) vs Predicciones en unidades reales."""
    x_labels = coord_labels if coord_labels is not None else coord_cols
    n = len(coord_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, coord, x_label in zip(axes, coord_cols, x_labels):
        dr_orig = drift_by_bins(df_orig, coord, VALUE_COL_ORIG_REAL, n_bins=n_bins)
        dr_val_real = drift_by_bins(df_val, coord, VALUE_COL_VAL_REAL, n_bins=n_bins)
        dr_knn = drift_by_bins(df_val, coord, "pred_real_knn", n_bins=n_bins)
        dr_xgb = drift_by_bins(df_val, coord, "pred_real_xgb", n_bins=n_bins)
        dr_svr = drift_by_bins(df_val, coord, "pred_real_svr", n_bins=n_bins)
        dr_mlp = drift_by_bins(df_val, coord, "pred_real_mlp", n_bins=n_bins)
        
        ax.plot(dr_orig["coord_center"], dr_orig["mean"], "o-", linewidth=1, markersize=4, color="#7c3aed", label="Orig. (Train)")
        ax.plot(dr_val_real["coord_center"], dr_val_real["mean"], "s-", linewidth=1, markersize=4, color="#ef4444", label="Val. (Real)")
        ax.plot(dr_knn["coord_center"], dr_knn["mean"], "--", linewidth=1, color="#2563eb", label="Pred. KNN")
        ax.plot(dr_xgb["coord_center"], dr_xgb["mean"], "--", linewidth=1, color="#059669", label="Pred. XGBoost")
        ax.plot(dr_svr["coord_center"], dr_svr["mean"], "--", linewidth=1, color="#7c3aed", label="Pred. SVR")
        ax.plot(dr_mlp["coord_center"], dr_mlp["mean"], "--", linewidth=1, color="#d97706", label="Pred. MLP")
        
        ax.set_xlabel(x_label)
        ax.set_ylabel("Promedio (%)")
        ax.set_title(f"Deriva Real: {x_label}")
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_3d_proportional(
    df: pd.DataFrame,
    color_col: str,
    title: str,
    label: str,
    cmap: str = "jet",
    save_path: Path | None = None,
) -> None:
    """Scatter 3D proporcional coloreado por una variable."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        df[COORD_COLS[0]], df[COORD_COLS[1]], df[COORD_COLS[2]],
        c=df[color_col], cmap=cmap, s=15, alpha=0.7,
    )
    ax.set_xlabel(COORD_LABELS[0])
    ax.set_ylabel(COORD_LABELS[1])
    ax.set_zlabel(COORD_LABELS[2])
    ax.set_title(title)
    set_proportional_aspect(ax, df)
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=label)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def export_tabla_validacion_tesina(
    df_val: pd.DataFrame,
    output_path: Path,
) -> None:
    """Genera tabla de métricas de validación contra el 'recpe' real en unidades de %."""
    y_real = df_val[VALUE_COL_VAL_REAL]
    
    metrics = []
    for model in ["knn", "xgb", "svr", "mlp"]:
        y_pred = df_val[f"pred_real_{model}"]
        r2 = r2_score(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        bias = np.mean(y_pred - y_real)
        metrics.append({
            "Modelo": model.upper(),
            "N": len(df_val),
            "R2 (unidades reales)": r2,
            "RMSE (unidades reales)": rmse,
            "Sesgo (Bias %)": bias
        })
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_excel(output_path, index=False)
    print(f"Métricas de validación real guardadas en: {output_path}")


# --- Ejecución ---
# %%
if __name__ == "__main__":
    setup_report_style()
    IMAGENES_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DF_VAL_PATH.exists():
        raise FileNotFoundError(f"No se encontró {INPUT_DF_VAL_PATH}. Ejecuta primero 04.train_ml.py.")

    # A) Cargar datos de validación
    df_val = pd.read_csv(INPUT_DF_VAL_PATH)
    df_full = pd.read_csv(INPUT_COMBINED_PATH)
    df_orig = df_full[df_full[ORIGEN_COL] == "original"].copy()

    # B) Re-instanciar y entrenar el transformer para des-transformar (Normal Score -> %)
    # Cargamos el archivo original de entrenamiento que tiene tanto raw como nscore
    if not INPUT_TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró {INPUT_TRAIN_DATA_PATH}. Se necesita para revertir nscore.")
    
    df_train_ref = pd.read_csv(INPUT_TRAIN_DATA_PATH)
    
    target_transformer = QuantileTransformer(
        n_quantiles=1000, 
        output_distribution="normal", 
        random_state=42
    )
    # Ajustamos con los valores originales (%)
    target_transformer.fit(df_train_ref[VALUE_COL_ORIG_REAL].values.reshape(-1, 1))

    # C) Des-transformar predicciones de validación
    for model in ["knn", "xgb", "svr", "mlp"]:
        nscore_preds = df_val[f"pred_nscore_{model}"].values.reshape(-1, 1)
        real_preds = target_transformer.inverse_transform(nscore_preds)
        df_val[f"pred_real_{model}"] = real_preds.flatten()

    print(f"Cargados {len(df_val)} puntos de validación. Predicciones des-transformadas a unidades reales (%).")

    # 1) Métricas de Validación en UNIDADES REALES
    export_tabla_validacion_tesina(df_val, OUTPUT_TABLA_MODELOS_PATH)

    # 2) Gráficos 3D Proporcionales (Unidades Reales %)
    plot_3d_proportional(df_val, VALUE_COL_VAL_REAL, "Validación: Recpe Real (%)", "Rec_Peso (%)", 
                         save_path=IMAGENES_DIR / "resultados_val_real_3d_target.png")
    plot_3d_proportional(df_val, "pred_real_knn", "Validación: Predicción KNN (%)", "Pred. Real (%)", 
                         save_path=IMAGENES_DIR / "resultados_val_real_3d_knn.png")
    plot_3d_proportional(df_val, "pred_real_xgb", "Validación: Predicción XGBoost (%)", "Pred. Real (%)", 
                         save_path=IMAGENES_DIR / "resultados_val_real_3d_xgb.png")
    plot_3d_proportional(df_val, "pred_real_svr", "Validación: Predicción SVR (%)", "Pred. Real (%)", 
                         save_path=IMAGENES_DIR / "resultados_val_real_3d_svr.png")
    plot_3d_proportional(df_val, "pred_real_mlp", "Validación: Predicción MLP (%)", "Pred. Real (%)", 
                         save_path=IMAGENES_DIR / "resultados_val_real_3d_mlp.png")

    # 3) Comparación de Deriva (Drift Analysis) en UNIDADES REALES
    plot_drift_comparison(df_orig, df_val, COORD_COLS, coord_labels=COORD_LABELS,
                          save_path=IMAGENES_DIR / "resultados_val_real_drift_comparativo.png")

    # 4) Histograma Comparativo en UNIDADES REALES
    plt.figure(figsize=(10, 6))
    plt.hist(df_val[VALUE_COL_VAL_REAL], bins=30, alpha=0.5, label="Val. Real (recpe)", color="#ef4444", density=True)
    plt.hist(df_val["pred_real_knn"], bins=30, alpha=0.5, label="Pred. KNN (%)", color="#2563eb", density=True)
    plt.hist(df_val["pred_real_xgb"], bins=30, alpha=0.5, label="Pred. XGBoost (%)", color="#059669", density=True)
    plt.hist(df_val["pred_real_svr"], bins=30, alpha=0.5, label="Pred. SVR (%)", color="#7c3aed", density=True)
    plt.hist(df_val["pred_real_mlp"], bins=30, alpha=0.5, label="Pred. MLP (%)", color="#d97706", density=True)
    plt.title("Distribución de Valores Reales: Real vs Predicho (%)")
    plt.xlabel("Recuperación (%)")
    plt.ylabel("Densidad")
    plt.legend()
    plt.savefig(IMAGENES_DIR / "resultados_val_real_histograma.png")
    plt.show()

    # 5) Scatter Real vs Predicho en UNIDADES REALES
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, model, color in zip(axes, ["knn", "xgb", "svr", "mlp"], ["#2563eb", "#059669", "#7c3aed", "#d97706"]):
        ax.scatter(df_val[VALUE_COL_VAL_REAL], df_val[f"pred_real_{model}"], alpha=0.4, s=10, color=color)
        min_v = min(df_val[VALUE_COL_VAL_REAL].min(), df_val[f"pred_real_{model}"].min())
        max_v = max(df_val[VALUE_COL_VAL_REAL].max(), df_val[f"pred_real_{model}"].max())
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label="1:1")
        ax.set_title(f"Validación Real: Real vs {model.upper()} (%)")
        ax.set_xlabel("Real (%)")
        ax.set_ylabel("Predicho (%)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGENES_DIR / "resultados_val_real_scatter_real_vs_pred.png")
    plt.show()

# %%

# %%
