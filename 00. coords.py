#%%
"""Extraccion de coordenadas de validacion y comparacion 1-a-1 con datos originales."""

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

try:
    from IPython.display import display as ipy_display
except ImportError:
    ipy_display = None

import config_figuras_tesina as cfg_fig

cfg_fig = importlib.reload(cfg_fig)
setup_figuras_tesina = cfg_fig.setup_figuras_tesina
figsize_cm = cfg_fig.figsize_cm

SHOW_FIGURES = True

INPUT_PATH = Path("data") / "validacion" / "modelo_validos.csv"
VAL_CSV_PATH = Path("data") / "processed" / "coords_validacion_estimacion.csv"
ORG_XLSX_PATH = Path("data") / "processed" / "df_rec_peso_pnd25.xlsx"
COMP_OUTPUT_PATH = Path("data") / "processed" / "comparacion_original_vs_validacion.csv"
IMAGENES_DIR = Path("imagenes") / "validacion"


# ─── Parte 1: Extraccion de coordenadas ──────────────────────────────────────

def load_csv(csv_path: Path, **kwargs) -> pd.DataFrame:
    """Carga un CSV con manejo de errores."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {csv_path.resolve()}")
    try:
        return pd.read_csv(csv_path, **kwargs)
    except Exception as exc:
        raise ValueError(f"No se pudo leer el CSV: {csv_path.resolve()}") from exc


def extract_coords_for_estimacion(source_df: pd.DataFrame) -> pd.DataFrame:
    """Extrae columnas clave para estimacion y comparacion.

    Args:
        source_df: DataFrame de entrada con columnas originales.

    Returns:
        DataFrame con xcentre, ycentre, zcentre, recpe y ue.

    Raises:
        ValueError: Si faltan columnas obligatorias.
    """
    required_columns = ["xcentre", "ycentre", "zcentre", "recpe", "ue"]
    missing_columns = [col for col in required_columns if col not in source_df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
    return source_df[required_columns].copy()


def show_df(df: pd.DataFrame) -> None:
    """Muestra un DataFrame en Jupyter o como texto plano."""
    if ipy_display is not None:
        ipy_display(df)
    else:
        print(df.to_string())


# ─── Parte 2: Busqueda del vecino mas cercano y comparacion ──────────────────

def find_nearest_pairs(
    org_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Para cada sondaje original, encuentra el bloque de validacion mas cercano en 3D.

    Utiliza un KD-Tree sobre las coordenadas del modelo de bloques.
    Guarda tanto las coordenadas del sondaje como las del bloque matcheado, para que
    los scripts posteriores (ej. 041) puedan predecir directamente en las coordenadas
    del bloque sin tener que recargar el modelo completo.

    Args:
        org_df: DataFrame original con columnas Este, Norte, Cota, recpe_og.
        val_df: DataFrame de validacion con columnas xcentre, ycentre, zcentre, recpe.

    Returns:
        DataFrame con: coordenadas sondaje (Este/Norte/Cota), valor real (recpe_og),
        coordenadas bloque matcheado (Este_val/Norte_val/Cota_val),
        valor kriging (recpe_val) y distancia (distancia_m).
    """
    val_coords = val_df[["xcentre", "ycentre", "zcentre"]].to_numpy()
    org_coords = org_df[["Este", "Norte", "Cota"]].to_numpy()

    tree = cKDTree(val_coords)
    distances, indices = tree.query(org_coords, k=1)

    matched = val_df.iloc[indices]

    result_df = org_df[["Este", "Norte", "Cota", "recpe_og"]].copy()
    result_df["Este_val"]    = matched["xcentre"].to_numpy()
    result_df["Norte_val"]   = matched["ycentre"].to_numpy()
    result_df["Cota_val"]    = matched["zcentre"].to_numpy()
    result_df["recpe_val"]   = matched["recpe"].to_numpy()
    result_df["distancia_m"] = distances.round(2)

    return result_df


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcula R2, RMSE y MAPE."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse,
        "mape_pct": mape,
    }


def plot_comparacion_real_vs_val(
    comp_df: pd.DataFrame,
    metrics: dict[str, float],
) -> plt.Figure:
    """Grafico real vs predicho: recpe_og (original) vs recpe_val (validacion).

    Args:
        comp_df: DataFrame con columnas recpe_og y recpe_val.
        metrics: Diccionario con r2, rmse y mape_pct.

    Returns:
        Figura de matplotlib.
    """
    y_true = comp_df["recpe_og"]
    y_pred = comp_df["recpe_val"]

    all_values = pd.concat([y_true, y_pred])
    lower_bound = float(all_values.quantile(0.01))
    upper_bound = float(all_values.quantile(0.99))
    margin = (upper_bound - lower_bound) * 0.04
    axis_min = lower_bound - margin
    axis_max = upper_bound + margin

    fig, ax = plt.subplots(figsize=figsize_cm(10.0, 10.0))

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.45,
        s=14,
        color="#4C78A8",
        edgecolors="none",
    )
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        "--",
        color="#222222",
        linewidth=1.2,
        label="y = x",
    )
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title("Original vs Validación — Recuperación en peso (%)")
    ax.set_xlabel("Recuperación original (%)")
    ax.set_ylabel("Recuperación validación (%)")
    ax.legend(fontsize=8, loc="upper left")

    stats_text = (
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.3f}\n"
        f"MAPE = {metrics['mape_pct']:.1f}%\n"
        f"n = {len(comp_df)}"
    )
    ax.text(
        0.97, 0.03,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7, "edgecolor": "#CCCCCC"},
    )

    return fig


# ─── Ejecucion ────────────────────────────────────────────────────────────────

setup_figuras_tesina()
IMAGENES_DIR.mkdir(parents=True, exist_ok=True)

# — Parte 1: extraer y guardar CSV de validacion —
raw_df = load_csv(INPUT_PATH)
coords_df = extract_coords_for_estimacion(raw_df)
coords_df.to_csv(VAL_CSV_PATH, index=False)
show_df(coords_df.head())
print(f"CSV validacion generado: {VAL_CSV_PATH.resolve()}")

# — Parte 2: comparacion 1-a-1 con vecino mas cercano —
org_df = pd.read_excel(ORG_XLSX_PATH)
val_df = load_csv(VAL_CSV_PATH)

comp_df = find_nearest_pairs(org_df, val_df)
comp_df.to_csv(COMP_OUTPUT_PATH, index=False)
print(f"\nComparacion guardada: {COMP_OUTPUT_PATH.resolve()}")
print(f"Distancia media al vecino mas cercano: {comp_df['distancia_m'].mean():.1f} m")
print(f"Distancia maxima: {comp_df['distancia_m'].max():.1f} m")

metrics = compute_metrics(comp_df["recpe_og"], comp_df["recpe_val"])
print(f"\nMetricas original vs validacion:")
print(f"  R²   = {metrics['r2']:.4f}")
print(f"  RMSE = {metrics['rmse']:.4f}")
print(f"  MAPE = {metrics['mape_pct']:.2f}%")

fig = plot_comparacion_real_vs_val(comp_df, metrics)
fig.savefig(IMAGENES_DIR / "comparacion_original_vs_validacion.png", dpi=300)
if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)

# %%
