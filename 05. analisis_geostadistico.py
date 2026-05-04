#%%
"""Analisis geostadistico comparativo: ML vs Kriging vs Datos Originales.

Metodologia:
- Histogramas normalizados (densidad): permiten comparar la forma de la distribucion
  entre metodos independientemente del tamano muestral.
- Grafica de deriva (drift): media espacial de cada estimacion en bins a lo largo de
  Este, Norte y Cota. Revela tendencias sistematicas y sesgos espaciales de cada metodo.
- "Datos Originales" = distribucion de recpe_og en todos los sondajes de entrenamiento.
- ML y Kriging = estimaciones en las coordenadas del bloque mas cercano a cada sondaje.
"""

import importlib
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import config_figuras_tesina as cfg_fig

cfg_fig = importlib.reload(cfg_fig)
setup_figuras_tesina = cfg_fig.setup_figuras_tesina


# ─── Rutas ────────────────────────────────────────────────────────────────────

DATA_DIR         = Path("data")
PREDICTIONS_PATH = DATA_DIR / "results" / "predicciones_ml_vs_kriging.csv"
# Para histogramas se puede usar el archivo gauss (tiene recpe_og igual + recpe_gauss)
TRAIN_PATH       = DATA_DIR / "processed" / "df_rec_peso_pnd25_gauss.xlsx"
# Para la deriva se usa el mismo dataset que 02. Eda_analisis.py para que la curva
# Original sea identica a la generada en el EDA.
ORIG_PATH        = DATA_DIR / "processed" / "df_rec_peso_pnd25.xlsx"
PLOTS_DIR        = Path("imagenes") / "analisis_geostadistico"

SHOW_FIGURES     = True
BIN_SIZE_DERIVA  = 100.0  # metros por bin — igual que en 02. Eda_analisis.py

COORD_COLS  = ["Este_val", "Norte_val", "Cota_val"]
TARGET_COL  = "recpe_og"
KRIGING_COL = "recpe_val"

COORD_LABELS: dict[str, str] = {
    "Este_val":  "Este (m)",
    "Norte_val": "Norte (m)",
    "Cota_val":  "Cota (m)",
}

# ─── Paleta de colores ────────────────────────────────────────────────────────
# Original usa morado sólido; las estimaciones usan colores distintos con línea punteada.
# Colores coherentes con config_figuras_tesina.py (Vega/Altair scheme).
COLOR_ORIGINAL = "#9467BD"          # morado Tableau

_COLORS_ESTIMACIONES: list[str] = [
    "#4C78A8",  # azul   → Kriging (primero)
    "#F58518",  # naranja
    "#54A24B",  # verde
    "#E45756",  # rojo
    "#72B7B2",  # teal
    "#B279A2",  # rosa
]

# ─── Graficas ─────────────────────────────────────────────────────────────────

def plot_histogramas_normalizados(
    series_dict: dict[str, pd.Series],
    n_bins: int = 30,
) -> plt.Figure:
    """Histogramas normalizados (densidad) y curva KDE superpuesta para cada metodo.

    Args:
        series_dict: {nombre_metodo: serie de valores}. El orden define la leyenda.
        n_bins: numero de bins del histograma.

    Returns:
        Figure con un unico Axes comparando todos los metodos.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    all_values = np.concatenate([s.dropna().values for s in series_dict.values()])
    x_min, x_max = np.percentile(all_values, 0.5), np.percentile(all_values, 99.5)
    bins = np.linspace(x_min, x_max, n_bins + 1)
    x_kde = np.linspace(x_min, x_max, 400)

    for label, series in series_dict.items():
        vals = series.dropna().values
        ax.hist(
            vals, bins=bins, density=True, alpha=0.18,
            label=f"{label} (n={len(vals):,})"
        )
        kde = gaussian_kde(vals, bw_method="scott")
        ax.plot(x_kde, kde(x_kde), linewidth=1.8, label=f"{label} KDE")

    ax.set_xlabel("recpe (%)")
    ax.set_ylabel("Densidad")
    ax.set_title("Histogramas normalizados: comparacion de distribuciones")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def _media_por_bin(
    coord_vals: np.ndarray,
    values: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula media y desviacion estandar de `values` en cada bin de `coord_vals`.

    Returns:
        Tupla (centros_de_bin, medias, desviaciones).
    """
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    indices  = np.digitize(coord_vals, bin_edges) - 1
    indices  = np.clip(indices, 0, len(bin_mids) - 1)

    medias = np.array([
        values[indices == i].mean() if (indices == i).any() else np.nan
        for i in range(len(bin_mids))
    ])
    stds = np.array([
        values[indices == i].std() if (indices == i).sum() > 1 else np.nan
        for i in range(len(bin_mids))
    ])
    return bin_mids, medias, stds


def plot_deriva(
    pred_df: pd.DataFrame,
    series_pred: dict[str, pd.Series],
    orig_df: pd.DataFrame,
    coord_col: str,
    bin_size: float = BIN_SIZE_DERIVA,
) -> plt.Figure:
    """Grafica de deriva espacial: media por bin a lo largo de una coordenada.

    Incluye los datos originales de sondaje (linea solida morada) y las estimaciones
    de Kriging y ML (lineas punteadas, colores distintos). Los bins son de ancho fijo
    en metros (bin_size=100 por defecto, igual que en 02. Eda_analisis.py) para que
    la curva Original sea identica a la del EDA.

    Args:
        pred_df: DataFrame con columnas Este_val/Norte_val/Cota_val y estimaciones.
        series_pred: {nombre: serie} de estimaciones alineadas con pred_df.
        orig_df: DataFrame original de sondajes (df_rec_peso_pnd25.xlsx).
        coord_col: columna de coordenada en pred_df (p.ej. "Este_val").
        bin_size: ancho de cada bin en metros.

    Returns:
        Figure con media ± std por bin. Original en morado solido; resto punteado.
    """
    orig_coord_col = coord_col.replace("_val", "")

    pred_coords = pred_df[coord_col].dropna().values
    orig_coords = orig_df[orig_coord_col].dropna().values

    coord_min = min(pred_coords.min(), orig_coords.min())
    coord_max = max(pred_coords.max(), orig_coords.max())
    bin_edges = np.arange(coord_min, coord_max + bin_size, bin_size)

    fig, ax = plt.subplots(figsize=(8, 4))

    # — Datos Originales: linea solida, morado —
    orig_mask = orig_df[orig_coord_col].notna()
    orig_c    = orig_df.loc[orig_mask, orig_coord_col].values
    orig_vals = orig_df.loc[orig_mask, TARGET_COL].values
    mids, medias, stds = _media_por_bin(orig_c, orig_vals, bin_edges)
    valid = ~np.isnan(medias)
    ax.plot(
        mids[valid], medias[valid],
        color=COLOR_ORIGINAL, linestyle="-",
        linewidth=1.1, marker="o", markersize=3.5,
        label="Original",
    )
    ax.fill_between(
        mids[valid], medias[valid] - stds[valid], medias[valid] + stds[valid],
        color=COLOR_ORIGINAL, alpha=0.08,
    )

    # — Estimaciones: linea punteada, colores de paleta —
    color_cycle = iter(_COLORS_ESTIMACIONES)
    for label, series in series_pred.items():
        color = next(color_cycle, "#888888")
        mids, medias, stds = _media_por_bin(pred_df[coord_col].values, series.values, bin_edges)
        valid = ~np.isnan(medias)
        ax.plot(
            mids[valid], medias[valid],
            color=color, linestyle="--",
            linewidth=1.0, marker="o", markersize=3.5,
            label=label,
        )
        ax.fill_between(
            mids[valid], medias[valid] - stds[valid], medias[valid] + stds[valid],
            color=color, alpha=0.07,
        )

    ax.set_xlabel(COORD_LABELS.get(coord_col, coord_col))
    ax.set_ylabel("Media Recuperación en Peso (%)")
    ax.set_title(f"Deriva — {COORD_LABELS.get(coord_col, coord_col)} ({int(bin_size)} m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ─── Ejecucion ────────────────────────────────────────────────────────────────

def main() -> None:
    """Carga predicciones de 04. y genera analisis geostadistico comparativo."""
    setup_figuras_tesina()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # — Cargar datos —
    try:
        pred_df  = pd.read_csv(PREDICTIONS_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontro {PREDICTIONS_PATH}. "
            "Ejecutar primero '04. comparacion_ml_vs_kriging.py'."
        )

    train_df = pd.read_excel(TRAIN_PATH)  # gauss → para histogramas
    orig_df  = pd.read_excel(ORIG_PATH)  # sin gauss → para deriva (igual que 02.)

    # Columnas ML = todo lo que no sea coordenadas, target ni kriging
    known_cols = set(COORD_COLS + [TARGET_COL, KRIGING_COL])
    ml_cols    = [c for c in pred_df.columns if c not in known_cols]

    print(f"Modelos ML encontrados en CSV: {ml_cols}")
    print(f"Puntos de prediccion         : {len(pred_df)}")
    print(f"Sondajes originales          : {len(orig_df)}\n")

    # — Construir dict ordenado para graficas —
    # Original primero, luego Kriging, luego ML (orden del CSV)
    series_para_histograma: dict[str, pd.Series] = {
        "Original": train_df[TARGET_COL].dropna().reset_index(drop=True),
        "Kriging":  pred_df[KRIGING_COL],
    }
    for col in ml_cols:
        series_para_histograma[col] = pred_df[col]

    # Para la deriva: Kriging + ML alineados con pred_df.
    # Original se pasa via train_df directamente a plot_deriva.
    series_para_deriva: dict[str, pd.Series] = {
        "Kriging": pred_df[KRIGING_COL],
    }
    for col in ml_cols:
        series_para_deriva[col] = pred_df[col]

    # — Histogramas normalizados —
    fig_hist = plot_histogramas_normalizados(series_para_histograma)
    fig_hist.savefig(PLOTS_DIR / "histogramas_normalizados.png", dpi=300)
    print(f"Histograma guardado: {(PLOTS_DIR / 'histogramas_normalizados.png').resolve()}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig_hist)

    # — Graficas de deriva por coordenada —
    for coord in COORD_COLS:
        fig_drift = plot_deriva(pred_df, series_para_deriva, orig_df, coord_col=coord)
        fname = f"deriva_{coord.lower().replace('_val', '')}.png"
        fig_drift.savefig(PLOTS_DIR / fname, dpi=300)
        print(f"Deriva guardada  : {(PLOTS_DIR / fname).resolve()}")
        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig_drift)

if __name__ == "__main__":
    main()

# %%
