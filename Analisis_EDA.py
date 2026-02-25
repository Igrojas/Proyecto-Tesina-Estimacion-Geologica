# %%
"""
EDA para datos geológicos: carga, calidad, transformación normal score y gráficos de deriva/espaciales.
Uso: definir COORDS y TARGET según tu CSV; el resto es reutilizable.
"""

import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

# --- Configuración (ajustar a tu archivo) ---
DATA_PATH = "data/raw/BD_RECPESO.csv"
COORDS = ["Este", "Norte", "Cota"]
TARGET = "Rec_Peso_PND25_(%)"
# Nombre de la variable para informes (títulos y ejes de figuras)
TARGET_LABEL = "Recuperación en peso (%)"
# Etiquetas de ejes para coordenadas (opcional; si None se usan los nombres de columna)
COORD_LABELS = ["Este (m)", "Norte (m)", "Cota (m)"]
OUTPUT_PATH = "data/processed/df_rec_peso_pnd25.xlsx"
# Carpeta donde se guardan las figuras (se crea al ejecutar)
IMAGENES_DIR = Path("imagenes")
# Valor sentinela: filas con este valor se eliminan (trazabilidad de limpieza)
SENTINEL_VALUE = -99


def setup_report_style() -> None:
    """
    Configura el estilo de matplotlib para figuras de informe/tesina.
    Llamar al inicio del script o antes de generar figuras.
    """
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
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_and_select(
    path: str,
    coord_cols: List[str],
    target_col: str,
    decimal: str = ".",
    sep: str = ";",
) -> pd.DataFrame:
    """Carga el CSV y devuelve un DataFrame solo con coordenadas y variable objetivo."""
    df = pd.read_csv(path, decimal=decimal, sep=sep)
    cols = coord_cols + [target_col]
    return df[cols].copy()


def drop_sentinel_values(
    df: pd.DataFrame,
    columns: List[str],
    sentinel: float = -99.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Elimina filas donde alguna de las columnas indicadas toma el valor sentinela.
    Trazabilidad: imprime filas antes/después y cantidad eliminada.
    """
    n_before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in columns:
        mask &= df[col] != sentinel
    df_clean = df.loc[mask].copy()
    n_after = len(df_clean)
    n_dropped = n_before - n_after
    if verbose:
        print(f"[Trazabilidad] Eliminación de valor sentinela {sentinel} en {columns}")
        print(f"  Filas antes: {n_before}  |  Filas después: {n_after}  |  Eliminadas: {n_dropped}")
    return df_clean


def summary_quality(df: pd.DataFrame) -> None:
    """Muestra info, nulos y describe()."""
    df.info()
    print("Datos faltantes (null/na):")
    display(df.isnull().sum())
    print("Estadística descriptiva:")
    display(df.describe().round(2))


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 20,
    kde: bool = True,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Histograma (y KDE opcional) de una columna. Formato para informe."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[column], bins=bins, kde=kde, ax=ax, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel or column)
    if title:
        ax.set_title(title)
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def add_normal_score(
    df: pd.DataFrame,
    column: str,
    n_quantiles: int = 1000,
    random_state: int = 42,
    suffix: str = "_nscore",
) -> str:
    """
    Añade columna con transformación normal score (QuantileTransformer).
    Devuelve el nombre de la nueva columna.
    """
    new_col = f"{column}{suffix}"
    transformer = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        random_state=random_state,
    )
    data = df[column].values.reshape(-1, 1)
    df[new_col] = transformer.fit_transform(data)
    return new_col


def print_normal_score_stats(
    df: pd.DataFrame,
    original_col: str,
    nscore_col: str,
) -> None:
    """Imprime medias, std, min, max de la columna original y de la normal score."""
    for label, col in [("ORIGINALES", original_col), ("TRANSFORMADAS (Normal Score)", nscore_col)]:
        s = df[col]
        print(f"Estadísticas {label}:")
        print(f"  Media: {s.mean():.4f}  Std: {s.std():.4f}  Min: {s.min():.4f}  Max: {s.max():.4f}\n")


def plot_normal_score_diagnostic(
    df: pd.DataFrame,
    original_col: str,
    nscore_col: str,
    bins: int = 50,
    value_label: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """2x2: histogramas original y nscore, y Q-Q plots. Formato para informe."""
    label = value_label or original_col
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(df[original_col], bins=bins, edgecolor="white", linewidth=0.5, alpha=0.85, color="steelblue")
    axes[0, 0].set_xlabel(label)
    axes[0, 0].set_ylabel("Frecuencia")
    axes[0, 0].set_title("Distribución original")

    axes[0, 1].hist(df[nscore_col], bins=bins, edgecolor="white", linewidth=0.5, alpha=0.85, color="coral")
    axes[0, 1].set_xlabel("Normal score")
    axes[0, 1].set_ylabel("Frecuencia")
    axes[0, 1].set_title("Distribución transformada (normal)")

    stats.probplot(df[original_col], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot (original)")
    axes[1, 0].get_lines()[0].set_markerfacecolor("steelblue")
    axes[1, 0].get_lines()[0].set_alpha(0.8)

    stats.probplot(df[nscore_col], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot (normal score)")
    axes[1, 1].get_lines()[0].set_markerfacecolor("coral")
    axes[1, 1].get_lines()[0].set_alpha(0.8)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


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


def plot_drift(
    df: pd.DataFrame,
    coord_cols: List[str],
    value_col: str,
    n_bins: int = 20,
    coord_labels: Optional[List[str]] = None,
    value_label: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Gráficos de deriva (promedio por bins) para cada coordenada. Formato para informe."""
    x_labels = coord_labels if coord_labels is not None else coord_cols
    y_label = value_label or value_col
    n = len(coord_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, coord, x_label in zip(axes, coord_cols, x_labels):
        dr = drift_by_bins(df, coord, value_col, n_bins=n_bins)
        ax.plot(dr["coord_center"], dr["mean"], "o-", linewidth=2, markersize=6, color="steelblue")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"Promedio {y_label}")
        ax.set_title(f"Deriva según {x_label}")
        ax.set_ylim(0, 35)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_3d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    color_label: Optional[str] = None,
    cmap: str = "jet",
    save_path: Optional[Path] = None,
) -> None:
    """Scatter 3D coloreado por variable. Formato para informe."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(df[x_col], df[y_col], df[z_col], c=df[color_col], cmap=cmap, s=25, alpha=0.7)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_zlabel(zlabel or z_col)
    if title:
        ax.set_title(title)
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.12, label=color_label or color_col)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_2d_projections(
    df: pd.DataFrame,
    coord_cols: List[str],
    color_col: str,
    suptitle: Optional[str] = None,
    coord_labels: Optional[List[str]] = None,
    color_label: Optional[str] = None,
    cmap: str = "jet",
    save_path: Optional[Path] = None,
) -> None:
    """Proyecciones 2D (XY, YZ, XZ) coloreadas por variable. Formato para informe."""
    x, y, z = coord_cols
    lab = coord_labels or coord_cols
    pairs = [(x, y), (y, z), (x, z)]
    pair_labels = [(lab[0], lab[1]), (lab[1], lab[2]), (lab[0], lab[2])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (c1, c2), (l1, l2) in zip(axes, pairs, pair_labels):
        sc = ax.scatter(df[c1], df[c2], c=df[color_col], cmap=cmap, s=12, alpha=0.65)
        ax.set_xlabel(l1)
        ax.set_ylabel(l2)
        ax.set_title(f"{l1} vs {l2}")
        plt.colorbar(sc, ax=ax, label=color_label or color_col)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, weight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


# --- Ejecución con configuración por defecto ---
# %%
if __name__ == "__main__":
    setup_report_style()
    IMAGENES_DIR.mkdir(parents=True, exist_ok=True)
    coord_labels = COORD_LABELS or COORDS  # etiquetas para ejes en figuras

    df = load_and_select(DATA_PATH, COORDS, TARGET)
    list_cols = COORDS + [TARGET]
    print("Columnas usadas:", list_cols)
    print("Variable en estudio (etiqueta):", TARGET_LABEL)

    # Eliminar valor sentinela -99 (trazabilidad en SENTINEL_VALUE y drop_sentinel_values)
    df = drop_sentinel_values(df, [TARGET], sentinel=SENTINEL_VALUE)

    summary_quality(df)
    display(df)

    # %%
    plot_histogram(
        df, TARGET, bins=20, kde=True,
        xlabel=TARGET_LABEL,
        title=f"Distribución de {TARGET_LABEL}",
        save_path=IMAGENES_DIR / "eda_histograma_distribucion.png",
    )

    # %%
    nscore_col = add_normal_score(df, TARGET)
    print_normal_score_stats(df, TARGET, nscore_col)
    plot_normal_score_diagnostic(
        df, TARGET, nscore_col, value_label=TARGET_LABEL,
        save_path=IMAGENES_DIR / "eda_diagnostico_normal_score.png",
    )

    # %%
    plot_drift(
        df, COORDS, TARGET, n_bins=20,
        coord_labels=coord_labels,
        value_label=TARGET_LABEL,
        save_path=IMAGENES_DIR / "eda_deriva_coordenadas.png",
    )

     # %%
    plot_3d(
        df, COORDS[0], COORDS[1], COORDS[2], TARGET,
        title=f"Distribución espacial — coloreado por {TARGET_LABEL}",
        xlabel=coord_labels[0], ylabel=coord_labels[1], zlabel=coord_labels[2],
        color_label=TARGET_LABEL,
        save_path=IMAGENES_DIR / "eda_3d_espacial_target.png",
    )
    plot_3d(
        df, COORDS[0], COORDS[1], COORDS[2], nscore_col,
        title=f"Distribución espacial — coloreado por normal score ({TARGET_LABEL})",
        xlabel=coord_labels[0], ylabel=coord_labels[1], zlabel=coord_labels[2],
        color_label=f"{TARGET_LABEL} (normal score)",
        save_path=IMAGENES_DIR / "eda_3d_espacial_nscore.png",
    )

    # %%
    plot_2d_projections(
        df, COORDS, TARGET,
        suptitle=f"Proyecciones 2D — {TARGET_LABEL}",
        coord_labels=coord_labels,
        color_label=TARGET_LABEL,
        save_path=IMAGENES_DIR / "eda_proyecciones_2d_target.png",
    )
    plot_2d_projections(
        df, COORDS, nscore_col,
        suptitle=f"Proyecciones 2D — {TARGET_LABEL} (normal score)",
        coord_labels=coord_labels,
        color_label=f"{TARGET_LABEL} (normal score)",
        save_path=IMAGENES_DIR / "eda_proyecciones_2d_nscore.png",
    )

    # %%
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Guardado: {OUTPUT_PATH}")

# %%
