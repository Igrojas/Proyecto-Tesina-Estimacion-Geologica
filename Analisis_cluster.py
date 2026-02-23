# %%
"""
Análisis de clustering (K-means) sobre datos geológicos procesados.
Usa el mismo esquema de configuración y estilo que Analisis_EDA.py.
Entrada: Excel generado por el EDA (coords + target + target_nscore).
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import norm, probplot
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Paleta con buen contraste y aspecto profesional (tonos medios, no oscuros)
CLUSTER_PALETTE = [
    "#2563eb",  # azul
    "#dc2626",  # rojo
    "#059669",  # verde esmeralda
    "#d97706",  # ámbar
    "#7c3aed",  # violeta
    "#0d9488",  # teal
    "#ea580c",  # naranja
    "#4f46e5",  # índigo
]

# --- Configuración (alineada con Analisis_EDA.py) ---
INPUT_PATH = "data/processed/df_rec_peso_pnd25.xlsx"
COORDS = ["Este", "Norte", "Cota"]
TARGET = "Rec_Peso_PND25_(%)"
NSCORE_SUFFIX = "_nscore"
TARGET_LABEL = "Recuperación en peso (%)"
COORD_LABELS = ["Este (m)", "Norte (m)", "Cota (m)"]
N_CLUSTERS = 5
# Rango de k para explorar (método del codo + silueta)
K_RANGE = (2, 11)  # prueba k desde 2 hasta k_max - 1
# Ejecutar análisis: con nscore, sin nscore, o ambos
RUN_CON_NSCORE = True   # clustering con (coords + variable normal score)
RUN_SIN_NSCORE = True   # clustering solo con coordenadas (sin nscore)
OUTPUT_DIR = Path("data/processed/cluster")


def _cluster_cmap_norm(n_clusters: int) -> Tuple[ListedColormap, BoundaryNorm]:
    """Colormap y norma discretos para que la colorbar muestre solo n_clusters."""
    colors = CLUSTER_PALETTE[:n_clusters]
    cmap = ListedColormap(colors)
    boundaries = np.arange(n_clusters + 1) - 0.5
    norm = BoundaryNorm(boundaries, n_clusters)
    return cmap, norm


def setup_report_style() -> None:
    """Estilo de figuras para informe/tesina (misma línea que Analisis_EDA.py)."""
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


def load_processed_data(path: str | Path) -> pd.DataFrame:
    """Carga el DataFrame procesado (salida del EDA)."""
    return pd.read_excel(path).copy()


def _build_X(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Matriz de características escalada a partir de las columnas indicadas."""
    X = df[feature_cols].values
    return StandardScaler().fit_transform(X)


def fit_clusters(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int,
    cluster_col: str = "cluster",
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Ajusta K-means sobre las columnas indicadas (escaladas) y añade columna de clusters.
    feature_cols: ej. [Este, Norte, Cota] o [Este, Norte, Cota, target_nscore].
    """
    X = _build_X(df, feature_cols)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    out = df.copy()
    out[cluster_col] = labels
    return out


def evaluate_k_range(
    df: pd.DataFrame,
    feature_cols: List[str],
    k_min: int = 2,
    k_max: int = 11,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Prueba K-means para k desde k_min hasta k_max - 1.
    Devuelve un DataFrame con k, inercia y coeficiente de silueta.
    """
    X = _build_X(df, feature_cols)
    k_values = range(k_min, k_max)
    rows = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "inercia": inertia, "silueta": sil})
    return pd.DataFrame(rows)


def plot_k_selection(
    results: pd.DataFrame,
    title: str = "Selección del número de clusters",
) -> None:
    """
    Gráfico doble: inercia (método del codo) y silueta vs k.
    Sirve para tantear el k más razonable antes de fijar N_CLUSTERS.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))
    k = results["k"].values
    ax1.plot(k, results["inercia"], "o-", color="#2563eb", linewidth=2, markersize=8, label="Inercia")
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inercia", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_xticks(k)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(k, results["silueta"], "s--", color="#059669", linewidth=2, markersize=8, label="Silueta")
    ax2.set_ylabel("Coef. silueta", color="#059669")
    ax2.tick_params(axis="y", labelcolor="#059669")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="center right")
    if title:
        ax1.set_title(title)
    plt.tight_layout()
    plt.show()


def _probability_scale_ticks() -> Tuple[np.ndarray, List[str]]:
    """Posiciones y etiquetas para eje Y en escala de probabilidad normal (%)."""
    percents = np.array(
        [0.01, 0.1, 0.2, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 99.8, 99.9, 99.99]
    )
    # Evitar 0 y 1 en ppf (dan -inf/inf)
    p = np.clip(percents / 100.0, 1e-6, 1 - 1e-6)
    positions = norm.ppf(p)
    labels = [f"{x:.2f}" if x < 1 or x > 99 else f"{x:.0f}" for x in percents]
    return positions, labels


def plot_cluster_summary(
    df: pd.DataFrame,
    nscore_col: str,
    coord_cols: List[str],
    n_clusters: int,
    value_label: Optional[str] = None,
    coord_labels: Optional[List[str]] = None,
    cluster_col: str = "cluster",
    suptitle_prefix: str = "",
    value_col: Optional[str] = None,
) -> None:
    """
    Gráficos de resumen del clustering.
    value_col: columna a usar en los 3 paneles (boxplot, efecto, prob). Si None, se usa nscore_col.
    En modo "sin nscore" conviene pasar value_col=target para que el eje X del gráfico de
    probabilidad esté en unidades originales (rango correcto).
    """
    # Variable mostrada: si no se indica, se usa nscore (comportamiento anterior)
    col = value_col if value_col is not None else nscore_col
    lab = value_label or col
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    palette = CLUSTER_PALETTE[:n_clusters]
    if suptitle_prefix:
        fig.suptitle(suptitle_prefix, fontsize=12, weight="bold", y=1.02)

    sns.boxplot(
        x=cluster_col, y=col, data=df, ax=axes[0],
        palette=palette, linewidth=1.2,
    )
    axes[0].set_title(f"Distribución de {lab} por cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel(lab)

    # Eje X = media, Eje Y = desviación estándar (un punto por cluster)
    stats_cluster = df.groupby(cluster_col)[col].agg(["mean", "std"]).reset_index()
    for c in range(n_clusters):
        row = stats_cluster[stats_cluster[cluster_col] == c].iloc[0]
        axes[1].scatter(
            row["mean"], row["std"],
            color=palette[c], s=80, edgecolor="black", linewidths=1,
            label=f"Cluster {c}", zorder=5,
        )
    axes[1].set_title("Efecto por cluster (media vs desv. estándar)")
    axes[1].set_xlabel(f"Media ({lab})")
    axes[1].set_ylabel(f"Desviación estándar ({lab})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.35)

    # Gráfico de probabilidad normal: X = col (rango según variable: nscore ~[-3,3] o target en unidades reales)
    y_ticks, y_labels = _probability_scale_ticks()
    for c in range(n_clusters):
        data = np.sort(df.loc[df[cluster_col] == c, col].values)
        n = len(data)
        if n == 0:
            continue
        p = np.clip((np.arange(1, n + 1) - 0.5) / n, 1e-6, 1 - 1e-6)
        y_plot = norm.ppf(p)
        axes[2].scatter(
            data, y_plot,
            color=palette[c], s=12, alpha=0.7, edgecolors="black", linewidths=0.3,
            label=f"Cluster {c}",
        )
    axes[2].set_title("Gráfico de probabilidad normal")
    axes[2].set_xlabel(lab)
    axes[2].set_ylabel("Frecuencia acumulada (%)")
    axes[2].set_yticks(y_ticks)
    axes[2].set_yticklabels(y_labels)
    axes[2].set_ylim(norm.ppf(0.0001), norm.ppf(0.9999))
    axes[2].legend()
    axes[2].grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_clusters_3d(
    df: pd.DataFrame,
    coord_cols: List[str],
    cluster_col: str = "cluster",
    n_clusters: Optional[int] = None,
    coord_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> None:
    """Visualización 3D de los clusters. Colorbar discreta: solo colores de clusters presentes."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    xl = coord_labels or coord_cols
    k = n_clusters or int(df[cluster_col].nunique())
    cmap, norm = _cluster_cmap_norm(k)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        df[coord_cols[0]], df[coord_cols[1]], df[coord_cols[2]],
        c=df[cluster_col], cmap=cmap, norm=norm, s=25, alpha=0.7,
    )
    ax.set_xlabel(xl[0])
    ax.set_ylabel(xl[1])
    ax.set_zlabel(xl[2])
    if title:
        ax.set_title(title)
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.12, label="Cluster", ticks=np.arange(k))
    plt.tight_layout()
    plt.show()


def plot_clusters_2d(
    df: pd.DataFrame,
    coord_cols: List[str],
    cluster_col: str = "cluster",
    n_clusters: Optional[int] = None,
    coord_labels: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
) -> None:
    """Proyecciones 2D (XY, YZ, XZ) coloreadas por cluster."""
    x, y, z = coord_cols
    lab = coord_labels or coord_cols
    k = n_clusters or int(df[cluster_col].nunique())
    cmap, norm = _cluster_cmap_norm(k)
    pairs = [(x, y), (y, z), (x, z)]
    pair_labels = [(lab[0], lab[1]), (lab[1], lab[2]), (lab[0], lab[2])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (c1, c2), (l1, l2) in zip(axes, pairs, pair_labels):
        ax.scatter(
            df[c1], df[c2], c=df[cluster_col], cmap=cmap, norm=norm,
            s=12, alpha=0.65,
        )
        ax.set_xlabel(l1)
        ax.set_ylabel(l2)
        ax.set_title(f"{l1} vs {l2}")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, weight="bold", y=1.02)
    plt.tight_layout()
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
    title_prefix: str = "Deriva",
) -> None:
    """Gráficos de deriva (promedio por bins) para cada coordenada."""
    x_labels = coord_labels or coord_cols
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
        ax.set_title(f"{title_prefix} según {x_label}")
    plt.tight_layout()
    ylim = (0, df[value_col].max())
    plt.ylim(ylim)
    plt.show()


def stats_by_cluster(
    df: pd.DataFrame,
    target_col: str,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Estadísticos descriptivos de la variable objetivo por cluster."""
    return (
        df.groupby(cluster_col)[target_col]
        .agg(count="count", media="mean", desv_std="std", min="min", max="max", mediana="median")
        .reset_index()
    )


def run_clustering_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    cluster_col: str,
    mode_label: str,
    coord_labels: List[str],
    nscore_col: str,
    target_col: str,
    n_clusters: int,
    k_range: Tuple[int, int],
) -> pd.DataFrame:
    """
    Ejecuta exploración de k, clustering y todos los gráficos para un modo (con o sin nscore).
    En modo sin nscore se usa target_col en el resumen para que el eje X del gráfico de
    probabilidad tenga el rango en unidades originales.
    """
    k_min, k_max = k_range
    print(f"\n{'='*60}\n  {mode_label}\n{'='*60}")

    k_results = evaluate_k_range(df, feature_cols, k_min=k_min, k_max=k_max)
    print("Métricas por k:")
    print(k_results.to_string(index=False))
    plot_k_selection(k_results, title=f"Selección de k — {mode_label}")

    df = fit_clusters(df, feature_cols, n_clusters, cluster_col=cluster_col)
    print(f"Clustering aplicado → columna '{cluster_col}'.")

    # Con nscore: resumen en normal score. Sin nscore: resumen en variable original (eje X con rango real)
    use_target_in_summary = nscore_col not in feature_cols
    value_col_summary = target_col if use_target_in_summary else None
    plot_cluster_summary(
        df, nscore_col, COORDS, n_clusters,
        value_label=TARGET_LABEL,
        cluster_col=cluster_col,
        suptitle_prefix=mode_label,
        value_col=value_col_summary,
    )
    plot_clusters_3d(
        df, COORDS, cluster_col=cluster_col, n_clusters=n_clusters,
        coord_labels=coord_labels,
        title=f"Distribución espacial de clusters — {mode_label}",
    )
    plot_clusters_2d(
        df, COORDS, cluster_col=cluster_col, n_clusters=n_clusters,
        coord_labels=coord_labels,
        suptitle=f"Proyecciones 2D — {mode_label}",
    )
    stats_df = stats_by_cluster(df, TARGET, cluster_col=cluster_col)
    print("Estadísticas por cluster:")
    print(stats_df.to_string(index=False))
    return df


# --- Ejecución ---
# %%
if __name__ == "__main__":
    setup_report_style()
    coord_labels = COORD_LABELS or COORDS
    nscore_col = f"{TARGET}{NSCORE_SUFFIX}"
    k_min, k_max = K_RANGE

    df = load_processed_data(INPUT_PATH)
    if nscore_col not in df.columns:
        raise ValueError(
            f"Columna '{nscore_col}' no encontrada. Ejecuta antes Analisis_EDA.py."
        )
    print("Columnas cargadas:", list(df.columns))
    print("Variable en estudio:", TARGET_LABEL, "| Clusters:", N_CLUSTERS)

    if RUN_CON_NSCORE:
        feature_cols_con = COORDS + [nscore_col]
        df = run_clustering_pipeline(
            df, feature_cols_con, "cluster_con_nscore", "Con nscore (coords + variable)",
            coord_labels, nscore_col, TARGET, N_CLUSTERS, K_RANGE,
        )

    if RUN_SIN_NSCORE:
        feature_cols_sin = COORDS
        df = run_clustering_pipeline(
            df, feature_cols_sin, "cluster_sin_nscore", "Sin nscore (solo coords)",
            coord_labels, nscore_col, TARGET, N_CLUSTERS, K_RANGE,
        )
    # df tiene ahora cluster_con_nscore y/o cluster_sin_nscore según lo ejecutado

    # %%
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_CON_NSCORE:
        out_con = OUTPUT_DIR / "clusters_df_con_nscore.csv"
        df.to_csv(out_con, index=False)
        print(f"Guardado (con nscore): {out_con}")
    if RUN_SIN_NSCORE:
        out_sin = OUTPUT_DIR / "clusters_df_sin_nscore.csv"
        df.to_csv(out_sin, index=False)
        print(f"Guardado (sin nscore): {out_sin}")

    # %%
    # Deriva para un cluster concreto (elegir cluster_col según el modo que quieras analizar)
    CLUSTER_PARA_DERIVA = 0
    cluster_col_deriva = "cluster_con_nscore" if RUN_CON_NSCORE else "cluster_sin_nscore"
    df_sub = df[df[cluster_col_deriva] == CLUSTER_PARA_DERIVA].copy()
    plot_drift(
        df_sub, COORDS, TARGET, n_bins=20,
        coord_labels=coord_labels,
        value_label=TARGET_LABEL,
        title_prefix=f"Deriva (cluster {CLUSTER_PARA_DERIVA}, {cluster_col_deriva})",
    )

# %%
