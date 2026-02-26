# %%
"""
Visualización 3D por clusters, relleno del convex hull y asignación de cluster (KNN).
Se genera un CSV con puntos originales y puntos de relleno (columna origen: original | relleno).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial import ConvexHull
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- Configuración ---
INPUT_PATH = Path("data/processed/cluster/clusters_df_con_nscore.csv")
COORD_COLS = ["Este", "Norte", "Cota"]
COORD_LABELS = ["Este (m)", "Norte (m)", "Cota (m)"]
CLUSTER_COL = "cluster_con_nscore"
TARGET_COL = "Rec_Peso_PND25_(%)_nscore"
TARGET_LABEL = "Recuperación en peso (%) (normal score)"

# Puntos con los que rellenar el interior de cada convex hull (repartidos entre clusters)
N_FILL = 5000
# KNN: vecinos para asignar cluster a cada punto de relleno (entrenado con originales)
KNN_NEIGHBORS = 5
# CSV con puntos originales + puntos de relleno (columna origen: original | relleno)
OUTPUT_COMBINED_PATH = Path("data/processed/cluster/puntos_originales_y_relleno.csv")
IMAGENES_DIR = Path("imagenes")
# Paleta (misma que Analisis_cluster)
CLUSTER_PALETTE = [
    "#2563eb", "#dc2626", "#059669", "#d97706",
    "#7c3aed", "#0d9488", "#ea580c", "#4f46e5",
]


def setup_report_style() -> None:
    """Estilo de figuras (misma línea que Analisis_EDA / Analisis_cluster)."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _cluster_cmap_norm(n_clusters: int) -> tuple[ListedColormap, BoundaryNorm]:
    """Colormap y norma discretos para la colorbar."""
    colors = CLUSTER_PALETTE[:n_clusters]
    cmap = ListedColormap(colors)
    boundaries = np.arange(n_clusters + 1) - 0.5
    norm = BoundaryNorm(boundaries, n_clusters)
    return cmap, norm


def sample_inside_hull(hull: ConvexHull, n: int) -> np.ndarray:
    """Muestrea n puntos uniformes dentro del convex hull 3D (rejection sampling)."""
    pts = hull.points
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    eq = hull.equations  # (n_eq, 4): interior <= 0
    out = []
    max_attempts = n * 50
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        p = np.random.uniform(lo, hi, size=3)
        if (eq[:, :3] @ p + eq[:, 3] <= 1e-8).all():
            out.append(p)
        attempts += 1
    return np.array(out) if out else np.empty((0, 3))


def fill_hull_per_cluster(
    coords: np.ndarray,
    labels: np.ndarray,
    n_fill: int,
) -> np.ndarray:
    """
    Aplica convex hull a cada cluster y rellena el interior. N_FILL se reparte entre clusters.
    Devuelve solo coords_relleno (sin cluster; lo asignas tú después).
    """
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    n_per_cluster = max(1, n_fill // n_clusters)
    filled_coords = []

    for c in clusters:
        mask = labels == c
        pts = coords[mask]
        if pts.shape[0] < 4:
            continue
        try:
            hull = ConvexHull(pts)
            samples = sample_inside_hull(hull, n_per_cluster)
            if len(samples) > 0:
                filled_coords.append(samples)
        except Exception:
            continue

    if not filled_coords:
        return np.empty((0, 3))
    return np.vstack(filled_coords)


def plot_3d_clusters(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    clusters: np.ndarray,
    coord_labels: list[str],
    title: str,
    n_clusters: int,
    s: int = 25,
    alpha: float = 0.7,
    save_path: Path | None = None,
) -> None:
    """Scatter 3D coloreado por cluster (misma línea que Analisis_cluster)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap, norm = _cluster_cmap_norm(n_clusters)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=clusters, cmap=cmap, norm=norm, s=s, alpha=alpha)
    ax.set_xlabel(coord_labels[0])
    ax.set_ylabel(coord_labels[1])
    ax.set_zlabel(coord_labels[2])
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.12, label="Cluster", ticks=np.arange(n_clusters))
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_3d_original_plus_filled(
    coords_orig: np.ndarray,
    labels_orig: np.ndarray,
    coords_filled: np.ndarray,
    coord_labels: list[str],
    title: str,
    n_clusters: int,
    save_path: Path | None = None,
) -> None:
    """3D: originales coloreados por cluster + puntos de relleno en gris (sin cluster)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap, norm = _cluster_cmap_norm(n_clusters)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords_orig[:, 0], coords_orig[:, 1], coords_orig[:, 2],
        c=labels_orig, cmap=cmap, norm=norm, s=25, alpha=0.8, label="Originales",
    )
    if len(coords_filled) > 0:
        ax.scatter(
            coords_filled[:, 0], coords_filled[:, 1], coords_filled[:, 2],
            c="gray", s=10, alpha=0.4, label="Relleno (sin cluster)",
        )
    ax.set_xlabel(coord_labels[0])
    ax.set_ylabel(coord_labels[1])
    ax.set_zlabel(coord_labels[2])
    ax.set_title(title)
    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, shrink=0.5, pad=0.12, label="Cluster", ticks=np.arange(n_clusters),
    )
    ax.legend()
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
    coords = df[COORD_COLS].values
    labels = df[CLUSTER_COL].values
    n_clusters = int(labels.max()) + 1

    # 1) 3D solo datos originales, coloreados por cluster
    plot_3d_clusters(
        coords[:, 0], coords[:, 1], coords[:, 2],
        labels,
        COORD_LABELS,
        "Datos originales — coloreados por cluster",
        n_clusters,
        save_path=IMAGENES_DIR / "estimacion_3d_originales_cluster.png",
    )

    # 2) Convex hull por cluster + relleno (puntos sin cluster; los asignas tú después)
    filled_coords = fill_hull_per_cluster(coords, labels, N_FILL)

    if len(filled_coords) > 0:
        # Gráfico: originales + relleno en gris (sin cluster aún)
        plot_3d_original_plus_filled(
            coords, labels, filled_coords,
            COORD_LABELS,
            f"Originales + relleno (N_FILL={N_FILL})",
            n_clusters,
            save_path=IMAGENES_DIR / "estimacion_3d_originales_mas_relleno.png",
        )

        # KNN: asignar cluster a cada punto de relleno (entrenado con originales, coords estandarizadas)
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        filled_scaled = scaler.transform(filled_coords)
        knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
        knn.fit(coords_scaled, labels)
        filled_labels = knn.predict(filled_scaled)

        df_filled = pd.DataFrame(filled_coords, columns=COORD_COLS)
        df_filled[CLUSTER_COL] = filled_labels

        # Gráfico: originales + relleno, todos coloreados por cluster
        all_coords = np.vstack([coords, filled_coords])
        all_labels = np.concatenate([labels, filled_labels])
        plot_3d_clusters(
            all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
            all_labels,
            COORD_LABELS,
            f"Originales + relleno con cluster (KNN k={KNN_NEIGHBORS})",
            n_clusters,
            s=15,
            alpha=0.5,
            save_path=IMAGENES_DIR / "estimacion_3d_originales_relleno_con_cluster.png",
        )

        # CSV único: puntos originales + puntos de relleno (columna origen)
        df_orig_out = df.copy()
        df_orig_out["origen"] = "original"
        for c in df.columns:
            if c not in df_filled.columns:
                df_filled[c] = np.nan
        df_filled["origen"] = "relleno"
        df_combined = pd.concat([df_orig_out, df_filled[df_orig_out.columns]], ignore_index=True)
        OUTPUT_COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(OUTPUT_COMBINED_PATH, index=False)
        print(f"CSV guardado: {OUTPUT_COMBINED_PATH} ({len(df_orig_out)} originales + {len(df_filled)} relleno = {len(df_combined)} filas)")
    else:
        print("No se generaron puntos de relleno (revisar N_FILL o datos).")

# %%
