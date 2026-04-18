# %%
"""
Visualización 3D por clusters, carga de datos de validación y asignación de cluster (KNN).
Se genera un CSV con puntos originales y puntos de validación (columna origen: original | validacion).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- Configuración ---
INPUT_PATH = Path("data/processed/cluster/clusters_df_con_nscore.csv")
INPUT_VAL_PATH = Path("data/validacion/modelo_validos.csv")
COORD_COLS = ["Este", "Norte", "Cota"]
COORD_LABELS = ["Este (m)", "Norte (m)", "Cota (m)"]
VAL_COORD_COLS = ["xcentre", "ycentre", "zcentre"]
CLUSTER_COL = "cluster_con_nscore"
TARGET_COL = "Rec_Peso_PND25_(%)_nscore"
TARGET_LABEL = "Recuperación en peso (%) (normal score)"

# KNN: vecinos para asignar cluster a cada punto de validación (entrenado con originales)
KNN_NEIGHBORS = 5
# CSV con puntos originales + puntos de validación (columna origen: original | validacion)
OUTPUT_COMBINED_PATH = Path("data/processed/cluster/puntos_originales_y_validacion.csv")
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


def set_proportional_aspect(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Ajusta el aspect ratio del gráfico 3D según el rango real de las coordenadas."""
    range_x = x.max() - x.min()
    range_y = y.max() - y.min()
    range_z = z.max() - z.min()
    # Evitar divisiones por cero si un rango es nulo
    max_range = max(range_x, range_y, range_z)
    if max_range > 0:
        ax.set_box_aspect((range_x / max_range, range_y / max_range, range_z / max_range))


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
    """Scatter 3D coloreado por cluster con aspecto proporcional."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap, norm = _cluster_cmap_norm(n_clusters)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=clusters, cmap=cmap, norm=norm, s=s, alpha=alpha)
    
    ax.set_xlabel(coord_labels[0])
    ax.set_ylabel(coord_labels[1])
    ax.set_zlabel(coord_labels[2])
    ax.set_title(title)
    
    set_proportional_aspect(ax, x, y, z)
    
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.12, label="Cluster", ticks=np.arange(n_clusters))
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_3d_original_plus_val(
    coords_orig: np.ndarray,
    labels_orig: np.ndarray,
    coords_val: np.ndarray,
    coord_labels: list[str],
    title: str,
    n_clusters: int,
    save_path: Path | None = None,
) -> None:
    """3D: originales coloreados por cluster + puntos de validación en gris."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap, norm = _cluster_cmap_norm(n_clusters)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords_orig[:, 0], coords_orig[:, 1], coords_orig[:, 2],
        c=labels_orig, cmap=cmap, norm=norm, s=25, alpha=0.8, label="Originales",
    )
    if len(coords_val) > 0:
        ax.scatter(
            coords_val[:, 0], coords_val[:, 1], coords_val[:, 2],
            c="gray", s=10, alpha=0.4, label="Validación (sin cluster)",
        )
    
    ax.set_xlabel(coord_labels[0])
    ax.set_ylabel(coord_labels[1])
    ax.set_zlabel(coord_labels[2])
    ax.set_title(title)
    
    all_x = np.concatenate([coords_orig[:, 0], coords_val[:, 0]])
    all_y = np.concatenate([coords_orig[:, 1], coords_val[:, 1]])
    all_z = np.concatenate([coords_orig[:, 2], coords_val[:, 2]])
    set_proportional_aspect(ax, all_x, all_y, all_z)
    
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

    # 1) Datos Originales
    df_orig = pd.read_csv(INPUT_PATH)
    coords_orig = df_orig[COORD_COLS].values
    labels_orig = df_orig[CLUSTER_COL].values
    n_clusters = int(labels_orig.max()) + 1

    # 2) Datos de Validación
    if not INPUT_VAL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de validación en {INPUT_VAL_PATH}")
    
    df_val = pd.read_csv(INPUT_VAL_PATH)
    # Renombrar columnas de validación a los nombres estándar del proyecto
    df_val = df_val.rename(columns={
        VAL_COORD_COLS[0]: COORD_COLS[0],
        VAL_COORD_COLS[1]: COORD_COLS[1],
        VAL_COORD_COLS[2]: COORD_COLS[2]
    })
    coords_val = df_val[COORD_COLS].values

    # 3) Visualización Inicial
    plot_3d_clusters(
        coords_orig[:, 0], coords_orig[:, 1], coords_orig[:, 2],
        labels_orig,
        COORD_LABELS,
        "Datos originales — coloreados por cluster",
        n_clusters,
        save_path=IMAGENES_DIR / "estimacion_3d_originales_cluster.png",
    )

    # 4) Visualización Originales + Validación (sin cluster aún)
    plot_3d_original_plus_val(
        coords_orig, labels_orig, coords_val,
        COORD_LABELS,
        "Originales + Puntos de Validación (sin cluster)",
        n_clusters,
        save_path=IMAGENES_DIR / "estimacion_3d_originales_mas_validacion.png",
    )

    # 5) Asignación de Cluster mediante KNN
    scaler = StandardScaler()
    coords_orig_scaled = scaler.fit_transform(coords_orig)
    coords_val_scaled = scaler.transform(coords_val)
    
    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    knn.fit(coords_orig_scaled, labels_orig)
    labels_val = knn.predict(coords_val_scaled)
    
    df_val[CLUSTER_COL] = labels_val

    # 6) Visualización Final: Todo coloreado por cluster
    all_coords = np.vstack([coords_orig, coords_val])
    all_labels = np.concatenate([labels_orig, labels_val])
    
    plot_3d_clusters(
        all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
        all_labels,
        COORD_LABELS,
        f"Originales + Validación con cluster (KNN k={KNN_NEIGHBORS})",
        n_clusters,
        s=15,
        alpha=0.5,
        save_path=IMAGENES_DIR / "estimacion_3d_originales_validacion_con_cluster.png",
    )

    # 7) Guardar CSV Combinado
    df_orig_out = df_orig.copy()
    df_orig_out["origen"] = "original"
    
    df_val_out = df_val.copy()
    df_val_out["origen"] = "validacion"
    
    # Asegurar que tengan las mismas columnas (rellenar con NaN las faltantes)
    for col in df_orig_out.columns:
        if col not in df_val_out.columns:
            df_val_out[col] = np.nan
            
    df_combined = pd.concat([df_orig_out, df_val_out], ignore_index=True)
    OUTPUT_COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(OUTPUT_COMBINED_PATH, index=False)
    
    print(f"CSV guardado: {OUTPUT_COMBINED_PATH}")
    print(f"  - Originales: {len(df_orig_out)} filas")
    print(f"  - Validación: {len(df_val_out)} filas")

# %%
