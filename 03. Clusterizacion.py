#%%
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config_figuras_tesina as cfg_fig

cfg_fig = importlib.reload(cfg_fig)

plot_mapa_3d_clusters = cfg_fig.plot_mapa_3d_clusters
plot_boxplot_efecto_proporcional_por_cluster = (
    cfg_fig.plot_boxplot_efecto_proporcional_por_cluster
)
plot_proyecciones_2d_clusters = cfg_fig.plot_proyecciones_2d_clusters
setup_figuras_tesina = cfg_fig.setup_figuras_tesina


DATA_PATH = Path("data") / "processed" / "df_rec_peso_pnd25_gauss.xlsx"
FEATURE_COLUMNS = ["Este", "Norte", "Cota", "recpe_gauss"]
TARGET_METRIC_COLUMN = "recpe_og"
N_CLUSTERS = 3
SEED_CANDIDATES = [7, 21, 42, 84, 123]
SVM_GAMMA_CANDIDATES = [0.2, 0.5, 1.0, 2.0]
SVM_NEIGHBOR_CANDIDATES = [10, 20, 30]
SVM_ASSIGN_LABELS = ["kmeans", "discretize"]
GMM_COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]
OUTPUT_PATH_KMEANS = Path("data") / "processed" / "df_rec_peso_pnd25_gauss_kmeans_clusters.xlsx"
OUTPUT_PATH_SVM = Path("data") / "processed" / "df_rec_peso_pnd25_gauss_svm_clusters.xlsx"
OUTPUT_PATH_GMM = Path("data") / "processed" / "df_rec_peso_pnd25_gauss_gmm_clusters.xlsx"
CLUSTER_IMAGE_DIR_KMEANS = Path("imagenes") / "cluster" / "kmeans"
CLUSTER_IMAGE_DIR_SVM = Path("imagenes") / "cluster" / "svm"
CLUSTER_IMAGE_DIR_GMM = Path("imagenes") / "cluster" / "gmm"
SHOW_FIGURES = True


def load_data(file_path: Path) -> pd.DataFrame:
    """Carga el archivo de datos para clusterizacion."""
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {file_path.resolve()}")
    return pd.read_excel(file_path)


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Valida que existan las columnas necesarias."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")


def evaluate_cluster_separation(
    labels_zero_based: list[int] | pd.Series,
    target_values: pd.Series,
    scaled_values,
) -> tuple[float, float]:
    """Calcula separacion de medias (objetivo) y silhouette (apoyo)."""
    labels_series = pd.Series(labels_zero_based, index=target_values.index)
    mean_by_cluster = target_values.groupby(labels_series).mean().sort_values()
    if len(mean_by_cluster) < 2:
        return 0.0, -1.0

    min_pairwise_gap = float("inf")
    mean_values = mean_by_cluster.tolist()
    for idx in range(len(mean_values) - 1):
        gap = abs(mean_values[idx + 1] - mean_values[idx])
        min_pairwise_gap = min(min_pairwise_gap, gap)

    silhouette = silhouette_score(scaled_values, labels_series.to_numpy())
    return float(min_pairwise_gap), float(silhouette)


def fit_best_kmeans(
    feature_df: pd.DataFrame,
    target_values: pd.Series,
    n_clusters: int,
    seed_candidates: list[int],
) -> tuple[list[int], float, float, int]:
    """Ajusta KMeans y selecciona la mejor separacion de medias."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_df)

    best_labels = None
    best_gap = float("-inf")
    best_silhouette = float("-inf")
    best_seed = seed_candidates[0]

    for seed in seed_candidates:
        model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=20,
            random_state=seed,
        )
        labels = model.fit_predict(scaled_values)
        mean_gap, silhouette = evaluate_cluster_separation(labels, target_values, scaled_values)
        if (mean_gap > best_gap) or (
            mean_gap == best_gap and silhouette > best_silhouette
        ):
            best_gap = mean_gap
            best_silhouette = silhouette
            best_labels = labels
            best_seed = seed

    if best_labels is None:
        raise ValueError("No fue posible ajustar un modelo KMeans valido.")

    return best_labels.tolist(), best_gap, best_silhouette, best_seed


def fit_best_svm_kernel_clustering(
    feature_df: pd.DataFrame,
    target_values: pd.Series,
    n_clusters: int,
    gamma_candidates: list[float],
    neighbor_candidates: list[int],
    assign_labels_candidates: list[str],
    seed_candidates: list[int],
) -> tuple[list[int], float, float, str, int]:
    """Aproxima clusterizacion tipo SVM via kernel RBF (SpectralClustering)."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_df)

    best_labels = None
    best_gap = float("-inf")
    best_silhouette = float("-inf")
    best_gamma = gamma_candidates[0]
    best_affinity = "rbf"
    best_assign_labels = assign_labels_candidates[0]
    best_seed = seed_candidates[0]

    for assign_labels in assign_labels_candidates:
        for gamma in gamma_candidates:
            for seed in seed_candidates:
                model = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="rbf",
                    gamma=gamma,
                    assign_labels=assign_labels,
                    random_state=seed,
                )
                labels = model.fit_predict(scaled_values)
                mean_gap, silhouette = evaluate_cluster_separation(labels, target_values, scaled_values)
                if (mean_gap > best_gap) or (
                    mean_gap == best_gap and silhouette > best_silhouette
                ):
                    best_gap = mean_gap
                    best_silhouette = silhouette
                    best_labels = labels
                    best_gamma = gamma
                    best_affinity = "rbf"
                    best_assign_labels = assign_labels
                    best_seed = seed

        for n_neighbors in neighbor_candidates:
            for seed in seed_candidates:
                model = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="nearest_neighbors",
                    n_neighbors=n_neighbors,
                    assign_labels=assign_labels,
                    random_state=seed,
                )
                labels = model.fit_predict(scaled_values)
                mean_gap, silhouette = evaluate_cluster_separation(labels, target_values, scaled_values)
                if (mean_gap > best_gap) or (
                    mean_gap == best_gap and silhouette > best_silhouette
                ):
                    best_gap = mean_gap
                    best_silhouette = silhouette
                    best_labels = labels
                    best_gamma = float(n_neighbors)
                    best_affinity = "nearest_neighbors"
                    best_assign_labels = assign_labels
                    best_seed = seed

    if best_labels is None:
        raise ValueError("No fue posible ajustar un modelo SVM-kernel valido.")

    return (
        best_labels.tolist(),
        best_gap,
        best_silhouette,
        best_gamma,
        f"{best_affinity}:{best_assign_labels}",
        best_seed,
    )


def fit_best_gmm(
    feature_df: pd.DataFrame,
    target_values: pd.Series,
    n_clusters: int,
    covariance_types: list[str],
    seed_candidates: list[int],
) -> tuple[list[int], float, float, str, int]:
    """Ajusta GMM y selecciona la mejor separacion de medias."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_df)

    best_labels = None
    best_gap = float("-inf")
    best_silhouette = float("-inf")
    best_covariance = covariance_types[0]
    best_seed = seed_candidates[0]

    for covariance_type in covariance_types:
        for seed in seed_candidates:
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type=covariance_type,
                random_state=seed,
                n_init=5,
            )
            labels = model.fit_predict(scaled_values)
            mean_gap, silhouette = evaluate_cluster_separation(labels, target_values, scaled_values)
            if (mean_gap > best_gap) or (
                mean_gap == best_gap and silhouette > best_silhouette
            ):
                best_gap = mean_gap
                best_silhouette = silhouette
                best_labels = labels
                best_covariance = covariance_type
                best_seed = seed

    if best_labels is None:
        raise ValueError("No fue posible ajustar un modelo GMM valido.")

    return best_labels.tolist(), best_gap, best_silhouette, best_covariance, best_seed


def relabel_clusters_from_zero_to_one(labels_zero_based: list[int]) -> list[int]:
    """Convierte etiquetas de 0..k-1 a 1..k."""
    return [label + 1 for label in labels_zero_based]


def save_clustered_data(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el DataFrame clusterizado."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def get_cluster_descriptive_stats(df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    """Construye estadisticas descriptivas por cluster."""
    grouped_stats = (
        df.groupby("cluster", observed=False)
        .agg(
            n_muestras=("cluster", "size"),
            recpe_og_mean=("recpe_og", "mean"),
            recpe_og_std=("recpe_og", "std"),
            recpe_og_min=("recpe_og", "min"),
            recpe_og_q25=("recpe_og", lambda s: s.quantile(0.25)),
            recpe_og_median=("recpe_og", "median"),
            recpe_og_q75=("recpe_og", lambda s: s.quantile(0.75)),
            recpe_og_max=("recpe_og", "max"),
            recpe_gauss_mean=("recpe_gauss", "mean"),
            recpe_gauss_std=("recpe_gauss", "std"),
            este_mean=("Este", "mean"),
            norte_mean=("Norte", "mean"),
            cota_mean=("Cota", "mean"),
        )
        .reset_index()
        .sort_values("cluster")
    )
    grouped_stats.insert(0, "metodo", method_name)
    return grouped_stats


def maybe_show_figure(fig, show_figures: bool) -> None:
    """Muestra la figura o la cierra segun configuracion."""
    if show_figures:
        plt.show()
    else:
        plt.close(fig)


def save_cluster_plots(df: pd.DataFrame, output_dir: Path, show_figures: bool) -> None:
    """Guarda graficos de clusterizacion."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mapa 3D por cluster
    fig_3d = plot_mapa_3d_clusters(
        df=df,
        x_col="Este",
        y_col="Norte",
        z_col="Cota",
        cluster_col="cluster",
        titulo="Distribución Espacial de la Recuperación en Peso (%)",
    )
    fig_3d.savefig(output_dir / f"{output_dir.name}_clusters_3d.png", dpi=300)
    maybe_show_figure(fig_3d, show_figures)

    # 2) Proyecciones 2D por cluster
    fig_2d = plot_proyecciones_2d_clusters(
        df=df,
        x_col="Este",
        y_col="Norte",
        z_col="Cota",
        cluster_col="cluster",
    )
    fig_2d.savefig(output_dir / f"{output_dir.name}_clusters_proyecciones_2d.png", dpi=300)
    maybe_show_figure(fig_2d, show_figures)

    # 3) Boxplot y coeficiente de variacion por cluster
    fig_stats = plot_boxplot_efecto_proporcional_por_cluster(
        df=df,
        cluster_col="cluster",
        target_col="recpe_og",
    )
    fig_stats.savefig(
        output_dir / f"{output_dir.name}_clusters_boxplot_efecto_proporcional.png",
        dpi=300,
    )
    maybe_show_figure(fig_stats, show_figures)


setup_figuras_tesina()
cluster_df = load_data(DATA_PATH)
validate_columns(cluster_df, FEATURE_COLUMNS)

labels_zero_kmeans, best_gap_kmeans, best_silhouette_kmeans, best_seed_kmeans = fit_best_kmeans(
    feature_df=cluster_df[FEATURE_COLUMNS],
    target_values=cluster_df[TARGET_METRIC_COLUMN],
    n_clusters=N_CLUSTERS,
    seed_candidates=SEED_CANDIDATES,
)
cluster_df_kmeans = cluster_df.copy()
cluster_df_kmeans["cluster"] = relabel_clusters_from_zero_to_one(labels_zero_kmeans)

save_clustered_data(cluster_df_kmeans, OUTPUT_PATH_KMEANS)
save_cluster_plots(cluster_df_kmeans, CLUSTER_IMAGE_DIR_KMEANS, SHOW_FIGURES)
stats_kmeans = get_cluster_descriptive_stats(cluster_df_kmeans, method_name="kmeans")

(
    labels_zero_svm,
    best_gap_svm,
    best_silhouette_svm,
    best_gamma_svm,
    best_svm_mode,
    best_seed_svm,
) = (
    fit_best_svm_kernel_clustering(
        feature_df=cluster_df[FEATURE_COLUMNS],
        target_values=cluster_df[TARGET_METRIC_COLUMN],
        n_clusters=N_CLUSTERS,
        gamma_candidates=SVM_GAMMA_CANDIDATES,
        neighbor_candidates=SVM_NEIGHBOR_CANDIDATES,
        assign_labels_candidates=SVM_ASSIGN_LABELS,
        seed_candidates=SEED_CANDIDATES,
    )
)
cluster_df_svm = cluster_df.copy()
cluster_df_svm["cluster"] = relabel_clusters_from_zero_to_one(labels_zero_svm)

save_clustered_data(cluster_df_svm, OUTPUT_PATH_SVM)
save_cluster_plots(cluster_df_svm, CLUSTER_IMAGE_DIR_SVM, SHOW_FIGURES)
stats_svm = get_cluster_descriptive_stats(cluster_df_svm, method_name="svm_kernel")

(
    labels_zero_gmm,
    best_gap_gmm,
    best_silhouette_gmm,
    best_covariance_gmm,
    best_seed_gmm,
) = fit_best_gmm(
    feature_df=cluster_df[FEATURE_COLUMNS],
    target_values=cluster_df[TARGET_METRIC_COLUMN],
    n_clusters=N_CLUSTERS,
    covariance_types=GMM_COVARIANCE_TYPES,
    seed_candidates=SEED_CANDIDATES,
)
cluster_df_gmm = cluster_df.copy()
cluster_df_gmm["cluster"] = relabel_clusters_from_zero_to_one(labels_zero_gmm)

save_clustered_data(cluster_df_gmm, OUTPUT_PATH_GMM)
save_cluster_plots(cluster_df_gmm, CLUSTER_IMAGE_DIR_GMM, SHOW_FIGURES)
stats_gmm = get_cluster_descriptive_stats(cluster_df_gmm, method_name="gmm")

print(
    "KMeans | mejor separacion de medias: "
    f"{best_gap_kmeans:.4f} | silhouette: {best_silhouette_kmeans:.4f} "
    f"(seed={best_seed_kmeans})"
)
print(cluster_df_kmeans["cluster"].value_counts().sort_index())
print("Estadisticas descriptivas por cluster (KMeans):")
print(stats_kmeans.to_string(index=False))
print(f"Archivo KMeans: {OUTPUT_PATH_KMEANS.resolve()}")
print(f"Imagenes KMeans: {CLUSTER_IMAGE_DIR_KMEANS.resolve()}")

print(
    "SVM-kernel | mejor separacion de medias: "
    f"{best_gap_svm:.4f} | silhouette: {best_silhouette_svm:.4f} "
    f"(modo={best_svm_mode}, parametro={best_gamma_svm}, seed={best_seed_svm})"
)
print(cluster_df_svm["cluster"].value_counts().sort_index())
print("Estadisticas descriptivas por cluster (SVM-kernel):")
print(stats_svm.to_string(index=False))
print(f"Archivo SVM: {OUTPUT_PATH_SVM.resolve()}")
print(f"Imagenes SVM: {CLUSTER_IMAGE_DIR_SVM.resolve()}")

print(
    "GMM | mejor separacion de medias: "
    f"{best_gap_gmm:.4f} | silhouette: {best_silhouette_gmm:.4f} "
    f"(covariance={best_covariance_gmm}, seed={best_seed_gmm})"
)
print(cluster_df_gmm["cluster"].value_counts().sort_index())
print("Estadisticas descriptivas por cluster (GMM):")
print(stats_gmm.to_string(index=False))
print(f"Archivo GMM: {OUTPUT_PATH_GMM.resolve()}")
print(f"Imagenes GMM: {CLUSTER_IMAGE_DIR_GMM.resolve()}")
# %%
