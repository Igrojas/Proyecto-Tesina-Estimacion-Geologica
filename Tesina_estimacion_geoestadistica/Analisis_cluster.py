#%%
"""
Análisis de clustering (KMeans y DBSCAN) sobre coordenadas (midx, midy, midz)
y variable de ley; incluye análisis por cluster y deriva.
"""

import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1) Carga y preparación
# ---------------------------------------------------------------------------

def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el DataFrame desde Excel. Asegura columnas midx, midy, midz y variable de ley."""
    df = pd.read_excel(ruta)
    return df


# ---------------------------------------------------------------------------
# 2) Clustering
# ---------------------------------------------------------------------------

def clustering_kmeans(
    df: pd.DataFrame,
    n_clusters: int,
    coord_cols: list[str] | None = None,
    var_extra: str | None = 'cut_nscore',
    random_state: int = 0,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    KMeans sobre coordenadas estandarizadas y opcionalmente una variable extra (ej. cut_nscore).

    Returns:
        (df con columna 'cluster', scaler ya ajustado)
    """
    coord_cols = coord_cols or ['midx', 'midy', 'midz']
    X_coords = df[coord_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_coords)

    if var_extra and var_extra in df.columns:
        X = np.hstack([X_scaled, df[[var_extra]].values])
    else:
        X = X_scaled

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)

    df_out = df.copy()
    df_out['cluster'] = labels
    return df_out, scaler


def clustering_dbscan(
    df: pd.DataFrame,
    eps: float = 2,
    min_samples: int = 5,
    coord_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    DBSCAN solo sobre coordenadas estandarizadas (espacial).
    Etiqueta -1 = ruido.

    Returns:
        (df con columna 'cluster', scaler ya ajustado)
    """
    coord_cols = coord_cols or ['midx', 'midy', 'midz']
    X = df[coord_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    df_out = df.copy()
    df_out['cluster'] = labels
    return df_out, scaler


# ---------------------------------------------------------------------------
# 3) Análisis de resultados (común para KMeans y DBSCAN)
# ---------------------------------------------------------------------------

def analisis_por_cluster(
    df: pd.DataFrame,
    variable_ley: str = 'cut',
    variable_nscore: str = 'cut_nscore',
    titulo_suffix: str = '',
) -> None:
    """
    Gráficos y estadísticos por cluster: boxplot, proporción, Q-Q, 3D y proyecciones 2D.
    Usar el mismo flujo para resultados de KMeans o DBSCAN.
    """
    clusters_unicos = sorted(df['cluster'].dropna().unique())
    n_clusters = len(clusters_unicos)

    # Estadísticos descriptivos por cluster
    if variable_ley in df.columns:
        stats_cluster = df.groupby('cluster', dropna=False).agg(
            count=(variable_ley, 'count'),
            mean=(variable_ley, 'mean'),
            std=(variable_ley, 'std'),
            min=(variable_ley, 'min'),
            max=(variable_ley, 'max'),
            median=(variable_ley, 'median'),
        ).reset_index()
        print(f"Estadísticas descriptivas por cluster ({variable_ley})" + (f" — {titulo_suffix}" if titulo_suffix else ""))
        print(stats_cluster)

    # Gráficos 1x3: boxplot, proporción, Q-Q
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(x='cluster', y=variable_nscore, data=df, ax=axs[0], palette='Set2')
    axs[0].set_title(f'Boxplot {variable_nscore} por cluster')
    axs[0].set_xlabel('Cluster')
    axs[0].set_ylabel(variable_nscore)

    prop = df['cluster'].value_counts(normalize=True).sort_index()
    axs[1].bar(prop.index.astype(str), prop.values, color='skyblue')
    axs[1].set_title('Proporción de cada cluster')
    axs[1].set_xlabel('Cluster')
    axs[1].set_ylabel('Proporción')
    axs[1].set_ylim(0, 1)

    for c in clusters_unicos:
        mask = df['cluster'] == c
        if mask.sum() > 0 and variable_nscore in df.columns:
            vals = df.loc[mask, variable_nscore].dropna()
            if len(vals) > 0:
                res = probplot(vals, dist='norm', plot=None)
                axs[2].plot(res[0][0], res[0][1], marker='.', linestyle='', label=f'Cluster {c}')
    axs[2].set_title(f'Gráfico log-probabilístico {variable_nscore}')
    axs[2].set_xlabel('Quantiles teóricos')
    axs[2].set_ylabel(f'{variable_nscore} ordenado')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    if titulo_suffix:
        fig.suptitle(titulo_suffix, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

    # 3D
    fig3d = plt.figure(figsize=(10, 7))
    ax3d = fig3d.add_subplot(111, projection='3d')
    sc = ax3d.scatter(df['midx'], df['midy'], df['midz'], c=df['cluster'], cmap='tab10', s=25, alpha=0.7)
    ax3d.set_xlabel('Easting [m] (midx)')
    ax3d.set_ylabel('Northing [m] (midy)')
    ax3d.set_zlabel('Elevation [m] (midz)')
    ax3d.set_title(f'Distribución 3D de clusters {titulo_suffix}')
    plt.colorbar(sc, ax=ax3d, shrink=0.6, label='Cluster')
    plt.tight_layout()
    plt.show()

    # Proyecciones 2D
    fig2d, axes2d = plt.subplots(1, 3, figsize=(18, 6))
    axes2d[0].scatter(df['midx'], df['midy'], c=df['cluster'], cmap='tab10', s=10, alpha=0.7)
    axes2d[0].set_xlabel('midx'); axes2d[0].set_ylabel('midy'); axes2d[0].set_title('XY')
    axes2d[1].scatter(df['midy'], df['midz'], c=df['cluster'], cmap='tab10', s=10, alpha=0.7)
    axes2d[1].set_xlabel('midy'); axes2d[1].set_ylabel('midz'); axes2d[1].set_title('YZ')
    axes2d[2].scatter(df['midx'], df['midz'], c=df['cluster'], cmap='tab10', s=10, alpha=0.7)
    axes2d[2].set_xlabel('midx'); axes2d[2].set_ylabel('midz'); axes2d[2].set_title('XZ')
    for ax in axes2d:
        ax.grid(True, alpha=0.3)
    fig2d.suptitle(f'Proyecciones 2D {titulo_suffix}', fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4) Deriva
# ---------------------------------------------------------------------------

def calcular_deriva_bins(
    df: pd.DataFrame,
    coord: str,
    variable: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Promedio de la variable por intervalos de la coordenada."""
    df_temp = df[[coord, variable]].copy()
    df_temp['bin'] = pd.cut(df_temp[coord], bins=n_bins)
    deriva = df_temp.groupby('bin', observed=True)[variable].agg(['mean', 'count'])
    deriva['coord_center'] = deriva.index.map(lambda x: x.mid)
    return deriva.reset_index()


def graficar_deriva(
    df: pd.DataFrame,
    cluster_id: int | None = None,
    variable: str = 'cut',
    n_bins: int = 20,
) -> None:
    """
    Gráficos de deriva (promedio de variable por eje) en X, Y, Z.
    Si cluster_id no es None, filtra por ese cluster antes de calcular.
    """
    if cluster_id is not None:
        df_plot = df[df['cluster'] == cluster_id].copy()
        tit = f' (cluster {cluster_id})'
    else:
        df_plot = df
        tit = ''

    deriva_x = calcular_deriva_bins(df_plot, 'midx', variable, n_bins=n_bins)
    deriva_y = calcular_deriva_bins(df_plot, 'midy', variable, n_bins=n_bins)
    deriva_z = calcular_deriva_bins(df_plot, 'midz', variable, n_bins=n_bins)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].plot(deriva_x['coord_center'], deriva_x['mean'], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Easting [m]'); axes[0].set_ylabel(f'{variable} (promedio)'); axes[0].set_title(f'Deriva X{tit}')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(deriva_y['coord_center'], deriva_y['mean'], 'o-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Northing [m]'); axes[1].set_ylabel(f'{variable} (promedio)'); axes[1].set_title(f'Deriva Y{tit}')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(deriva_z['coord_center'], deriva_z['mean'], 'o-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Elevation [m]'); axes[2].set_ylabel(f'{variable} (promedio)'); axes[2].set_title(f'Deriva Z{tit}')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5) Guardado
# ---------------------------------------------------------------------------

def guardar_resultados_cluster(
    df: pd.DataFrame,
    metodo: str,
    output_dir: str = 'data/processed/cluster',
) -> str:
    """
    Guarda el DataFrame con clusters en Excel.
    metodo: 'kmeans' o 'dbscan' (se usa en el nombre del archivo).

    Returns:
        Ruta del archivo guardado.
    """
    os.makedirs(output_dir, exist_ok=True)
    fecha_hora = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre = f'clusters_{metodo}_{fecha_hora}.xlsx'
    ruta = os.path.join(output_dir, nombre)
    try:
        df.to_excel(ruta, index=False)
    except Exception:
        ruta = os.path.join(output_dir, f'clusters_{metodo}_{fecha_hora}.csv')
        df.to_csv(ruta, index=False)
        print('(openpyxl no disponible; guardado como CSV)')
    print(f'DataFrame ({metodo}) guardado en: {ruta}')
    return ruta


# ---------------------------------------------------------------------------
# 6) Flujo principal: datos que vos des
# ---------------------------------------------------------------------------

def run_analisis_completo(
    ruta_datos: str = 'data/processed/df_cut.xlsx',
    n_clusters_kmeans: int = 4,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    variable_ley: str = 'cut',
    variable_nscore: str = 'cut_nscore',
    cluster_para_deriva: int | None = 3,
    n_bins_deriva: int = 20,
    guardar_excel: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga datos, corre KMeans y DBSCAN, aplica el mismo análisis a ambos,
    opcionalmente grafica deriva para un cluster y guarda ambos en Excel.

    Returns:
        (df_kmeans, df_dbscan)
    """
    df = cargar_datos(ruta_datos)
    print('Columnas:', df.columns.tolist())

    # ---- KMeans ----
    df_kmeans, _ = clustering_kmeans(df, n_clusters=n_clusters_kmeans, var_extra=variable_nscore)
    print('\n--- Resultados KMeans ---')
    analisis_por_cluster(df_kmeans, variable_ley=variable_ley, variable_nscore=variable_nscore, titulo_suffix='KMeans')

    if cluster_para_deriva is not None and (df_kmeans['cluster'] == cluster_para_deriva).any():
        graficar_deriva(df_kmeans, cluster_id=cluster_para_deriva, variable=variable_ley, n_bins=n_bins_deriva)

    if guardar_excel:
        guardar_resultados_cluster(df_kmeans, 'kmeans')

    # ---- DBSCAN ----
    df_dbscan, _ = clustering_dbscan(df, eps=dbscan_eps, min_samples=dbscan_min_samples)
    n_ruido = (df_dbscan['cluster'] == -1).sum()
    print(f'\n--- Resultados DBSCAN (ruido: {n_ruido} puntos) ---')
    analisis_por_cluster(df_dbscan, variable_ley=variable_ley, variable_nscore=variable_nscore, titulo_suffix='DBSCAN')

    if cluster_para_deriva is not None and (df_dbscan['cluster'] == cluster_para_deriva).any():
        graficar_deriva(df_dbscan, cluster_id=cluster_para_deriva, variable=variable_ley, n_bins=n_bins_deriva)

    if guardar_excel:
        guardar_resultados_cluster(df_dbscan, 'dbscan')

    return df_kmeans, df_dbscan


# %% --- Ejecución con los datos que quieras ---

if __name__ == '__main__' or True:
    df_kmeans, df_dbscan = run_analisis_completo(
        ruta_datos='data/processed/df_cut.xlsx',
        n_clusters_kmeans=4,
        dbscan_eps=1,
        dbscan_min_samples=30,
        variable_ley='cut',
        variable_nscore='cut_nscore',
        cluster_para_deriva=3,
        n_bins_deriva=20,
        guardar_excel=True,
    )

# %%
