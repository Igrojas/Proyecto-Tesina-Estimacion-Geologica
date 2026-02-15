#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, NearestNeighbors
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

df_cluster = pd.read_csv('data/processed/cluster/clusters_df_20260201_162519.csv')
# Si hay más de 10 000 filas, toma un muestreo representativo (estratificado por 'cluster')
# if len(df_cluster) > 10000:
#     df_cluster = df_cluster.groupby('cluster', group_keys=False).apply(
#         lambda x: x.sample(int(10000/df_cluster['cluster'].nunique()), random_state=42) 
#         if len(x) >= int(10000/df_cluster['cluster'].nunique()) 
#         else x
#     )
#     df_cluster = df_cluster.sample(10000, random_state=42) if len(df_cluster) > 10000 else df_cluster
#     df_cluster = df_cluster.reset_index(drop=True)

# %%
df_cluster.describe()
#%%

print("Aplicamos el modelo KNN para predecir el cut_nscore")

# Aplica get_dummies (dummies) a la columna 'cluster'
X_cluster_dummies = pd.get_dummies(df_cluster['cluster'], prefix='cluster', dtype=float)

# Estandariza las features
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(df_cluster[['midx', 'midy', 'midz']]),
    columns=['midx', 'midy', 'midz'],
    index=df_cluster.index
)

# Combina las dummies con las coordenadas escaladas
X_full = pd.concat([X_cluster_dummies, X_scaled], axis=1)
y = df_cluster['cut_nscore']

# Split en train/test (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.3)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")


sns.scatterplot(x=y_test, y=y_pred, color='b', edgecolor='k')
# Dibuja la línea 1:1 para referencia
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r-', label='Línea 1:1 (y=x)')
plt.xlabel('Valor real')
plt.ylabel('Valor predicho')
plt.title('Scatter plot de valores reales vs predichos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Convex hull → puntos nuevos → KNN asigna cluster → graficar

from scipy.spatial import ConvexHull, Delaunay
import itertools

# 1. Convex hull en (midx, midy)
points_2d = df_cluster[['midx', 'midy']].values
if len(points_2d) < 3:
    raise ValueError("Se necesitan al menos 3 puntos para el convex hull.")
hull = ConvexHull(points_2d)

# 2. Grilla dentro del hull (puntos nuevos para estimar)
PASO_M = 5
grid_x = np.arange(points_2d[:, 0].min(), points_2d[:, 0].max() + PASO_M, PASO_M)
grid_y = np.arange(points_2d[:, 1].min(), points_2d[:, 1].max() + PASO_M, PASO_M)
grid_points = np.array(list(itertools.product(grid_x, grid_y)))

hull_delaunay = Delaunay(points_2d[hull.vertices])
mask_inside = hull_delaunay.find_simplex(grid_points) >= 0
puntos_dentro = grid_points[mask_inside]

# Quitar puntos que ya existen en df_cluster
tol = 1e-6
def is_duplicate(pt: np.ndarray, df: pd.DataFrame) -> bool:
    return ((np.abs(df['midx'] - pt[0]) < tol) & (np.abs(df['midy'] - pt[1]) < tol)).any()

puntos_nuevos = np.array([pt for pt in puntos_dentro if not is_duplicate(pt, df_cluster)])

if len(puntos_nuevos) == 0:
    df_nuevos = pd.DataFrame(columns=['midx', 'midy', 'midz'])
else:
    df_nuevos = pd.DataFrame(puntos_nuevos, columns=['midx', 'midy'])
    # Asignar midz por vecino más cercano en (midx, midy): cada punto nuevo tiene (x, y, z) como los existentes
    nn_xy = NearestNeighbors(n_neighbors=1).fit(df_cluster[['midx', 'midy']])
    _, idx = nn_xy.kneighbors(df_nuevos[['midx', 'midy']])
    df_nuevos['midz'] = df_cluster['midz'].iloc[idx.flatten()].values

# 3. Asignar cluster a cada punto nuevo con KNN
if len(df_nuevos) > 0:
    scaler_h = StandardScaler()
    X_train = scaler_h.fit_transform(df_cluster[['midx', 'midy', 'midz']])
    y_train = df_cluster['cluster']
    clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
    X_nuevos = scaler_h.transform(df_nuevos[['midx', 'midy', 'midz']])
    df_nuevos['cluster'] = clf.predict(X_nuevos)

# 4. Graficar: existentes y nuevos por cluster (2D: midx, midy)
plt.figure(figsize=(8, 6))
plt.scatter(df_cluster['midx'], df_cluster['midy'], c=df_cluster['cluster'], cmap='tab10', s=25, alpha=0.9, label='Existentes', edgecolors='k', linewidths=0.3)
if len(df_nuevos) > 0:
    plt.scatter(df_nuevos['midx'], df_nuevos['midy'], c=df_nuevos['cluster'], cmap='tab10', s=8, alpha=0.6, label='Nuevos (cluster KNN)', marker='s')
plt.xlabel('Easting [m] (midx)')
plt.ylabel('Northing [m] (midy)')
plt.title('Convex hull: puntos existentes y nuevos coloreados por cluster (xy)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Graficar en 3D (midx, midy, midz) para ver que los puntos nuevos tienen z
if len(df_nuevos) > 0:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_cluster['midx'], df_cluster['midy'], df_cluster['midz'], c=df_cluster['cluster'], cmap='tab10', s=15, alpha=0.5, label='Existentes')
    ax.scatter(df_nuevos['midx'], df_nuevos['midy'], df_nuevos['midz'], c=df_nuevos['cluster'], cmap='tab10', s=8, alpha=0.8, label='Nuevos', marker='s')
    ax.set_xlabel('midx [m]')
    ax.set_ylabel('midy [m]')
    ax.set_zlabel('midz [m]')
    ax.set_title('Puntos en 3D (midx, midy, midz) por cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()


print("Aplicamos el modelo KNN para los puntos nuevos")

df_nuevos["cut_nscore"] = np.nan
# Aplica get_dummies (dummies) a la columna 'cluster'
X_cluster_dummies = pd.get_dummies(df_nuevos['cluster'], prefix='cluster', dtype=float)

# Estandariza las features
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(df_nuevos[['midx', 'midy', 'midz']]),
    columns=['midx', 'midy', 'midz'],
    index=df_nuevos.index
)

# Combina las dummies con las coordenadas escaladas
X_full = pd.concat([X_cluster_dummies, X_scaled], axis=1)

y_pred = knn.predict(X_full)
df_nuevos["cut_nscore"] = y_pred


# Comparación estadística por cluster entre df_cluster y df_nuevos

# Calculamos los estadísticos descriptivos por cluster en ambos dataframes
clusters_comunes = np.intersect1d(df_cluster['cluster'].unique(), df_nuevos['cluster'].unique())

resultados = []

for cl in clusters_comunes:
    stats_exist = df_cluster[df_cluster['cluster'] == cl]['cut_nscore'].describe()
    stats_nuevos = df_nuevos[df_nuevos['cluster'] == cl]['cut_nscore'].describe()
    resultados.append({
        'cluster': cl,
        'n_exist': (df_cluster['cluster'] == cl).sum(),
        'media_exist': stats_exist['mean'],
        'std_exist': stats_exist['std'],
        'min_exist': stats_exist['min'],
        'max_exist': stats_exist['max'],
        'n_nuevos': (df_nuevos['cluster'] == cl).sum(),
        'media_nuevos': stats_nuevos['mean'],
        'std_nuevos': stats_nuevos['std'],
        'min_nuevos': stats_nuevos['min'],
        'max_nuevos': stats_nuevos['max'],
    })

df_stats = pd.DataFrame(resultados)

# Mostramos la tabla comparativa por cluster
print("Comparación estadística de cut_nscore por cluster:\n")
print(df_stats.to_string(index=False, float_format='%.2f'))

# Adicional: Gráfica comparativa de medias y cajas por cluster
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.plot(df_stats['cluster'], df_stats['media_exist'], 'o-', label='Existentes')
plt.plot(df_stats['cluster'], df_stats['media_nuevos'], 's-', label='Nuevos')
plt.title("Media de cut_nscore por cluster")
plt.xlabel("Cluster")
plt.ylabel("Media cut_nscore")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1,2,2)
# Cajas para cada cluster: existentes y nuevos
data_exist = [df_cluster[df_cluster['cluster']==cl]['cut_nscore'] for cl in clusters_comunes]
data_nuevos = [df_nuevos[df_nuevos['cluster']==cl]['cut_nscore'] for cl in clusters_comunes]
plt.boxplot(data_exist, positions=np.arange(len(clusters_comunes))-0.15, widths=0.25, patch_artist=True, labels=clusters_comunes)
plt.boxplot(data_nuevos, positions=np.arange(len(clusters_comunes))+0.15, widths=0.25, patch_artist=True, labels=clusters_comunes)
plt.xticks(np.arange(len(clusters_comunes)), clusters_comunes)
plt.xlabel("Cluster")
plt.ylabel("cut_nscore")
plt.title("Boxplot de cut_nscore por cluster")
plt.legend(['Existentes', 'Nuevos'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Función para calcular deriva por bins
def calcular_deriva_bins(df, coord, variable, n_bins=10):
    """
    Calcula el promedio de la variable por intervalos de coordenada
    
    Args:
        df: DataFrame
        coord: nombre de la columna de coordenada ('midx', 'midy', 'midz')
        variable: nombre de la variable ('cut')
        n_bins: número de intervalos
    
    Returns:
        DataFrame con los promedios por intervalo
    """
    # Crear bins (intervalos)
    df_temp = df[[coord, variable]].copy()
    df_temp['bin'] = pd.cut(df_temp[coord], bins=n_bins)
    
    # Calcular promedio por bin
    deriva = df_temp.groupby('bin', observed=True)[variable].agg(['mean', 'count'])
    deriva['coord_center'] = deriva.index.map(lambda x: x.mid)
    
    return deriva.reset_index()

# Calcular deriva para cada eje para ambos: existentes y nuevos
n_bins = 20

# Deriva para puntos existentes
deriva_x_exist = calcular_deriva_bins(df_cluster, 'midx', 'cut_nscore', n_bins=n_bins)
deriva_y_exist = calcular_deriva_bins(df_cluster, 'midy', 'cut_nscore', n_bins=n_bins)
deriva_z_exist = calcular_deriva_bins(df_cluster, 'midz', 'cut_nscore', n_bins=n_bins)

# Deriva para puntos nuevos
deriva_x_nuevo = calcular_deriva_bins(df_nuevos, 'midx', 'cut_nscore', n_bins=n_bins)
deriva_y_nuevo = calcular_deriva_bins(df_nuevos, 'midy', 'cut_nscore', n_bins=n_bins)
deriva_z_nuevo = calcular_deriva_bins(df_nuevos, 'midz', 'cut_nscore', n_bins=n_bins)

# Graficar en un solo gráfico cada uno de los 3 ejes, comparando existentes y nuevos
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

# (a) Deriva en X
axes[0].plot(deriva_x_exist['coord_center'], deriva_x_exist['mean'], 'o-', linewidth=2, markersize=8, label='Existentes')
axes[0].plot(deriva_x_nuevo['coord_center'], deriva_x_nuevo['mean'], 's--', linewidth=2, markersize=8, label='Nuevos')
axes[0].set_xlabel('Easting [m] (coordenada)', fontsize=11)
axes[0].set_ylabel('Ley Cu [%] (Promedio)', fontsize=11)
axes[0].set_title('Deriva de Cu a lo largo de Easting [m]')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# (b) Deriva en Y
axes[1].plot(deriva_y_exist['coord_center'], deriva_y_exist['mean'], 'o-', linewidth=2, markersize=8, label='Existentes')
axes[1].plot(deriva_y_nuevo['coord_center'], deriva_y_nuevo['mean'], 's--', linewidth=2, markersize=8, label='Nuevos')
axes[1].set_xlabel('Northing [m] (coordenada)', fontsize=11)
axes[1].set_title('Deriva de Cu a lo largo de Northing [m]')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# (c) Deriva en Z
axes[2].plot(deriva_z_exist['coord_center'], deriva_z_exist['mean'], 'o-', linewidth=2, markersize=8, label='Existentes')
axes[2].plot(deriva_z_nuevo['coord_center'], deriva_z_nuevo['mean'], 's--', linewidth=2, markersize=8, label='Nuevos')
axes[2].set_xlabel('Elevation [m] (coordenada)', fontsize=11)
axes[2].set_title('Deriva de Cu a lo largo de Elevation [m]')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()
# %%

# Histograma comparativo de cut_nscore en existentes y nuevos

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
bins = 30

ax.hist(df_cluster['cut_nscore'].dropna(), bins=bins, alpha=0.7, label='Existentes', color='b', edgecolor='k', density=True)
ax.hist(df_nuevos['cut_nscore'].dropna(), bins=bins, alpha=0.7, label='Nuevos', color='orange', edgecolor='k', density=True)
ax.set_xlabel('cut_nscore')
ax.set_ylabel('Densidad')
ax.set_title('Histograma comparativo de cut_nscore\n(Existentes vs Nuevos)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#%%

print(df_cluster["midz"].min())
print(df_cluster["midz"].max())


