#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from IPython.display import display
except ImportError:
    display = print


def analizar_variable(
    df: pd.DataFrame,
    columna: str,
    crear_nscore: bool = True,
    n_quantiles: int = 1000,
    bins_hist: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Carga el df, analiza la variable indicada y opcionalmente crea su normal score.

    - Estadísticas descriptivas, nulos y gráficos según la variable ingresada.
    - Si crear_nscore=True, agrega la columna '{columna}_nscore' al dataframe.

    Returns:
        DataFrame con la columna _nscore agregada (si crear_nscore=True).
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe. Columnas: {df.columns.tolist()}")

    # 1. Análisis básico de la variable
    print(f"--- Análisis de la variable: {columna} ---")
    print("Datos faltantes / nulos:")
    display(df[columna].isnull().sum())
    print("\nEstadística descriptiva:")
    display(df[columna].describe())

    # 2. Histograma simple
    plt.figure(figsize=(6, 4))
    sns.histplot(df[columna].dropna(), bins=min(30, bins_hist), kde=True)
    plt.xlabel(columna)
    plt.ylabel("Frecuencia")
    plt.title(f"Histograma de {columna}")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    if not crear_nscore:
        return df

    # 3. Normal score (QuantileTransformer)
    col_nscore = f"{columna}_nscore"
    data_original = df[columna].values.reshape(-1, 1)
    transformer = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(df)),
        output_distribution="normal",
        random_state=random_state,
    )
    df = df.copy()
    df[col_nscore] = transformer.fit_transform(data_original)

    # 4. Comparación de estadísticas
    print("Estadísticas ORIGINALES:")
    print(f"  Media: {df[columna].mean():.4f}  Std: {df[columna].std():.4f}  Min: {df[columna].min():.4f}  Max: {df[columna].max():.4f}")
    print(f"Estadísticas TRANSFORMADAS ({col_nscore}):")
    print(f"  Media: {df[col_nscore].mean():.4f}  Std: {df[col_nscore].std():.4f}  Min: {df[col_nscore].min():.4f}  Max: {df[col_nscore].max():.4f}")

    # 5. Gráficos 2x2: histogramas y Q-Q
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(df[columna].dropna(), bins=bins_hist, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel(f"{columna} (original)")
    axes[0, 0].set_ylabel("Frecuencia")
    axes[0, 0].set_title("Distribución original")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(df[col_nscore], bins=bins_hist, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("Normal score")
    axes[0, 1].set_ylabel("Frecuencia")
    axes[0, 1].set_title("Distribución transformada (Gaussiana)")
    axes[0, 1].grid(True, alpha=0.3)

    stats.probplot(df[columna].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot original")
    axes[1, 0].grid(True, alpha=0.3)

    stats.probplot(df[col_nscore], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot transformado")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Variable: {columna}", y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"\n✓ Columna '{col_nscore}' agregada al dataframe.")
    return df


# --- Uso: cargar df, elegir columnas y analizar la variable que quieras ---
df = pd.read_excel("data/raw/com_p_plt_entry_1_guardado.xlsx", decimal=",")
print("Columnas disponibles:", df.columns.tolist())
#%%
list_cols = ["midx", "midy", "midz", "cut", "cus"]
df_cut = df[list_cols].copy()
df_cut["RL"] = df_cut["cus"] / df_cut["cut"]

# Eliminar solo los valores de 'cut' mayores a 15
df_cut = df_cut[df_cut["cut"] <= 15].copy()

# Analizar la variable que quieras (ej: 'cut'). Crea 'cut_nscore' y hace todos los análisis.
df_cut = analizar_variable(df_cut, "cut", crear_nscore=True)

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tu dataframe con: midx, midy, midz, cut

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

# Calcular deriva para cada eje
n_bins = 20

deriva_x = calcular_deriva_bins(df_cut, 'midx', 'cut', n_bins=n_bins)
deriva_y = calcular_deriva_bins(df_cut, 'midy', 'cut', n_bins=n_bins)
deriva_z = calcular_deriva_bins(df_cut, 'midz', 'cut', n_bins=n_bins)

# Graficar
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# (a) Deriva en X
axes[0].plot(deriva_x['coord_center'], deriva_x['mean'], 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Easting [m] (coordenada)', fontsize=11)
axes[0].set_ylabel('Ley Cu [%] (Promedio)', fontsize=11)
axes[0].set_title('Gráfico de Deriva para la Ley de Cu a lo largo de la dirección Easting [m]')
axes[0].grid(True, alpha=0.3)

# (b) Deriva en Y
axes[1].plot(deriva_y['coord_center'], deriva_y['mean'], 'o-', linewidth=2, markersize=8)
axes[1].set_xlabel('Northing [m] (coordenada)', fontsize=11)
axes[1].set_ylabel('Ley Cu [%] (Promedio)', fontsize=11)
axes[1].set_title('Gráfico de Deriva para la Ley de Cu a lo largo de la dirección Northing [m]')
axes[1].grid(True, alpha=0.3)

# (c) Deriva en Z
axes[2].plot(deriva_z['coord_center'], deriva_z['mean'], 'o-', linewidth=2, markersize=8)
axes[2].set_xlabel('Elevation [m] (coordenada)', fontsize=11)
axes[2].set_ylabel('Ley Cu [%] (Promedio)', fontsize=11)
axes[2].set_title('Gráfico de Deriva para la Ley de Cu a lo largo de la dirección Elevation [m]')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# # Imprimir estadísticas
# print("=" * 60)
# print("ANÁLISIS DE DERIVA")
# print("=" * 60)
# print("\nDeriva en X (Easting):")
# print(deriva_x[['coord_center', 'mean', 'count']])

# print("\nDeriva en Y (Northing):")
# print(deriva_y[['coord_center', 'mean', 'count']])

# print("\nDeriva en Z (Elevation):")
# print(deriva_z[['coord_center', 'mean', 'count']])
# %%

# Gráfico 3D con seaborn y 'cut' como cmap
from mpl_toolkits.mplot3d import Axes3D  # Comentado si ya está importado
import seaborn as sns

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(df_cut['midx'], df_cut['midy'], df_cut['midz'], 
                c=df_cut['cut'], cmap='RdYlBu', s=30)
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Elevation [m]')
ax.set_title('Distribución 3D con Cu (cut) como color')
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cb.set_label('Ley Cu [%] (cut)')
plt.show()

# Gráfico 3D con seaborn y 'cut_nscore' como cmap
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc2 = ax.scatter(df_cut['midx'], df_cut['midy'], df_cut['midz'], 
                 c=df_cut['cut_nscore'], cmap='RdYlBu', s=30)
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Elevation [m]')
ax.set_title('Distribución 3D con Cu Nscore (cut_nscore) como color')
cb2 = fig.colorbar(sc2, ax=ax, shrink=0.6, pad=0.1)
cb2.set_label('Ley Cu Nscore (cut_nscore)')
plt.show()

# %%
# Gráfico 1: Proyecciones 2D (xy, yz, xz) con cmap de variable 'cut'
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# xy (midx vs midy)
sc0 = axes[0].scatter(df_cut['midx'], df_cut['midy'], c=df_cut['cut'], cmap='RdYlBu', s=10, alpha=0.6)
axes[0].set_xlabel('Easting [m] (midx)')
axes[0].set_ylabel('Northing [m] (midy)')
axes[0].set_title('Proyección XY - coloreado por cut')
axes[0].grid(True, alpha=0.3)
plt.colorbar(sc0, ax=axes[0], label='Ley Cu [%] (cut)')

# yz (midy vs midz)
sc1 = axes[1].scatter(df_cut['midy'], df_cut['midz'], c=df_cut['cut'], cmap='RdYlBu', s=10, alpha=0.6)
axes[1].set_xlabel('Northing [m] (midy)')
axes[1].set_ylabel('Elevation [m] (midz)')
axes[1].set_title('Proyección YZ - coloreado por cut')
axes[1].grid(True, alpha=0.3)
plt.colorbar(sc1, ax=axes[1], label='Ley Cu [%] (cut)')

# xz (midx vs midz)
sc2 = axes[2].scatter(df_cut['midx'], df_cut['midz'], c=df_cut['cut'], cmap='RdYlBu', s=10, alpha=0.6)
axes[2].set_xlabel('Easting [m] (midx)')
axes[2].set_ylabel('Elevation [m] (midz)')
axes[2].set_title('Proyección XZ - coloreado por cut')
axes[2].grid(True, alpha=0.3)
plt.colorbar(sc2, ax=axes[2], label='Ley Cu [%] (cut)')

fig.suptitle('Proyecciones 2D coloreadas por cut (Ley Cu [%])', fontsize=14, weight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Gráfico 2: Proyecciones 2D (xy, yz, xz) con cmap de variable 'cut_nscore'
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# xy (midx vs midy)
sc0 = axes[0].scatter(df_cut['midx'], df_cut['midy'], c=df_cut['cut_nscore'], cmap='RdYlBu', s=10, alpha=0.6)
axes[0].set_xlabel('Easting [m] (midx)')
axes[0].set_ylabel('Northing [m] (midy)')
axes[0].set_title('Proyección XY - coloreado por cut_nscore')
axes[0].grid(True, alpha=0.3)
plt.colorbar(sc0, ax=axes[0], label='Ley Cu Nscore (cut_nscore)')

# yz (midy vs midz)
sc1 = axes[1].scatter(df_cut['midy'], df_cut['midz'], c=df_cut['cut_nscore'], cmap='RdYlBu', s=10, alpha=0.6)
axes[1].set_xlabel('Northing [m] (midy)')
axes[1].set_ylabel('Elevation [m] (midz)')
axes[1].set_title('Proyección YZ - coloreado por cut_nscore')
axes[1].grid(True, alpha=0.3)
plt.colorbar(sc1, ax=axes[1], label='Ley Cu Nscore (cut_nscore)')

# xz (midx vs midz)
sc2 = axes[2].scatter(df_cut['midx'], df_cut['midz'], c=df_cut['cut_nscore'], cmap='RdYlBu', s=10, alpha=0.6)
axes[2].set_xlabel('Easting [m] (midx)')
axes[2].set_ylabel('Elevation [m] (midz)')
axes[2].set_title('Proyección XZ - coloreado por cut_nscore')
axes[2].grid(True, alpha=0.3)
plt.colorbar(sc2, ax=axes[2], label='Ley Cu Nscore (cut_nscore)')

fig.suptitle('Proyecciones 2D coloreadas por cut_nscore (Normal Score)', fontsize=14, weight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %%

df_cut.to_excel('data/processed/df_cut_less_15.xlsx', index=False)


# %%
