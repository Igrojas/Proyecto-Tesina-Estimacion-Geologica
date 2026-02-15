#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings('ignore')

df = pd.read_excel('data/processed/df_cut.xlsx')

# Tomar un sample de 10000 filas de manera representativa (estratificada por deciles de cut_nscore)
# if len(df) > 10000:
#     # Crear deciles de la columna 'cut_nscore' para estratificar
#     df['cut_nscore_bin'] = pd.qcut(df['cut_nscore'], q=10, duplicates='drop')
#     df_sample = df.groupby('cut_nscore_bin', group_keys=False).apply(lambda x: x.sample(int(10000/10), random_state=42) if len(x) >= int(10000/10) else x)
#     df_sample = df_sample.sample(10000, random_state=42) if len(df_sample) > 10000 else df_sample
#     df = df_sample.drop(columns=['cut_nscore_bin'])

#%%

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.hist(df['cut_nscore'], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('cut_nscore')
plt.ylabel('Frecuencia')
plt.title('Histograma de cut_nscore')
plt.grid(True, alpha=0.3)
plt.show()
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

#%%
X = df[['midx', 'midy']]
y = df['cut_nscore']

# Aplica StandardScaler a X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

svm = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")


sns.scatterplot(x=y_test, y=y_pred)
# Dibuja la línea 1:1 para referencia
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea 1:1 (y=x)')
plt.xlabel('Valor real')
plt.ylabel('Valor predicho')
plt.title('Scatter plot de valores reales vs predichos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%

print("Aca le metemos interaccion entre las coordenadas x y")

df['midx*midy'] = df['midx'] * df['midy']
df['midx/midy'] = df['midx'] / df['midy']
df['midx**2'] = df['midx'] ** 2
df['midy**2'] = df['midy'] ** 2

X = df[['midx', 'midy', 'midx*midy', 'midx/midy','midx**2', 'midy**2']]
y = df['cut_nscore']

# Aplica StandardScaler a X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

svm = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")


sns.scatterplot(x=y_test, y=y_pred)
# Dibuja la línea 1:1 para referencia
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea 1:1 (y=x)')
plt.xlabel('Valor real')
plt.ylabel('Valor predicho')
plt.title('Scatter plot de valores reales vs predichos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
X = df[['midx', 'midy', 'midz']]
y = df['cut_nscore']

# Aplica StandardScaler a X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")

sns.scatterplot(x=y_test, y=y_pred, color='b', edgecolor='k')
# Dibuja la línea 1:1 para referencia
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea 1:1 (y=x)')
plt.xlabel('Valor real')
plt.ylabel('Valor predicho')
plt.title('Scatter plot de valores reales vs predichos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


#%%
print("Aca le metemos interaccion entre las coordenadas x y")

df['midx*midy'] = df['midx'] * df['midy']
df['midx/midy'] = df['midx'] / df['midy']
df['midx**2'] = df['midx'] ** 2
df['midy**2'] = df['midy'] ** 2

X = df[['midx', 'midy', 'midx*midy', 'midx/midy','midx**2', 'midy**2']]
y = df['cut_nscore']

# Aplica StandardScaler a X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")

sns.scatterplot(x=y_test, y=y_pred)
# Dibuja la línea 1:1 para referencia
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea 1:1 (y=x)')
plt.xlabel('Valor real')
plt.ylabel('Valor predicho')
plt.title('Scatter plot de valores reales vs predichos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
print("Aca le metemos XGBRegressor")
print("="*100)
print("Aca le metemos interaccion entre las coordenadas x y")

df['midx*midy'] = df['midx'] * df['midy']
df['midx/midy'] = df['midx'] / df['midy']
df['midx**2'] = df['midx'] ** 2
df['midy**2'] = df['midy'] ** 2

from sklearn.model_selection import cross_val_predict, KFold

# feature_names = ['midx', 'midy', 'midx*midy', 'midx/midy','midx**2', 'midy**2']
feature_names = ['midx', 'midy', 'midz']
X = df[feature_names]
y = df['cut_nscore']

# Aplica StandardScaler a X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configurar el modelo XGBoost
xgb = XGBRegressor(n_estimators=500, learning_rate=1, max_depth=10, random_state=42)

# === CROSS-VALIDATION ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(xgb, X_scaled, y, cv=kf)

mse_cv = mean_squared_error(y, y_pred_cv)
r2_cv = r2_score(y, y_pred_cv)
print(f"[CROSS-VAL] Mean Squared Error (CV): {mse_cv:.4f}")
print(f"[CROSS-VAL] R2 Score (CV): {r2_cv:.4f}")

# Scatter Plot: Cross-validation
plt.figure(figsize=(7,6))
sns.scatterplot(x=y, y=y_pred_cv, s=40, color='dodgerblue', alpha=0.7, edgecolor='k')
min_val = min(y.min(), y_pred_cv.min())
max_val = max(y.max(), y_pred_cv.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Línea 1:1 (y=x)')
plt.xlabel('Valor real', fontsize=13, fontweight='bold')
plt.ylabel('Valor predicho', fontsize=13, fontweight='bold')
plt.title('Scatter plot de valores reales vs predichos (CV)', fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.25, linestyle='--')
plt.tight_layout()
plt.show()

# ==== TEST SPLIT EVALUATION ==== 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
xgb.fit(X_train, y_train)
y_pred_test = xgb.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
mae_test = np.mean(np.abs(y_test - y_pred_test))
print("="*50)
print("[TEST SPLIT] Estadísticas del set de testeo:")
print(f"Mean Squared Error (Test): {mse_test:.4f}")
print(f"R2 Score (Test): {r2_test:.4f}")
print(f"Mean Absolute Error (Test): {mae_test:.4f}")
print(f"y_test mean: {np.mean(y_test):.2f} | y_test std: {np.std(y_test):.2f}")

plt.figure(figsize=(7,6))
sns.scatterplot(x=y_test, y=y_pred_test, s=40, color='green', alpha=0.7, edgecolor='k')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Línea 1:1 (y=x)')
plt.xlabel('Valor real', fontsize=13, fontweight='bold')
plt.ylabel('Valor predicho', fontsize=13, fontweight='bold')
plt.title('Scatter plot (test split)', fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==== Importancia de las variables (Feature Importance) ====
# Entrenar XGBoost sobre todos los datos para obtener importancias
xgb.fit(X_scaled, y)
importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,5))
bars = plt.bar(range(X.shape[1]), importances[indices], color=plt.cm.viridis(np.linspace(0.3, 0.7, X.shape[1])))
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=30, ha='right', fontsize=12)
plt.xlabel('Variables', fontsize=13, fontweight='bold')
plt.ylabel('Importancia', fontsize=13, fontweight='bold')
plt.title('Importancia de Variables\nXGBoost', fontsize=16, fontweight='bold')
plt.tight_layout()

# Añadir porcentajes arriba de las barras para más profesionalismo
for rect, imp in zip(bars, importances[indices]):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2., height + 0.01, f'{imp*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='semibold')

plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.25)
plt.show()
# %%
