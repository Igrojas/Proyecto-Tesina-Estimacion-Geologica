# %%
"""
Predicción de Rec_Peso_PND25_(%)_nscore con Optimización Bayesiana (Optuna).
Features: Este, Norte, Cota (estandarizadas) + get_dummies(cluster_con_nscore).
Modelos: KNN, XGBoost, SVR, MLPRegressor.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from config_figuras_tesina import setup_figuras_tesina

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuración ---
INPUT_PATH = Path("data/processed/cluster/clusters_df_con_nscore.csv")
INPUT_COMBINED_PATH = Path("data/processed/cluster/puntos_originales_y_validacion.csv")
OUTPUT_DF_VAL_PATH = Path("data/processed/df_validacion_predicciones.csv")
OUTPUT_METRICAS_ML_PATH = Path("data/processed/metricas_ml.csv")
IMAGENES_DIR = Path("imagenes")
COORD_COLS = ["Este", "Norte", "Cota"]
COORD_LABELS = ["Este (m)", "Norte (m)", "Cota (m)"]
CLUSTER_COL = "cluster_con_nscore"
TARGET_COL = "Rec_Peso_PND25_(%)_nscore"
TARGET_LABEL = "Recuperación en peso (%) (nscore)"
ORIGEN_COL = "origen"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_TRIALS = 30  # Iteraciones de Optuna por modelo


def build_X(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    dummy_cols_order: list[str] | None = None,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Coords estandarizadas + get_dummies(cluster). Si scaler/dummy_cols_order se pasan, transforma (mismo orden)."""
    coords = df[COORD_COLS].values
    if scaler is None:
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
    else:
        coords_scaled = scaler.transform(coords)
    dummies = pd.get_dummies(df[CLUSTER_COL], prefix="cluster", dtype=float)
    if dummy_cols_order is not None:
        for c in dummy_cols_order:
            if c not in dummies.columns:
                dummies[c] = 0.0
        dummies = dummies[dummy_cols_order]
    dummy_cols = list(dummies.columns)
    X = np.hstack([coords_scaled, dummies.values])
    return X, scaler, dummy_cols


def plot_real_vs_pred(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    ax: plt.Axes,
) -> plt.Axes:
    """Gráfico real vs predicho (scatter 1:1)."""
    ax.scatter(y_real, y_pred, alpha=0.6, s=20, color="#2563eb", edgecolors="white", linewidths=0.3)
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="y = x")
    ax.set_xlabel(f"Real")
    ax.set_ylabel(f"Predicho")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.35)
    return ax


# --- Funciones Objetivo de Optuna ---
def objective_knn(trial, X, y):
    n_neighbors = trial.suggest_int("n_neighbors", 2, 20)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 2)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=-1)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmse_scores)

def objective_xgb(trial, X, y):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 9)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                             learning_rate=learning_rate, subsample=subsample, 
                             random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmse_scores)

def objective_svr(trial, X, y):
    C = trial.suggest_float("C", 0.1, 100, log=True)
    epsilon = trial.suggest_float("epsilon", 0.01, 1, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = SVR(C=C, epsilon=epsilon, gamma=gamma)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmse_scores)

def objective_mlp(trial, X, y):
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50)])
    learning_rate_init = trial.suggest_float("learning_rate_init", 0.001, 0.1, log=True)
    alpha = trial.suggest_float("alpha", 0.0001, 0.1, log=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                             learning_rate_init=learning_rate_init, alpha=alpha, 
                             max_iter=500, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmse_scores)

# --- Ejecución ---
# %%
if __name__ == "__main__":
    setup_figuras_tesina()
    IMAGENES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    X, scaler, dummy_cols = build_X(df)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\n--- Iniciando Optimización Bayesiana ({N_TRIALS} trials por modelo) ---")
    
    # 1. KNN
    print("Optimizando KNN...")
    study_knn = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_knn.optimize(lambda trial: objective_knn(trial, X_train, y_train), n_trials=N_TRIALS)
    print(f"Mejores parámetros KNN: {study_knn.best_params}")
    knn_best = KNeighborsRegressor(**study_knn.best_params, n_jobs=-1)
    knn_best.fit(X_train, y_train)

    # 2. XGBoost
    print("Optimizando XGBoost...")
    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=N_TRIALS)
    print(f"Mejores parámetros XGBoost: {study_xgb.best_params}")
    xgb_best = XGBRegressor(**study_xgb.best_params, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_best.fit(X_train, y_train)

    # 3. SVR
    print("Optimizando SVR...")
    study_svr = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_svr.optimize(lambda trial: objective_svr(trial, X_train, y_train), n_trials=N_TRIALS)
    print(f"Mejores parámetros SVR: {study_svr.best_params}")
    svr_best = SVR(**study_svr.best_params)
    svr_best.fit(X_train, y_train)

    # 4. MLP
    print("Optimizando MLPRegressor...")
    study_mlp = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_mlp.optimize(lambda trial: objective_mlp(trial, X_train, y_train), n_trials=N_TRIALS)
    print(f"Mejores parámetros MLP: {study_mlp.best_params}")
    mlp_best = MLPRegressor(**study_mlp.best_params, max_iter=500, random_state=RANDOM_STATE)
    mlp_best.fit(X_train, y_train)

    # --- Evaluaciones Train y Test ---
    modelos = {
        "KNN": knn_best,
        "XGBoost": xgb_best,
        "SVR": svr_best,
        "MLP": mlp_best
    }

    resultados_metricas = []
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Real vs Predicho - Entrenamiento y Testeo\n{TARGET_LABEL}", fontsize=16)

    print("\n--- Resultados Finales ---")
    for idx, (nombre_modelo, modelo) in enumerate(modelos.items()):
        # Train
        y_pred_train = modelo.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        # Test
        y_pred_test = modelo.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        resultados_metricas.append({
            "modelo": nombre_modelo,
            "n_train": len(y_train),
            "R2_train": r2_train,
            "RMSE_train": rmse_train,
            "n_test": len(y_test),
            "R2_test": r2_test,
            "RMSE_test": rmse_test
        })

        print(f"{nombre_modelo}:")
        print(f"  Train -> R²: {r2_train:.4f} | RMSE: {rmse_train:.4f}")
        print(f"  Test  -> R²: {r2_test:.4f} | RMSE: {rmse_test:.4f}")

        # Gráficos
        plot_real_vs_pred(y_train.values, y_pred_train, f"{nombre_modelo} (Train)", ax=axes[0, idx])
        plot_real_vs_pred(y_test.values, y_pred_test, f"{nombre_modelo} (Test)", ax=axes[1, idx])

    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Dar espacio al suptitle
    plt.savefig(IMAGENES_DIR / "train_ml_real_vs_predicho_optimizado.png")
    # plt.show() # comentado para que no bloquee la consola durante el test.

    # Guardar métricas
    df_metricas = pd.DataFrame(resultados_metricas)
    OUTPUT_METRICAS_ML_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_metricas.to_csv(OUTPUT_METRICAS_ML_PATH, index=False)
    print(f"\nMétricas guardadas en: {OUTPUT_METRICAS_ML_PATH}")

    # --- Predicción sobre puntos de VALIDACIÓN ---
    if not INPUT_COMBINED_PATH.exists():
        print(f"AVISO: No se encontró {INPUT_COMBINED_PATH}. Ejecuta primero 03.estimacion.py.")
    else:
        df_full = pd.read_csv(INPUT_COMBINED_PATH)
        df_val = df_full[df_full[ORIGEN_COL] == "validacion"].copy()
        
        if df_val.empty:
            print("ERROR: No se encontraron puntos con origen 'validacion' en el archivo combinado.")
        else:
            X_val, _, _ = build_X(df_val, scaler, dummy_cols)

            df_val["pred_nscore_knn"] = knn_best.predict(X_val)
            df_val["pred_nscore_xgb"] = xgb_best.predict(X_val)
            df_val["pred_nscore_svr"] = svr_best.predict(X_val)
            df_val["pred_nscore_mlp"] = mlp_best.predict(X_val)

            OUTPUT_DF_VAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            df_val.to_csv(OUTPUT_DF_VAL_PATH, index=False)
            print(f"DataFrame con predicciones de validación guardado en: {OUTPUT_DF_VAL_PATH}")

# %%
