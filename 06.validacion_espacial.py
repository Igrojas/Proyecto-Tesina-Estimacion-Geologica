# %%
"""
06. Validación Espacial: Filtro de Validación Cercana a Originales

El objetivo es:
1. Tomar cada punto Original (muestras reales, ~1000 datos).
2. Buscar, para cada punto Original, cuál es el único punto de Validación 
   (empresa) que está más cercano a él.
3. Extraer solo esos puntos de validación (descartando el resto).
4. Unir la información Original vs Validación Seleccionada para comparar
   real vs real vs Machine Learning.
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score

from config_figuras_tesina import setup_figuras_tesina, set_proportional_aspect

warnings.filterwarnings("ignore")

# --- Configuración de rutas ---
INPUT_ORIG_PATH = Path("data/processed/cluster/clusters_df_con_nscore.csv")
INPUT_VAL_PATH = Path("data/processed/df_validacion_predicciones.csv")
OUTPUT_DIR = Path("data/validacion_final")
OUTPUT_CSV_PATH = OUTPUT_DIR / "06_cruce_validacion_vs_original.csv"
OUTPUT_EXCEL_PATH = OUTPUT_DIR / "06_cruce_validacion_vs_original.xlsx"
IMAGENES_DIR = Path("imagenes")

# --- Columnas ---
COORD_COLS = ["Este", "Norte", "Cota"]
COORD_LABELS = ["Este (m)", "Norte (m)", "Cota (m)"]
ORIG_TARGET = "Rec_Peso_PND25_(%)"

def plot_3d_superpuesto_filtrado(coords_orig: np.ndarray, coords_val_filtrado: np.ndarray, save_path: Path | None = None) -> None:
    """Gráfico 3D Originales vs SOLO las Validaciones Más Cercanas."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Originales
    ax.scatter(coords_orig[:, 0], coords_orig[:, 1], coords_orig[:, 2],
               c="#2563eb", s=40, alpha=0.9, label="Puntos Originales", edgecolor="white", linewidth=0.2)
    # Validaciones Seleccionadas
    ax.scatter(coords_val_filtrado[:, 0], coords_val_filtrado[:, 1], coords_val_filtrado[:, 2],
               c="#ef4444", s=15, alpha=0.6, label="Validación (Más cercana a c/original)")
    
    ax.set_xlabel(COORD_LABELS[0])
    ax.set_ylabel(COORD_LABELS[1])
    ax.set_zlabel(COORD_LABELS[2])
    ax.set_title("Nubes de Puntos (Pareo 1 a 1 Original -> Validación Cercana)")
    
    # Aspecto proporcional
    df_temp = pd.DataFrame(coords_orig, columns=COORD_COLS)
    set_proportional_aspect(ax, df_temp, COORD_COLS[0], COORD_COLS[1], COORD_COLS[2])
    
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Calcula R2, RMSE y Error Porcentual (MAPE)."""
    # Evitar divisiones por cero en el error porcentual
    mask = y_real != 0
    y_r = y_real[mask]
    y_p = y_pred[mask]
    
    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = np.mean(np.abs((y_r - y_p) / y_r)) * 100 if len(y_r) > 0 else np.nan
    
    return r2, rmse, mape

def plot_real_vs_pred_modelos(df_cruce: pd.DataFrame, save_dir: Path) -> None:
    """Gráfico separado para Empresa, y subplot 2x2 para Modelos ML."""
    y_real = df_cruce["Orig_Real_RecPeso(%)"].values
    metricas_list = []
    
    # 1. Gráfico Individual: Original vs Empresa
    col_empresa = "Val_Real_Empresa_recpe(%)"
    if col_empresa in df_cruce.columns:
        y_emp = df_cruce[col_empresa].values
        r2_e, rmse_e, mape_e = calcular_metricas(y_real, y_emp)
        metricas_list.append({"Modelo/Fuente": "Validación Empresa", "R2": r2_e, "RMSE": rmse_e, "Error Porcentual (%)": mape_e})
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_real, y_emp, alpha=0.5, c="#ef4444", s=20, edgecolor="white", linewidth=0.2)
        mini = min(y_real.min(), y_emp.min())
        maxi = max(y_real.max(), y_emp.max())
        ax.plot([mini, maxi], [mini, maxi], "k--", label="1:1")
        
        ax.set_title(f"Muestra Original vs Validación Empresa\nR²={r2_e:.3f} | RMSE={rmse_e:.2f} | Err%={mape_e:.1f}%", fontsize=11)
        ax.set_xlabel("Muestra Original (%)")
        ax.set_ylabel("Validación Empresa (%)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "06_cruce_comparacion_pareada_empresa.png")
        plt.show()

    # 2. Gráfico 2x2: Original vs ML
    modelos_ml = [
        ("KNN", "pred_real_knn", "#2563eb"),
        ("XGBoost", "pred_real_xgb", "#059669"),
        ("SVR", "pred_real_svr", "#7c3aed"),
        ("MLP", "pred_real_mlp", "#d97706")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes_flat = axes.flatten()
    
    for ax, (nombre, col, color) in zip(axes_flat, modelos_ml):
        if col in df_cruce.columns:
            y_pred = df_cruce[col].values
            r2, rmse, mape = calcular_metricas(y_real, y_pred)
            metricas_list.append({"Modelo/Fuente": nombre, "R2": r2, "RMSE": rmse, "Error Porcentual (%)": mape})
            
            ax.scatter(y_real, y_pred, alpha=0.5, c=color, s=15, edgecolor="white", linewidth=0.2)
            mini = min(y_real.min(), y_pred.min())
            maxi = max(y_real.max(), y_pred.max())
            ax.plot([mini, maxi], [mini, maxi], "k--", label="1:1")
            
            ax.set_title(f"Muestra Original vs {nombre}\nR²={r2:.3f} | RMSE={rmse:.2f} | Err%={mape:.1f}%", fontsize=10)
            ax.set_xlabel("Muestra Original (%)")
            ax.set_ylabel(f"{nombre} (%)")
            ax.grid(True, alpha=0.3)
            
    plt.tight_layout()
    plt.savefig(save_dir / "06_cruce_comparacion_pareada_modelosML.png")
    plt.show()
    
    # Imprimir tabla de métricas en consola
    df_metricas = pd.DataFrame(metricas_list)
    print("\n" + "="*60)
    print("MÉTRICAS: PUNTOS ORIGINALES VS PUNTOS DE VALIDACIÓN PAREADOS")
    print("="*60)
    print(df_metricas.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*60 + "\n")

# --- Ejecución ---
if __name__ == "__main__":
    setup_figuras_tesina()
    
    if not INPUT_ORIG_PATH.exists() or not INPUT_VAL_PATH.exists():
        raise FileNotFoundError("Error al leer archivos preprocesados.")
        
    df_orig = pd.read_csv(INPUT_ORIG_PATH)
    df_val = pd.read_csv(INPUT_VAL_PATH)
    
    coords_orig = df_orig[COORD_COLS].values
    coords_val = df_val[COORD_COLS].values
    
    print(f"Buscando pareo exacto para los {len(df_orig)} puntos originales...")
    
    # 1. Des-transformar variables de ML a unidades reales (%) 
    target_transformer = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)
    target_transformer.fit(df_orig[ORIG_TARGET].values.reshape(-1, 1))

    for m in ["knn", "xgb", "svr", "mlp"]:
        if f"pred_nscore_{m}" in df_val.columns:
            nscore_preds = df_val[f"pred_nscore_{m}"].values.reshape(-1, 1)
            real_preds = target_transformer.inverse_transform(nscore_preds)
            df_val[f"pred_real_{m}"] = real_preds.flatten()
            
    # 2. CRUCE ESPACIAL (BÚSQUEDA VECINO DESDE ORIGINAL -> VALIDACIÓN)
    # Se construye el KDTree usando las posiciones geométricas de VALIDACIÓN
    tree = KDTree(coords_val)
    
    # Para cada coordenada ORIGINAL, buscamos y extraemos su punto de VALIDACIÓN más cercano
    distancias, indices = tree.query(coords_orig)
    
    # Extraemos solo el subset de Validación que está pegado a los Originales
    df_val_pareado = df_val.iloc[indices].copy()
    df_val_pareado.reset_index(drop=True, inplace=True)
    coords_val_filtrado = df_val_pareado[COORD_COLS].values
    
    # 3. Graficar en 3D (Solo Originales vs los poquitos Validados pegados)
    IMAGENES_DIR.mkdir(parents=True, exist_ok=True)
    plot_3d_superpuesto_filtrado(coords_orig, coords_val_filtrado, save_path=IMAGENES_DIR / "06_cruce_pareo_3d.png")

    # 4. Consolidar la Tabla Final
    df_cruce = df_orig.copy()
    
    # Renombrar originales
    df_cruce.rename(columns={
        "Este": "Orig_Este",
        "Norte": "Orig_Norte",
        "Cota": "Orig_Cota",
        ORIG_TARGET: "Orig_Real_RecPeso(%)"
    }, inplace=True)
    
    # Acoplar las columnas encontradas de Validación y ML
    df_cruce["Val_Este"] = df_val_pareado["Este"]
    df_cruce["Val_Norte"] = df_val_pareado["Norte"]
    df_cruce["Val_Cota"] = df_val_pareado["Cota"]
    df_cruce["Val_Real_Empresa_recpe(%)"] = df_val_pareado["recpe"]
    
    for m in ["knn", "xgb", "svr", "mlp"]:
        if f"pred_real_{m}" in df_val_pareado.columns:
            df_cruce[f"pred_real_{m}"] = df_val_pareado[f"pred_real_{m}"]

    # 5. Exportar y Graficar Resumen
    column_order = (
        ["Orig_Este", "Orig_Norte", "Orig_Cota", "Orig_Real_RecPeso(%)"] + 
        ["Val_Este", "Val_Norte", "Val_Cota", "Val_Real_Empresa_recpe(%)"] +
        [f"pred_real_{m}" for m in ["knn", "xgb", "svr", "mlp"]]
    )
    
    # Ordenamos dejando lo importante primero
    other_cols = [c for c in df_cruce.columns if c not in column_order]
    df_cruce = df_cruce[column_order + other_cols]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_cruce.to_csv(OUTPUT_CSV_PATH, index=False)
    df_cruce.to_excel(OUTPUT_EXCEL_PATH, index=False)
    
    print(f"\n¡Listo! Generados {len(df_cruce)} cruces geológicos (1 pareo por cada muestra original).")
    print(f"Archivo exportado a:\n -> {OUTPUT_EXCEL_PATH}")
    
    # Grafico final y métricas (separado en 2 figuras)
    plot_real_vs_pred_modelos(df_cruce, save_dir=IMAGENES_DIR)

# %%
