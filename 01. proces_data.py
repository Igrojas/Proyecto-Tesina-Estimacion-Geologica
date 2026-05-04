# %%
"""
Primer paso de procesamiento:
- Cargar archivo crudo.
- Extraer variables de interés.
- Guardar resultado en carpeta processed.
"""

from pathlib import Path

import pandas as pd
from IPython.display import display

# --- Configuración (ajustar a tu archivo) ---
DATA_PATH = "data/raw/BD_RECPESO.csv"
COORDS = ["Este", "Norte", "Cota"]
TARGET = "Rec_Peso_PND25_(%)"
OUTPUT_PATH = "data/processed/df_rec_peso_pnd25.xlsx"
SENTINEL_VALUE = -99
MAX_NORTE = 27000


def load_data(file_path: Path) -> pd.DataFrame:
    """Carga el archivo de datos crudos.

    Args:
        file_path: Ruta del archivo CSV fuente.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si falla la lectura del archivo.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path.resolve()}")

    try:
        return pd.read_csv(file_path, sep=";")
    except Exception as exc:
        raise ValueError(
            f"No fue posible leer el archivo CSV: {file_path.resolve()}"
        ) from exc


def select_variables(source_df: pd.DataFrame, coords: list[str], target: str) -> pd.DataFrame:
    """Selecciona coordenadas y variable objetivo.

    Args:
        source_df: DataFrame original.
        coords: Lista de columnas de coordenadas.
        target: Nombre de la columna objetivo.

    Returns:
        DataFrame con las columnas seleccionadas.

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    selected_columns = coords + [target]
    missing_columns = [col for col in selected_columns if col not in source_df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    selected_df = source_df[selected_columns].copy()
    selected_df = selected_df.rename(columns={target: "recpe_og"})
    return selected_df


def remove_sentinel_rows(data_df: pd.DataFrame, value_col: str, sentinel_value: float) -> pd.DataFrame:
    """Elimina filas con valor sentinela en la variable objetivo.

    Args:
        data_df: DataFrame procesado.
        value_col: Columna objetivo para filtrar.
        sentinel_value: Valor sentinela a eliminar.

    Returns:
        DataFrame sin filas con valor sentinela.
    """
    return data_df.loc[data_df[value_col] != sentinel_value].copy()


def filter_max_norte(data_df: pd.DataFrame, max_norte: float) -> pd.DataFrame:
    """Filtra filas manteniendo solo Norte menor o igual al valor máximo.

    Args:
        data_df: DataFrame procesado.
        max_norte: Valor máximo permitido para la columna Norte.

    Returns:
        DataFrame filtrado por Norte.
    """
    return data_df.loc[data_df["Norte"] <= max_norte].copy()


def save_processed_data(data_df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el DataFrame procesado en carpeta processed.

    Args:
        data_df: DataFrame a guardar.
        output_path: Ruta final del archivo de salida.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_excel(output_path, index=False)


input_path = Path(DATA_PATH)
processed_path = Path(OUTPUT_PATH)

raw_df       = load_data(input_path)
selected_df  = select_variables(raw_df, COORDS, TARGET)
no_sent_df   = remove_sentinel_rows(selected_df, "recpe_og", SENTINEL_VALUE)
processed_df = filter_max_norte(no_sent_df, MAX_NORTE)

n_raw      = len(raw_df)
n_selected = len(selected_df)
n_sentinel = n_selected - len(no_sent_df)
n_norte    = len(no_sent_df) - len(processed_df)
n_final    = len(processed_df)

print("=" * 45)
print("  Resumen de carga y filtrado")
print("=" * 45)
print(f"  Registros crudos leidos    : {n_raw:>6,}")
print(f"  Columnas seleccionadas     : {len(selected_df.columns)} ({', '.join(selected_df.columns.tolist())})")
print(f"  Eliminados (valor centinela {SENTINEL_VALUE}): {n_sentinel:>4,}")
print(f"  Eliminados (Norte > {MAX_NORTE:,})  : {n_norte:>4,}")
print("-" * 45)
print(f"  Registros finales          : {n_final:>6,}  ({n_final/n_raw*100:.1f}% del total)")
print("=" * 45)

save_processed_data(processed_df, processed_path)

display(processed_df.head())
print(f"\nArchivo procesado generado en: {processed_path.resolve()}")


# %%
