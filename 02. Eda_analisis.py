#%%
"""EDA simple para df_rec_peso_pnd25.xlsx."""

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

import config_figuras_tesina as cfg_fig

cfg_fig = importlib.reload(cfg_fig)

setup_figuras_tesina = cfg_fig.setup_figuras_tesina
plot_deriva = cfg_fig.plot_deriva
plot_hist_original_vs_gaussian = cfg_fig.plot_hist_original_vs_gaussian
plot_qq_original_vs_gaussian = cfg_fig.plot_qq_original_vs_gaussian
plot_mapa_3d = cfg_fig.plot_mapa_3d
plot_proyecciones_2d = cfg_fig.plot_proyecciones_2d

# --- Configuracion ---
DATA_PATH = Path("data") / "processed" / "df_rec_peso_pnd25.xlsx"
GAUSS_OUTPUT_PATH = Path("data") / "processed" / "df_rec_peso_pnd25_gauss.xlsx"
COORDS = ["Este", "Norte", "Cota"]
TARGET = "recpe_og"
EDA_IMAGENES_DIR = Path("imagenes") / "eda"
SHOW_FIGURES = True


def load_processed_data(file_path: Path) -> pd.DataFrame:
    """Carga archivo procesado desde Excel."""
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {file_path.resolve()}")
    return pd.read_excel(file_path)


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Valida columnas requeridas."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")


def gaussian_transform(series: pd.Series) -> pd.Series:
    """Aplica transformacion gaussiana univariada."""
    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(series)),
        output_distribution="normal",
        random_state=42,
    )
    transformed = transformer.fit_transform(series.to_numpy().reshape(-1, 1)).ravel()
    return pd.Series(transformed, index=series.index, name=f"{series.name}_gauss")


def show_table(df: pd.DataFrame) -> None:
    """Muestra tabla en notebook o consola."""
    try:
        from IPython.display import display

        display(df)
    except Exception:
        print(df.to_string())


def maybe_show_figure(show_figures: bool) -> None:
    """Muestra figuras solo si esta habilitado."""
    if show_figures:
        plt.show()
    else:
        plt.close()


def save_gaussian_data(data_df: pd.DataFrame, output_path: Path) -> None:
    """Guarda un nuevo Excel con la variable transformada."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_excel(output_path, index=False)


setup_figuras_tesina()
EDA_IMAGENES_DIR.mkdir(parents=True, exist_ok=True)
eda_df = load_processed_data(DATA_PATH)
validate_columns(eda_df, COORDS + [TARGET])

# 1) Estadisticas descriptivas
print("Estadisticas descriptivas:")
show_table(eda_df[COORDS + [TARGET]].describe().T)

# 2) Histograma original vs transformacion gaussiana
eda_df["recpe_gauss"] = gaussian_transform(eda_df[TARGET])
save_gaussian_data(eda_df, GAUSS_OUTPUT_PATH)

print(f"Archivo con transformacion gaussiana: {GAUSS_OUTPUT_PATH.resolve()}")
fig_hist = plot_hist_original_vs_gaussian(
    original_values=eda_df[TARGET],
    gaussian_values=eda_df["recpe_gauss"],
    target_label=TARGET,
)
fig_hist.savefig(EDA_IMAGENES_DIR / "hist_original_vs_gauss_recpe.png", dpi=300)
maybe_show_figure(SHOW_FIGURES)

# 3) Q-Q plot original vs transformado
fig_qq = plot_qq_original_vs_gaussian(
    original_values=eda_df[TARGET],
    gaussian_values=eda_df["recpe_gauss"],
    target_label=TARGET,
)
fig_qq.savefig(EDA_IMAGENES_DIR / "qq_original_vs_gauss_recpe.png", dpi=300)
maybe_show_figure(SHOW_FIGURES)

# 4) Mapa 3D del yacimiento aproximado
fig_3d = plot_mapa_3d(
    df=eda_df,
    x_col=COORDS[0],
    y_col=COORDS[1],
    z_col=COORDS[2],
    value_col=TARGET,
)

fig_3d.savefig(EDA_IMAGENES_DIR / "mapa_3d_recpe.png", dpi=300)
maybe_show_figure(SHOW_FIGURES)

# 5) Subplot de proyecciones 2D
fig_proj = plot_proyecciones_2d(
    df=eda_df,
    x_col=COORDS[0],
    y_col=COORDS[1],
    z_col=COORDS[2],
    value_col=TARGET,
)
fig_proj.savefig(EDA_IMAGENES_DIR / "proyecciones_2d_recpe.png", dpi=300)
maybe_show_figure(SHOW_FIGURES)

# 6) Subplot de deriva por coordenada (promedios cada 100 m)
fig_deriva = plot_deriva(
    df=eda_df,
    coord_cols=COORDS,
    value_col=TARGET,
    bin_size=100.0,
)
fig_deriva.savefig(EDA_IMAGENES_DIR / "deriva_recpe.png", dpi=300)
maybe_show_figure(SHOW_FIGURES)

print(f"EDA simple completado. Figuras guardadas en: {EDA_IMAGENES_DIR.resolve()}")


# %%
