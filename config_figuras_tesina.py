# %%
"""
Configuración estándar de figuras para tesina/tesis.

Estándares habituales en trabajos académicos:
- Resolución: 300 DPI (impresión)
- Formato: PDF (vector, escalable) o PNG a 300 DPI
- Dimensiones: ancho útil en A4 con márgenes ~2.5–3 cm suele ser 12–16 cm.
  Con márgenes 3 cm cada lado: 21 - 6 = 15 cm de ancho útil.
- Tipografía en figuras: tamaños coherentes (títulos 12 pt, ejes 11 pt, etc.).

Uso en matplotlib:
  from config_figuras_tesina import (
      setup_figuras_tesina,
      figsize_cm,
      ANCHO_ESTANDAR_CM,
      DPI_TESINA,
      FORMATO_FIGURA,
      IMAGENES_DIR,
  )
  setup_figuras_tesina()  # en __main__ o al inicio
  fig, ax = plt.subplots(figsize=figsize_cm(14, 5))
  ...
  plt.savefig(IMAGENES_DIR / "nombre.pdf")  # o extensión según FORMATO_FIGURA
"""

from pathlib import Path

# --- Dimensiones estándar (en cm) ---
# Ancho útil típico en A4 (21 cm) con márgenes 2.5–3 cm: 14–16 cm
ANCHO_ESTANDAR_CM = 14.0
# Altura por defecto para una fila de ejes (ej. 1x2, 1x3)
ALTO_ESTANDAR_CM = 5.0
# Para figuras más altas (ej. 2x2, diagnósticos)
ALTO_GRANDE_CM = 10.0

# --- Calidad y formato ---
# 300 DPI es el estándar para impresión en tesis/revistas
DPI_TESINA = 300
# "pdf" = vector (recomendado para tesina), "png" = mapa de bits a 300 DPI
FORMATO_FIGURA = "pdf"

# Carpeta donde se guardan las figuras (raíz del proyecto)
IMAGENES_DIR = Path("imagenes")


def path_imagen(nombre_sin_ext: str) -> Path:
    """Ruta completa para guardar una figura (nombre sin extensión → imagenes/nombre.pdf o .png)."""
    return IMAGENES_DIR / f"{nombre_sin_ext}.{extension_figura()}"


def figsize_cm(ancho_cm: float, alto_cm: float) -> tuple[float, float]:
    """Convierte dimensiones en cm a (ancho, alto) en pulgadas para matplotlib.

    matplotlib usa pulgadas en figsize; 1 inch = 2.54 cm.
    """
    return (ancho_cm / 2.54, alto_cm / 2.54)


def extension_figura() -> str:
    """Devuelve la extensión de archivo según FORMATO_FIGURA."""
    return "pdf" if FORMATO_FIGURA.lower() == "pdf" else "png"


def setup_figuras_tesina() -> None:
    """
    Configura rcParams de matplotlib para figuras de tesina:
    DPI de guardado, tipografía, grid, fondo blanco.
    Llamar al inicio del script.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": DPI_TESINA,
        "savefig.bbox": "tight",
        "savefig.format": FORMATO_FIGURA,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def set_proportional_aspect(ax, df, x_col, y_col, z_col, zoom_x: float = 2.0) -> None:
    """Ajusta el aspect ratio del gráfico 3D según el rango real de las coordenadas.
    Se ha agregado zoom_x para ensanchar el eje X (Este) y evitar que se vea muy delgado.
    """
    x, y, z = df[x_col], df[y_col], df[z_col]
    range_x = x.max() - x.min()
    range_y = y.max() - y.min()
    range_z = z.max() - z.min()
    max_range = max(range_x, range_y, range_z)
    if max_range > 0:
        rx = (range_x / max_range) * zoom_x
        ry = (range_y / max_range)
        rz = (range_z / max_range)
        ax.set_box_aspect((rx, ry, rz))

