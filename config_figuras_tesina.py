# %%
"""Configuración estándar de figuras para tesina."""

import pandas as pd

# ── Calidad y formato ─────────────────────────────────────────────────────────
DPI_TESINA = 300
FORMATO_FIGURA = "pdf"  # "pdf" (vectorial) o "png" (300 DPI)

# ── Estilo visual profesional (tesis) ─────────────────────────────────────────
CMAP_ESPACIAL = "jet"
COLOR_HIST_ORIGINAL = "#4C78A8"
COLOR_HIST_TRANSFORMADO = "#F58518"
COLOR_DERIVA = "#2F4B7C"
COLOR_QQ_RECTA = "#6E6E6E"
COLORS_CLUSTER = ["#4C78A8", "#F58518", "#54A24B"]

# Configuracion central de dimensiones (cm) por tipo de figura.
# Edita solo estos valores para reutilizarlos en todos tus scripts.
FIGSIZE_CM = {
    "hist": (15.0, 6.0),
    "qq": (15.0, 6.0),
    "mapa_3d": (15.0, 10.0),
    "proyecciones_2d": (25.0, 8),
    "deriva": (25.0, 8.0),
    "cluster_3d": (15.0, 10.0),
    "cluster_2d": (18.0, 6.5),
    "cluster_stats": (17.0, 7.5),
    "real_vs_pred": (18.0, 9.0),
}

# ── Utilidades ────────────────────────────────────────────────────────────────

def figsize_cm(ancho_cm: float, alto_cm: float) -> tuple[float, float]:
    """Convierte cm → pulgadas para matplotlib (1 in = 2.54 cm)."""
    return (ancho_cm / 2.54, alto_cm / 2.54)


def get_figsize_cm(fig_key: str) -> tuple[float, float]:
    """Obtiene dimensiones por tipo de figura desde configuracion central."""
    if fig_key not in FIGSIZE_CM:
        raise KeyError(f"Tipo de figura no configurado: {fig_key}")
    return FIGSIZE_CM[fig_key]


def setup_figuras_tesina() -> None:
    """
    Aplica rcParams globales para figuras de tesina académica.
    Llamar una sola vez al inicio del script/notebook.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # ── Guardado ──────────────────────────────────────────────────────────
        "figure.dpi"       : 150,          # previsualización en pantalla
        "savefig.dpi"      : DPI_TESINA,
        "savefig.bbox"     : "tight",
        "savefig.format"   : FORMATO_FIGURA,

        # ── Tipografía ────────────────────────────────────────────────────────
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # ── Grilla y bordes ───────────────────────────────────────────────────
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "grid.color": "#B0B0B0",
        "axes.spines.top": False,
        "axes.spines.right": False,

        # ── Fondo ─────────────────────────────────────────────────────────────
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ── Funciones de gráfico ──────────────────────────────────────────────────────

def _set_box_aspect_3d(ax, df, x_col, y_col, z_col,
                        zoom_x=2.0, zoom_y=1.0, zoom_z=1.0):
    """Aspect ratio proporcional para ejes 3D."""
    rx = df[x_col].max() - df[x_col].min()
    ry = df[y_col].max() - df[y_col].min()
    rz = df[z_col].max() - df[z_col].min()
    m  = max(rx, ry, rz)
    if m > 0:
        ax.set_box_aspect((rx / m * zoom_x, ry / m * zoom_y, rz / m * zoom_z))


def plot_hist_original_vs_gaussian(
    original_values,
    gaussian_values,
    target_label: str,
    ancho_cm: float | None = None,
    alto_cm: float | None = None,
):
    """
    2 paneles: histograma original | histograma transformado (normal score).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    default_ancho_cm, default_alto_cm = get_figsize_cm("hist")
    ancho = default_ancho_cm if ancho_cm is None else ancho_cm
    alto = default_alto_cm if alto_cm is None else alto_cm
    fig, axes = plt.subplots(1, 2, figsize=figsize_cm(ancho, alto),
                             constrained_layout=True)

    sns.histplot(
        original_values,
        kde=True,
        ax=axes[0],
        color=COLOR_HIST_ORIGINAL,
        edgecolor="white",
        linewidth=0.3,
    )
    axes[0].set_title(f"Distribución original:\n{target_label}")
    axes[0].set_xlabel(target_label)
    axes[0].set_ylabel("Frecuencia")

    sns.histplot(
        gaussian_values,
        kde=True,
        ax=axes[1],
        color=COLOR_HIST_TRANSFORMADO,
        edgecolor="white",
        linewidth=0.3,
    )
    axes[1].set_title(f"Distribución transformada:\n{target_label}")
    axes[1].set_xlabel(f"{target_label} (normal score)")
    axes[1].set_ylabel("Frecuencia")

    return fig


def plot_qq_original_vs_gaussian(
    original_values,
    gaussian_values,
    target_label: str,
    ancho_cm: float | None = None,
    alto_cm: float | None = None,
):
    """
    2 paneles: Q-Q original | Q-Q transformado.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    default_ancho_cm, default_alto_cm = get_figsize_cm("qq")
    ancho = default_ancho_cm if ancho_cm is None else ancho_cm
    alto = default_alto_cm if alto_cm is None else alto_cm
    fig, axes = plt.subplots(1, 2, figsize=figsize_cm(ancho, alto),
                             constrained_layout=True)

    stats.probplot(original_values, dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q plot original:\n{target_label}")
    axes[0].get_lines()[0].set(markersize=2.5, alpha=0.7, color=COLOR_HIST_ORIGINAL)
    axes[0].get_lines()[1].set(color=COLOR_QQ_RECTA, linewidth=1.0)

    stats.probplot(gaussian_values, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q plot transformado:\n{target_label} (normal score)")
    axes[1].get_lines()[0].set(markersize=2.5, alpha=0.7, color=COLOR_HIST_TRANSFORMADO)
    axes[1].get_lines()[1].set(color=COLOR_QQ_RECTA, linewidth=1.0)

    return fig


def plot_mapa_3d(df, x_col, y_col, z_col, value_col,
                 zoom_x=2.0, zoom_z=1.4,
                 titulo="Distribución Espacial de la Recuperación en Peso (%)",
                 ancho_cm: float | None = None,
                 alto_cm: float | None = None,
                 color_values=None,
                 cmap=None,
                 colorbar_label: str | None = None,
                 colorbar_ticks: list[float] | None = None,
                 colorbar_ticklabels: list[str] | None = None):
    """Mapa 3D respetando proporciones espaciales."""
    import matplotlib.pyplot as plt

    default_ancho_cm, default_alto_cm = get_figsize_cm("mapa_3d")
    ancho = default_ancho_cm if ancho_cm is None else ancho_cm
    alto = default_alto_cm if alto_cm is None else alto_cm
    fig = plt.figure(figsize=figsize_cm(ancho, alto))
    ax  = fig.add_subplot(111, projection="3d")

    values = df[value_col] if color_values is None else color_values
    colormap = CMAP_ESPACIAL if cmap is None else cmap
    sc = ax.scatter(
        df[x_col],
        df[y_col],
        df[z_col],
        c=values,
        cmap=colormap,
        s=10,
        alpha=1.0,
    )

    ax.set_title(titulo, pad=10)
    ax.set_xlabel(x_col, labelpad=10)
    ax.set_ylabel(y_col, labelpad=10)
    ax.set_zlabel(z_col, labelpad=10)
    ax.tick_params(axis="x", pad=3)
    ax.tick_params(axis="y", pad=3)
    ax.tick_params(axis="z", pad=5)

    _set_box_aspect_3d(ax, df, x_col, y_col, z_col,
                        zoom_x=zoom_x, zoom_y=1.0, zoom_z=zoom_z)
    cb_label = value_col if colorbar_label is None else colorbar_label
    # pad=0.15 evita que la barra de color se solape con el label del eje Z (Cota).
    # tight_layout se omite porque no funciona correctamente con ejes 3D.
    colorbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.15, label=cb_label)
    if colorbar_ticks is not None:
        colorbar.set_ticks(colorbar_ticks)
    if colorbar_ticklabels is not None:
        colorbar.set_ticklabels(colorbar_ticklabels)
    return fig


def plot_proyecciones_2d(
    df,
    x_col,
    y_col,
    z_col,
    value_col,
    ancho_cm: float | None = None,
    alto_cm: float | None = None,
    color_values=None,
    cmap=None,
    colorbar_label: str | None = None,
    colorbar_ticks: list[float] | None = None,
    colorbar_ticklabels: list[str] | None = None,
):
    """3 proyecciones 2D: XY, XZ, YZ."""
    import matplotlib.pyplot as plt

    default_ancho_cm, default_alto_cm = get_figsize_cm("proyecciones_2d")
    ancho = default_ancho_cm if ancho_cm is None else ancho_cm
    alto = default_alto_cm if alto_cm is None else alto_cm
    fig, axes = plt.subplots(1, 3, figsize=figsize_cm(ancho, alto),
                             constrained_layout=True)
    pares = [(x_col, y_col), (x_col, z_col), (y_col, z_col)]
    values = df[value_col] if color_values is None else color_values
    colormap = CMAP_ESPACIAL if cmap is None else cmap

    for ax, (xc, yc) in zip(axes, pares):
        sc = ax.scatter(
            df[xc],
            df[yc],
            c=values,
            cmap=colormap,
            s=8,
            alpha=0.68,
        )
        ax.set_title(f"{xc} vs {yc}")
        ax.set_xlabel(xc)
        ax.set_ylabel(yc)

    cb_label = value_col if colorbar_label is None else colorbar_label
    colorbar = fig.colorbar(
        sc,
        ax=list(axes),
        location="right",
        shrink=0.85,
        pad=0.02,
        fraction=0.03,
        aspect=30,
        label=cb_label,
    )
    if colorbar_ticks is not None:
        colorbar.set_ticks(colorbar_ticks)
    if colorbar_ticklabels is not None:
        colorbar.set_ticklabels(colorbar_ticklabels)
    return fig


def plot_deriva(
    df,
    coord_cols: list[str],
    value_col: str,
    bin_size: float = 10.0,
    ancho_cm: float | None = None,
    alto_cm: float | None = None,
    value_label: str | None = None,
):
    """
    3 paneles de deriva espacial (promedio por tramos de bin_size metros).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    default_ancho_cm, default_alto_cm = get_figsize_cm("deriva")
    ancho = default_ancho_cm if ancho_cm is None else ancho_cm
    alto = default_alto_cm if alto_cm is None else alto_cm
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize_cm(ancho, alto),
        constrained_layout=True,
    )

    for ax, coord in zip(axes, coord_cols):
        lo, hi = df[coord].min(), df[coord].max()
        bins = np.arange(lo, hi + bin_size, bin_size)
        if len(bins) < 2:
            bins = np.array([lo, hi + 1.0])

        tmp = df[[coord, value_col]].copy()
        tmp["bin"] = pd.cut(tmp[coord], bins=bins, include_lowest=True)
        mean_df = (tmp.groupby("bin", observed=False)
                      .agg(coord_mean=(coord, "mean"),
                           value_mean=(value_col, "mean"))
                      .dropna()
                      .sort_values("coord_mean"))

        ax.plot(
            mean_df["coord_mean"],
            mean_df["value_mean"],
            marker="o",
            linewidth=1.3,
            markersize=3.0,
            color=COLOR_DERIVA,
        )
        ylabel = f"Promedio {value_label}" if value_label else f"Promedio {value_col}"
        ax.set_title(f"Deriva — {coord} ({int(bin_size)} m)")
        ax.set_xlabel(coord)
        ax.set_ylabel(ylabel)
        ax.set_ylim(10, 30)
    return fig


def plot_mapa_3d_clusters(
    df,
    x_col: str,
    y_col: str,
    z_col: str,
    cluster_col: str = "cluster",
    titulo: str = "Distribución Espacial de la Recuperación en Peso (%)",
    ancho_cm: float | None = None,
    alto_cm: float | None = None,
):
    """Mapa 3D coloreado por etiqueta de cluster."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cluster_values = df[cluster_col] - 1
    cmap_cluster = ListedColormap(COLORS_CLUSTER)
    fig = plot_mapa_3d(
        df=df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        value_col=cluster_col,
        titulo=titulo,
        ancho_cm=ancho_cm,
        alto_cm=alto_cm,
        color_values=cluster_values,
        cmap=cmap_cluster,
        colorbar_label="Cluster",
        colorbar_ticks=[0, 1, 2],
        colorbar_ticklabels=["1", "2", "3"],
    )
    return fig


def plot_proyecciones_2d_clusters(
    df,
    x_col: str,
    y_col: str,
    z_col: str,
    cluster_col: str = "cluster",
    ancho_cm: float | None = None,
    alto_cm: float | None = None,
):
    """Proyecciones 2D coloreadas por etiqueta de cluster."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cluster_values = df[cluster_col] - 1
    cmap_cluster = ListedColormap(COLORS_CLUSTER)
    fig = plot_proyecciones_2d(
        df=df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        value_col=cluster_col,
        ancho_cm=ancho_cm,
        alto_cm=alto_cm,
        color_values=cluster_values,
        cmap=cmap_cluster,
        colorbar_label="Cluster",
        colorbar_ticks=[0, 1, 2],
        colorbar_ticklabels=["1", "2", "3"],
    )
    return fig


def plot_boxplot_efecto_proporcional_por_cluster(
    df,
    cluster_col: str = "cluster",
    target_col: str = "recpe_og",
    ancho_cm: float | None = None,
    alto_cm: float | None = None,
):
    """Subplot con boxplot por cluster y efecto proporcional."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    default_ancho_cm, default_alto_cm = get_figsize_cm("cluster_stats")
    ancho = default_ancho_cm if ancho_cm is None else ancho_cm
    alto = default_alto_cm if alto_cm is None else alto_cm

    fig, axes = plt.subplots(1, 2, figsize=figsize_cm(ancho, alto), constrained_layout=True)
    cluster_order = sorted(df[cluster_col].dropna().unique().tolist())

    sns.boxplot(
        data=df,
        x=cluster_col,
        y=target_col,
        order=cluster_order,
        ax=axes[0],
        palette=COLORS_CLUSTER[:len(cluster_order)],
        linewidth=0.8,
    )
    axes[0].set_title("Boxplot por cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel(target_col)

    ratio_df = (
        df.groupby(cluster_col, observed=False)[target_col]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(cluster_col)
    )
    ratio_df["efecto_proporcional_pct"] = (ratio_df["std"] / ratio_df["mean"]) * 100.0
    ratio_df = ratio_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["efecto_proporcional_pct"]
    )

    axes[1].scatter(
        ratio_df[cluster_col],
        ratio_df["efecto_proporcional_pct"],
        c=COLORS_CLUSTER[:len(ratio_df)],
        s=55,
        zorder=3,
    )
    axes[1].plot(
        ratio_df[cluster_col],
        ratio_df["efecto_proporcional_pct"],
        color="#5B5B5B",
        linewidth=0.9,
        alpha=0.7,
        zorder=2,
    )
    axes[1].set_title("Efecto proporcional por cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Efecto proporcional (%)")
    axes[1].set_xticks(ratio_df[cluster_col].tolist())
    axes[0].set_ylim(bottom=0)

    # Escala automatica para que el efecto proporcional sea legible.
    min_effect = float(ratio_df["efecto_proporcional_pct"].min())
    max_effect = float(ratio_df["efecto_proporcional_pct"].max())
    effect_span = max_effect - min_effect
    padding = max(1.0, effect_span * 0.20)
    upper_limit = max_effect + padding
    axes[1].set_ylim(bottom=0, top=upper_limit)

    return fig


def plot_real_vs_pred_subplot(
    y_true_og: "pd.Series",
    pred_knn_og: "pd.Series",
    pred_xgb_og: "pd.Series",
    metrics_knn: dict,
    metrics_xgb: dict,
    target_label: str = "Recuperación en peso (%)",
    xlabel_prefix: str = "Real",
    ylabel_prefix: str = "Predicho",
    title_knn: str = "KNN — Real vs predicho",
    title_xgb: str = "XGBoost — Real vs predicho",
    ancho_cm: float | None = None,
    alto_cm: float | None = None,
):
    """Subplot lado a lado: KNN | XGBoost con valores en escala original.

    Args:
        y_true_og: Valores de referencia en escala original (eje X).
        pred_knn_og: Predicciones KNN en escala original.
        pred_xgb_og: Predicciones XGBoost en escala original.
        metrics_knn: Diccionario con r2, rmse y mape_pct para KNN.
        metrics_xgb: Diccionario con r2, rmse y mape_pct para XGBoost.
        target_label: Etiqueta de la variable en los ejes.
        xlabel_prefix: Prefijo del eje X (ej. "Real", "Kriging").
        ylabel_prefix: Prefijo del eje Y (ej. "Predicho", "ML predicho").
        title_knn: Titulo del panel KNN.
        title_xgb: Titulo del panel XGBoost.
        ancho_cm: Ancho total en cm (usa FIGSIZE_CM si None).
        alto_cm: Alto total en cm (usa FIGSIZE_CM si None).
    """
    import matplotlib.pyplot as plt

    default_ancho_cm, default_alto_cm = get_figsize_cm("real_vs_pred")
    ancho = default_ancho_cm if ancho_cm is None else ancho_cm
    alto = default_alto_cm if alto_cm is None else alto_cm

    fig, axes = plt.subplots(1, 2, figsize=figsize_cm(ancho, alto), constrained_layout=True)

    model_configs = [
        (title_knn, pred_knn_og, metrics_knn),
        (title_xgb, pred_xgb_og, metrics_xgb),
    ]

    for ax, (title, pred_og, metrics) in zip(axes, model_configs):
        all_values = list(y_true_og) + list(pred_og)
        lower_bound = float(pd.Series(all_values).quantile(0.01))
        upper_bound = float(pd.Series(all_values).quantile(0.99))
        margin = (upper_bound - lower_bound) * 0.04
        axis_min = lower_bound - margin
        axis_max = upper_bound + margin

        ax.scatter(
            y_true_og,
            pred_og,
            alpha=0.55,
            s=16,
            color="#4C78A8",
            edgecolors="none",
        )
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            "--",
            color="#222222",
            linewidth=1.2,
            label="y = x",
        )
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel(f"{xlabel_prefix} ({target_label})")
        ax.set_ylabel(f"{ylabel_prefix} ({target_label})")
        ax.legend(fontsize=8, loc="upper left")

        stats_text = (
            f"$R^2$ = {metrics['r2']:.3f}\n"
            f"RMSE = {metrics['rmse']:.3f}\n"
            f"MAPE = {metrics['mape_pct']:.1f}%"
        )
        ax.text(
            0.97,
            0.03,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7, "edgecolor": "#CCCCCC"},
        )

    return fig


def _scatter_panel(
    ax,
    x: "pd.Series",
    y: "pd.Series",
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Panel scatter cuadrado con linea y=x y metricas internas."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

    all_values = list(x) + list(y)
    lower = float(pd.Series(all_values).quantile(0.01))
    upper = float(pd.Series(all_values).quantile(0.99))
    margin = (upper - lower) * 0.04

    ax.scatter(x, y, alpha=0.45, s=14, color="#4C78A8", edgecolors="none")
    ax.plot([lower - margin, upper + margin], [lower - margin, upper + margin],
            "--", color="#222222", linewidth=1.2, label="y = x")
    ax.set_xlim(lower - margin, upper + margin)
    ax.set_ylim(lower - margin, upper + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="upper left")

    rmse = mean_squared_error(x, y) ** 0.5
    mape = mean_absolute_percentage_error(x, y) * 100.0
    r2 = r2_score(x, y)
    stats_text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAPE = {mape:.1f}%\nn = {len(x)}"
    ax.text(
        0.97, 0.03, stats_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom", horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7, "edgecolor": "#CCCCCC"},
    )


def plot_real_vs_ml_vs_kriging(
    y_true_og: "pd.Series",
    y_ml: "pd.Series",
    y_kriging: "pd.Series",
    model_name: str,
    target_label: str = "Recuperación en peso (%)",
    ancho_cm: float | None = None,
    alto_cm: float | None = None,
) -> "plt.Figure":
    """Subplot 1x2 para un modelo: [real vs ML] y [real vs Kriging].

    Permite comparar visualmente el desempeño del modelo ML contra la
    estimacion kriging usando los mismos datos originales como referencia.

    Args:
        y_true_og: Valores originales de referencia (recpe_og).
        y_ml: Predicciones del modelo ML.
        y_kriging: Estimaciones kriging (recpe_val).
        model_name: Nombre del modelo para los titulos (ej. "KNN").
        target_label: Etiqueta de la variable.
        ancho_cm: Ancho total en cm (usa FIGSIZE_CM['real_vs_pred'] si None).
        alto_cm: Alto total en cm (usa FIGSIZE_CM['real_vs_pred'] si None).
    """
    import matplotlib.pyplot as plt

    default_w, default_h = get_figsize_cm("real_vs_pred")
    ancho = default_w if ancho_cm is None else ancho_cm
    alto = default_h if alto_cm is None else alto_cm

    fig, axes = plt.subplots(1, 2, figsize=figsize_cm(ancho, alto), constrained_layout=True)

    _scatter_panel(
        axes[0],
        x=y_true_og,
        y=y_ml,
        title=f"{model_name} — Real vs ML",
        xlabel=f"Real ({target_label})",
        ylabel=f"{model_name} predicho ({target_label})",
    )
    _scatter_panel(
        axes[1],
        x=y_true_og,
        y=y_kriging,
        title="Kriging — Real vs Kriging",
        xlabel=f"Real ({target_label})",
        ylabel=f"Kriging ({target_label})",
    )

    return fig