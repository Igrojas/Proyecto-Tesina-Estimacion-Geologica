#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# ── Cargar datos (sin conversión automática de tipos) ────────
file_path = r"data/raw/BD_RECPESO.csv"
file_val  = r"data/validacion/modelo_validos.csv"

df     = pd.read_csv(file_path, decimal=".", sep=";",  dtype=str)
df_val = pd.read_csv(file_val,  decimal=".", sep=",",  dtype=str)

# ============================================================
# DIAGNÓSTICO — qué formato tienen las columnas de coords
# ============================================================
#%%
COORDS_ORIG = ["Este", "Norte", "Cota"]
COORDS_VAL  = ["xcentre", "ycentre", "zcentre"]

print("=" * 60)
print("DIAGNÓSTICO df (original)")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes.to_string()}")
print(f"\nMuestra cruda de coordenadas (primeras 5 filas):")
for col in COORDS_ORIG:
    if col in df.columns:
        vals = df[col].head(5).tolist()
        print(f"  {col}: {vals}")

print("\n" + "=" * 60)
print("DIAGNÓSTICO df_val (validación)")
print("=" * 60)
print(f"Shape: {df_val.shape}")
print(f"\nDtypes:\n{df_val.dtypes.to_string()}")
print(f"\nMuestra cruda de coordenadas (primeras 5 filas):")
for col in COORDS_VAL:
    if col in df_val.columns:
        vals = df_val[col].head(5).tolist()
        print(f"  {col}: {vals}")

# ============================================================
# CONVERSIÓN ROBUSTA a float Python
# Maneja formatos: "1234.56", "1.234,56", "1,234.56", "1234,56"
# ============================================================
#%%
def to_float(series: pd.Series) -> pd.Series:
    """
    Convierte un valor string a float con detección automática de formato.
    Usa apply() para operar elemento a elemento sin problemas de índice.
    """
    def _conv(val):
        s = str(val).strip()
        if s in ("", "nan", "None", "NaN", "-"):
            return np.nan
        n_coma  = s.count(",")
        n_punto = s.count(".")
        if n_coma > 0 and n_punto > 0:
            if s.index(".") < s.index(","):
                # Europeo: "1.234,56" → punto=miles, coma=decimal
                s = s.replace(".", "").replace(",", ".")
            else:
                # US: "1,234.56" → coma=miles, punto=decimal
                s = s.replace(",", "")
        elif n_coma > 0:
            # Solo coma: "1234,56" → coma es decimal
            s = s.replace(",", ".")
        # Solo punto o sin separadores: ya está bien
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.apply(_conv)





# Aplicar conversión a TODAS las columnas numéricas de df
print("Convirtiendo df (original)...")
for col in df.columns:
    converted = to_float(df[col])
    if converted.notna().sum() > len(df) * 0.5:   # al menos 50% numéricos
        df[col] = converted
        print(f"  ✓ {col}: {df[col].dtype}  "
              f"[{df[col].min():.2f} … {df[col].max():.2f}]")
    else:
        # Columna de texto — dejarla como string
        df[col] = df[col].astype(str)

print("\nConvirtiendo df_val (validación)...")
for col in df_val.columns:
    converted = to_float(df_val[col])
    if converted.notna().sum() > len(df_val) * 0.5:
        df_val[col] = converted
        print(f"  ✓ {col}: {df_val[col].dtype}  "
              f"[{df_val[col].min():.2f} … {df_val[col].max():.2f}]")
    else:
        df_val[col] = df_val[col].astype(str)

# ── Mapear coordenadas de validación a nombres estándar ─────
df_val["Este"]  = df_val["xcentre"]
df_val["Norte"] = df_val["ycentre"]
df_val["Cota"]  = df_val["zcentre"]

print("\n" + "=" * 60)
print("VERIFICACIÓN FINAL — rangos de coordenadas")
print("=" * 60)
COORDS = ["Este", "Norte", "Cota"]
for col in COORDS:
    print(f"\n  {col}")
    print(f"    df     → min={df[col].min():.1f}  max={df[col].max():.1f}  "
          f"NaN={df[col].isna().sum()}")
    print(f"    df_val → min={df_val[col].min():.1f}  max={df_val[col].max():.1f}  "
          f"NaN={df_val[col].isna().sum()}")



# ── Extraer matrices numpy para KDTree ──────────────────────
pts_orig = df[COORDS].values.astype(float)
pts_val  = df_val[COORDS].values.astype(float)

# ── KDTree: para cada punto de df, el más cercano en df_val ──
tree = KDTree(pts_val)
distancias, indices = tree.query(pts_orig, k=1)  # k=1 → 1 solo vecino

# Construir tabla de cruces
cruce = df[COORDS].copy().reset_index(drop=True)
cruce["Orig_idx"]  = cruce.index
cruce["Val_idx"]   = indices
cruce["dist_3d"]   = distancias
cruce["Val_Este"]  = pts_val[indices, 0]
cruce["Val_Norte"] = pts_val[indices, 1]
cruce["Val_Cota"]  = pts_val[indices, 2]

print(f"Puntos originales : {len(df)}")
print(f"Puntos validación : {len(df_val)}")
print(f"\nEstadísticas de distancia 3D (m):")
print(cruce["dist_3d"].describe().round(2).to_string())

#%%
# ── Figura 1: Histograma de distancias ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(cruce["dist_3d"], bins=40, color="#2563eb", edgecolor="white", alpha=0.85)
axes[0].axvline(cruce["dist_3d"].median(), color="#dc2626", lw=2,
                label=f"Mediana: {cruce['dist_3d'].median():.1f} m")
axes[0].axvline(cruce["dist_3d"].mean(),   color="#d97706", lw=2, ls="--",
                label=f"Media: {cruce['dist_3d'].mean():.1f} m")
axes[0].set_xlabel("Distancia 3D al vecino más cercano (m)")
axes[0].set_ylabel("N° de puntos originales")
axes[0].set_title("Distribución de distancias orig → validación")
axes[0].legend()

# CDF acumulada
sorted_d = np.sort(cruce["dist_3d"].values)
cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
axes[1].plot(sorted_d, cdf*100, color="#2563eb", lw=2)
axes[1].axhline(50, color="#dc2626", lw=1.2, ls="--", label="P50")
axes[1].axhline(90, color="#d97706", lw=1.2, ls="--", label="P90")
p50 = np.percentile(cruce["dist_3d"], 50)
p90 = np.percentile(cruce["dist_3d"], 90)
axes[1].axvline(p50, color="#dc2626", lw=1, ls=":")
axes[1].axvline(p90, color="#d97706", lw=1, ls=":")
axes[1].set_xlabel("Distancia 3D (m)")
axes[1].set_ylabel("% de puntos")
axes[1].set_title(f"CDF acumulada  |  P50={p50:.1f}m  P90={p90:.1f}m")
axes[1].legend(); axes[1].grid(True, alpha=0.35)

plt.suptitle("Distancia entre cada punto original y su vecino más cercano en validación",
             fontsize=12, weight="bold")
plt.tight_layout()
plt.savefig("distancias_orig_vs_val.png", dpi=150, bbox_inches="tight")
plt.show()
print("[fig] distancias_orig_vs_val.png")

#%%
# ── Figura 2: Mapa 2D Este-Norte con líneas de cruce ────────
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(pts_val[:, 0],  pts_val[:, 1],  s=8,  color="#6b7280", alpha=0.4,
           label="Validación", zorder=2)
sc = ax.scatter(cruce["Este"], cruce["Norte"], c=cruce["dist_3d"],
                cmap="YlOrRd", s=18, edgecolors="black", lw=0.3,
                label="Original (color=distancia)", zorder=3)
plt.colorbar(sc, ax=ax, label="Distancia 3D al vecino más cercano (m)")

# Líneas de conexión (solo las más lejanas para no saturar)
umbral_viz = np.percentile(cruce["dist_3d"], 75)
for _, row in cruce[cruce["dist_3d"] >= umbral_viz].iterrows():
    ax.plot([row["Este"], row["Val_Este"]],
            [row["Norte"], row["Val_Norte"]],
            color="#dc2626", lw=0.6, alpha=0.5, zorder=1)

ax.set_xlabel("Este (m)"); ax.set_ylabel("Norte (m)")
ax.set_title("Mapa de pares más cercanos\n"
             f"Líneas rojas = distancias ≥ P75 ({umbral_viz:.1f} m)")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("mapa_pares_orig_vs_val.png", dpi=150, bbox_inches="tight")
plt.show()
print("[fig] mapa_pares_orig_vs_val.png")

# Guardar cruce
cruce.to_csv("cruce_orig_val_distancias.csv", index=False)
print("\n[csv] cruce_orig_val_distancias.csv")

#%%
# ============================================================
# COMPARACION: Rec_Peso_PND25_(%) (original) vs recpe (validacion)
# Solo puntos con Norte <= 27000
# ============================================================
from sklearn.metrics import r2_score, mean_squared_error

NORTE_MAX = 27000
COL_ORIG  = "Rec_Peso_PND25_(%)"
COL_VAL   = "recpe"

# 1) Filtrar cruce a Norte <= 27000
cruce_filt = cruce[cruce["Norte"] <= NORTE_MAX].copy().reset_index(drop=True)
print(f"Puntos Norte <= {NORTE_MAX}: {len(cruce_filt)}  (descartados: {len(cruce) - len(cruce_filt)})")

# 2) Extraer valores usando los indices del cruce
df_r     = df.reset_index(drop=True)
df_val_r = df_val.reset_index(drop=True)

y_orig = df_r.loc[cruce_filt["Orig_idx"].values, COL_ORIG].values.astype(float)
y_val  = df_val_r.loc[cruce_filt["Val_idx"].values, COL_VAL].values.astype(float)

SENTINEL = -99.0
# Eliminar NaN y sentinelas (-99) en cualquiera de los dos
mask = (
    ~np.isnan(y_orig) & ~np.isnan(y_val) &
    (y_orig != SENTINEL) & (y_val != SENTINEL)
)
n_nan      = (np.isnan(y_orig) | np.isnan(y_val)).sum()
n_sentinel = ((y_orig == SENTINEL) | (y_val == SENTINEL)).sum()
print(f"  Descartados NaN: {n_nan}  |  Sentinel (-99): {n_sentinel}")

y_orig = y_orig[mask]
y_val  = y_val[mask]
dist_f = cruce_filt["dist_3d"].values[mask]
print(f"Pares validos (sin NaN ni sentinel): {mask.sum()}")


# 3) Metricas
r2   = r2_score(y_orig, y_val)
rmse = np.sqrt(mean_squared_error(y_orig, y_val))
denom = np.where(y_orig != 0, y_orig, np.nan)
mape  = np.nanmean(np.abs((y_orig - y_val) / denom)) * 100
print(f"R2={r2:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

#%%
# Figura 3 paneles
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Original vs Estimacion empresa (Norte<={NORTE_MAX}) | n={len(y_orig)} | R2={r2:.3f} | RMSE={rmse:.3f} | MAPE={mape:.1f}%",
             fontsize=11, weight="bold")

# Panel 1: Scatter 1:1
mn = min(y_orig.min(), y_val.min()); mx = max(y_orig.max(), y_val.max())
sc = axes[0].scatter(y_orig, y_val, c=dist_f, cmap="YlOrRd", s=20, alpha=0.7, edgecolors="black", lw=0.3)
axes[0].plot([mn, mx], [mn, mx], "k--", lw=1.5, label="1:1")
plt.colorbar(sc, ax=axes[0], label="Distancia 3D (m)")
axes[0].set_xlabel(f"Original  {COL_ORIG}"); axes[0].set_ylabel(f"Empresa  {COL_VAL}")
axes[0].set_title("Real vs Predicho"); axes[0].legend()

# Panel 2: Histogramas superpuestos
bins = np.linspace(min(y_orig.min(), y_val.min()), max(y_orig.max(), y_val.max()), 30)
axes[1].hist(y_orig, bins=bins, color="#2563eb", alpha=0.6, edgecolor="white", label=f"Original  media={y_orig.mean():.2f}")
axes[1].hist(y_val,  bins=bins, color="#dc2626", alpha=0.6, edgecolor="white", label=f"Empresa   media={y_val.mean():.2f}")
axes[1].set_xlabel("Recuperacion en peso (%)"); axes[1].set_ylabel("Frecuencia")
axes[1].set_title("Distribuciones comparadas"); axes[1].legend()

# Panel 3: Residuos ordenados
orden = np.argsort(y_orig)
diff  = y_orig[orden] - y_val[orden]
axes[2].bar(range(len(diff)), diff, color=np.where(diff >= 0, "#2563eb", "#dc2626"), alpha=0.75, width=1.0)
axes[2].axhline(0, color="black", lw=1)
axes[2].axhline(diff.mean(), color="#d97706", lw=1.5, ls="--", label=f"Media diff={diff.mean():.2f}")
axes[2].set_xlabel("Par (ordenado)"); axes[2].set_ylabel("Diferencia (orig - empresa)")
axes[2].set_title("Residuos por par"); axes[2].legend()

plt.tight_layout()
plt.savefig("comparacion_original_vs_empresa.png", dpi=150, bbox_inches="tight")
plt.show()
print("[fig] comparacion_original_vs_empresa.png")

# CSV de comparacion
pd.DataFrame({
    "Norte": cruce_filt["Norte"].values[mask],
    "Este":  cruce_filt["Este"].values[mask],
    "Cota":  cruce_filt["Cota"].values[mask],
    "dist_3d_m": dist_f,
    COL_ORIG: y_orig,
    COL_VAL:  y_val,
    "diferencia": y_orig - y_val,
}).to_csv("comparacion_original_vs_empresa.csv", index=False)
print("[csv] comparacion_original_vs_empresa.csv")



