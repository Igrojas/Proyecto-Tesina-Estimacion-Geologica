#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('data/raw/com_p_plt_entry_1_guardado.xlsx')
# %%

# Para cada valor único de zmin, genera un gráfico 3D diferente
zmin_unicos = np.sort(df["zmin"].unique())

for zmin_val in zmin_unicos:
    df_zmin = df[df["zmin"] == zmin_val]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Gráfico de los puntos correspondientes a este zmin
    scatter = ax.scatter(
        df_zmin["midx"], df_zmin["midy"], df_zmin["midz"],
        color='b', s=25, label=f"zmin = {zmin_val}"
    )

    ax.set_xlabel("midx")
    ax.set_ylabel("midy")
    ax.set_zlabel("midz")
    ax.set_title(f"3D scatter: midx, midy, midz (zmin = {zmin_val})")
    ax.legend()
    plt.show()

# %%

print("Zona 401")
print(f" Min: {df[df["zmin"] == 401]["midz"].min()}")
print(f" Max: {df[df["zmin"] == 401]["midz"].max()}")


print("Zona 407")
print(f" Min: {df[df["zmin"] == 407]["midz"].min()}")
print(f" Max: {df[df["zmin"] == 407]["midz"].max()}")

print("Zona 407")
print(f" Min: {df[df["zmin"] == 407]["midz"].min()}")
print(f" Max: {df[df["zmin"] == 407]["midz"].max()}")
# %%