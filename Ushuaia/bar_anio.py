import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# RUTAS REPRODUCIBLES
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "delitos_total.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "bar_delitos_por_anio.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# VERIFICAR EXISTENCIA
# =========================

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"No se encontró el archivo de entrada: {INPUT_PATH}\n"
        "Primero ejecutá el script que genera 'delitos_total.csv'."
    )

# =========================
# CARGAR DATASET
# =========================

print(f"📂 Cargando dataset desde: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

# =========================
# VALIDAR COLUMNA
# =========================

if "anio" not in df.columns:
    raise KeyError("El dataset no contiene la columna 'anio'.")

# =========================
# ANÁLISIS
# =========================

conteo = df["anio"].value_counts().sort_index()
porcentajes = conteo / conteo.sum() * 100

# =========================
# VISUALIZACIÓN
# =========================

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=conteo.index.astype(str), y=conteo.values, hue=conteo.index.astype(str), legend=False)

for i, (valor, pct) in enumerate(zip(conteo.values, porcentajes)):
    ax.text(i, valor + valor * 0.01, f"{pct:.1f}%", ha="center", va="bottom", fontsize=11)

plt.title("Cantidad de delitos por año", fontsize=15)
plt.xlabel("Año")
plt.ylabel("Cantidad de delitos")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# Guardar figura
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
print(f"💾 Gráfico guardado en: {OUTPUT_PATH}")

plt.show()