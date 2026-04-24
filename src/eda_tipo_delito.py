from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# RUTAS REPRODUCIBLES
# =========================================

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv"
OUTPUT_IMG = OUTPUT_DIR / "eda_delitos_por_tipo.png"

print(f"📂 Cargando archivo desde: {INPUT_FILE}")

# =========================================
# CARGA
# =========================================

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"

df = pd.read_csv(INPUT_FILE)

# =========================================
# LIMPIEZA
# =========================================

df["tipo"] = df["tipo"].astype(str).str.strip()

# Opcional: quedarte con tipos válidos (evita NaN/raros)
df = df[df["tipo"].notna() & (df["tipo"] != "")]

# =========================================
# CONTEO
# =========================================

conteo_tipo = df["tipo"].value_counts().sort_values(ascending=False)

# =========================================
# COLORES (azul → rojo)
# =========================================

norm = plt.Normalize(conteo_tipo.min(), conteo_tipo.max())
colors = plt.cm.coolwarm(norm(conteo_tipo.values))

# =========================================
# GRÁFICO
# =========================================

plt.figure(figsize=(12, 7))
ax = sns.barplot(x=conteo_tipo.index, y=conteo_tipo.values, palette=colors)

plt.title("Cantidad de delitos por tipo", fontsize=16)
plt.xlabel("Tipo de delito")
plt.ylabel("Cantidad de delitos")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Etiquetas numéricas
for i, v in enumerate(conteo_tipo.values):
    ax.text(i, v + (v * 0.01), str(v), ha="center", va="bottom", fontsize=10)

plt.tight_layout()

# Guardar imagen
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()