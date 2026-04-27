from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parent  # porque este script está en tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_IMG = OUTPUT_DIR / "hist_delitos_barrio.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"BASE_DIR:   {BASE_DIR}")
print(f"INPUT_FILE: {INPUT_FILE}")
print(f"EXISTS:     {INPUT_FILE.exists()}")

# CARGA

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# LIMPIEZA

df["barrio"] = df["barrio"].astype(str).str.strip()

# CONTEO

conteo = df["barrio"].value_counts().sort_values(ascending=False)

# COLORES

norm = plt.Normalize(conteo.min(), conteo.max())
colors = plt.cm.coolwarm(norm(conteo.values))

# GRÁFICO

plt.figure(figsize=(16, 8))
sns.barplot(x=conteo.index, y=conteo.values, palette=colors)

plt.title("Cantidad de delitos por barrio", fontsize=16)
plt.xlabel("Barrio")
plt.ylabel("Cantidad de delitos")
plt.xticks(rotation=75, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(OUTPUT_IMG, dpi=300)
print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()