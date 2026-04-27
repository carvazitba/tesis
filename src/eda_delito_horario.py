from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

# Como este script está en tesis/src/, subimos un nivel hasta tesis/
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_IMG = OUTPUT_DIR / "histograma_franja.png"

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

df["franja"] = pd.to_numeric(df["franja"], errors="coerce")
df = df.dropna(subset=["franja"]).copy()
df["franja"] = df["franja"].astype(int)

# Mantener solo horas válidas 0–23
df = df[df["franja"].between(0, 23)].copy()

# CONTEO

conteo = df["franja"].value_counts().sort_index()

# COLORES

norm = plt.Normalize(conteo.min(), conteo.max())
colors = plt.cm.coolwarm(norm(conteo.values))

# GRÁFICO

plt.figure(figsize=(12, 6))
sns.barplot(x=conteo.index.astype(str), y=conteo.values, palette=colors)

plt.title("Cantidad de delitos por franja horaria", fontsize=14)
plt.xlabel("Hora del día (0–23)")
plt.ylabel("Cantidad de delitos")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(OUTPUT_IMG, dpi=300)
print(f"\n📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()