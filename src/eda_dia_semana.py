from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# RUTAS REPRODUCIBLES
# =========================================

# Como este script está en tesis/src/, subimos un nivel hasta tesis/
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_IMG = OUTPUT_DIR / "eda_delitos_por_dia.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"BASE_DIR:   {BASE_DIR}")
print(f"INPUT_FILE: {INPUT_FILE}")
print(f"EXISTS:     {INPUT_FILE.exists()}")

# =========================================
# CARGA
# =========================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# =========================================
# LIMPIEZA
# =========================================

df["dia"] = (
    df["dia"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str[:3]
)

orden_dias = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]

# =========================================
# CONTEO
# =========================================

conteo_dia = df["dia"].value_counts()
conteo_dia = conteo_dia.reindex(orden_dias).dropna()

# =========================================
# COLORES
# =========================================

norm = plt.Normalize(conteo_dia.min(), conteo_dia.max())
colors = plt.cm.coolwarm(norm(conteo_dia.values))

# =========================================
# GRÁFICO
# =========================================

plt.figure(figsize=(12, 6))
sns.barplot(x=conteo_dia.index, y=conteo_dia.values, palette=colors)

plt.title("Cantidad de delitos por día de la semana", fontsize=16)
plt.xlabel("Día")
plt.ylabel("Cantidad de delitos")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)

print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()