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

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_IMG = OUTPUT_DIR / "heatmap_dia_hora.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"INPUT_FILE: {INPUT_FILE}")
print(f"EXISTS:     {INPUT_FILE.exists()}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró: {INPUT_FILE}")

# =========================================
# CARGA Y LIMPIEZA DE DATOS
# =========================================

print("Cargando dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

df["franja"] = pd.to_numeric(df["franja"], errors="coerce")
df = df.dropna(subset=["franja", "dia"]).copy()
df["franja"] = df["franja"].astype(int)

# Mantener solo horas válidas
df = df[df["franja"].between(0, 23)].copy()

df["dia"] = (
    df["dia"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str[:3]
)

orden_dias = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]
df = df[df["dia"].isin(orden_dias)].copy()

# =========================================
# MATRIZ BIVARIADA DÍA x HORA
# =========================================

print("Generando matriz de calor...")

matriz_calor = pd.crosstab(df["dia"], df["franja"])
matriz_calor = matriz_calor.reindex(orden_dias)
matriz_calor = matriz_calor.reindex(columns=range(24), fill_value=0)

# =========================================
# HEATMAP
# =========================================

plt.figure(figsize=(14, 6))

sns.heatmap(
    matriz_calor,
    cmap="YlOrRd",
    linewidths=0.5,
    annot=False,
    cbar_kws={"label": "Cantidad de delitos"}
)

plt.title(
    "Hotspots temporales: concentración de delitos por día y hora",
    fontsize=16,
    pad=15
)
plt.xlabel("Franja horaria (00:00–23:00 hs)")
plt.ylabel("Día de la semana")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)

print(f"📊 Heatmap guardado en: {OUTPUT_IMG}")

plt.show()