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
OUTPUT_IMG = OUTPUT_DIR / "eda_delitos_por_mes.png"

print(f"📂 Cargando archivo desde: {INPUT_FILE}")

# =========================================
# CARGA
# =========================================

df = pd.read_csv(INPUT_FILE, low_memory=False)

# =========================================
# LIMPIEZA
# =========================================

df['mes'] = (
    df['mes']
    .astype(str)
    .str.strip()
    .str.lower()
    .str[:3]
)

orden_meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun',
               'jul', 'ago', 'sep', 'oct', 'nov', 'dic']

# =========================================
# CONTEO
# =========================================

conteo_mes = df['mes'].value_counts()
conteo_mes = conteo_mes.reindex(orden_meses).dropna()

# =========================================
# COLORES (azul → rojo)
# =========================================

norm = plt.Normalize(conteo_mes.min(), conteo_mes.max())
colors = plt.cm.coolwarm(norm(conteo_mes.values))

# =========================================
# GRÁFICO
# =========================================

plt.figure(figsize=(14, 6))
sns.barplot(x=conteo_mes.index, y=conteo_mes.values, palette=colors)

plt.title('Cantidad de delitos por mes', fontsize=16)
plt.xlabel('Mes')
plt.ylabel('Cantidad de delitos')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

# Guardar imagen
plt.savefig(OUTPUT_IMG, dpi=300)

print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()