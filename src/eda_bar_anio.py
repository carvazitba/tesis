from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv"
OUTPUT_IMG = OUTPUT_DIR / "bar_delitos_por_anio.png"

print(f"📂 Cargando archivo desde: {INPUT_FILE}")

# CARGA

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"

df = pd.read_csv(INPUT_FILE)

# AGRUPACIÓN

conteo = df['anio'].value_counts().sort_index()
porcentajes = conteo / conteo.sum() * 100

# GRÁFICO

plt.figure(figsize=(8,5))
ax = sns.barplot(x=conteo.index.astype(str), y=conteo.values)

# Etiquetas %
for i, (valor, pct) in enumerate(zip(conteo.values, porcentajes)):
    ax.text(i, valor + valor*0.01, f"{pct:.1f}%", ha='center', va='bottom', fontsize=11)

plt.title('Cantidad de delitos por año', fontsize=14)
plt.xlabel('Año')
plt.ylabel('Cantidad de delitos')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# Guardar imagen
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()