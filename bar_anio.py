import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset consolidado
df = pd.read_csv('C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv')

# Contar delitos por año
conteo = df['anio'].value_counts().sort_index()  # orden natural ascendente

# Calcular porcentajes
porcentajes = conteo / conteo.sum() * 100

plt.figure(figsize=(8,5))
ax = sns.barplot(x=conteo.index.astype(str), y=conteo.values, palette="coolwarm")

# Agregar porcentaje arriba de cada barra
for i, (valor, pct) in enumerate(zip(conteo.values, porcentajes)):
    ax.text(i, valor + valor*0.01, f"{pct:.1f}%", ha='center', va='bottom', fontsize=12)

plt.title('Cantidad de delitos por año', fontsize=15)
plt.xlabel('Año')
plt.ylabel('Cantidad de delitos')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
