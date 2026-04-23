import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar dataset consolidado
delitos_total = pd.read_csv('C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv')

# Contar delitos por tipo
conteo_tipo = delitos_total['tipo'].value_counts()

# Ordenar (opcional: de mayor a menor)
conteo_tipo = conteo_tipo.sort_values(ascending=False)

# Escala de colores azul → rojo
norm = plt.Normalize(conteo_tipo.min(), conteo_tipo.max())
colors = plt.cm.coolwarm(norm(conteo_tipo.values))

# Graficar
plt.figure(figsize=(12, 7))
ax = sns.barplot(x=conteo_tipo.index, y=conteo_tipo.values, palette=colors)

plt.title('Cantidad de delitos por tipo', fontsize=16)
plt.xlabel('Tipo de delito', fontsize=12)
plt.ylabel('Cantidad de delitos', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# ---- Agregar números encima de cada barra ----
for i, v in enumerate(conteo_tipo.values):
    ax.text(i, v + (v * 0.01), str(v), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
