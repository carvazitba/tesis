import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar dataset consolidado
delitos_total = pd.read_csv('C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv')

# Contar delitos por barrio
conteo_barrio = delitos_total['barrio'].value_counts()

# Ordenar de mayor a menor
conteo_barrio = conteo_barrio.sort_values(ascending=False)

# Escala de colores azul → rojo
norm = plt.Normalize(conteo_barrio.min(), conteo_barrio.max())
colors = plt.cm.coolwarm(norm(conteo_barrio.values))

# Graficar
plt.figure(figsize=(16, 8))
sns.barplot(x=conteo_barrio.index, y=conteo_barrio.values, palette=colors)

plt.title('Cantidad de delitos por barrio', fontsize=16)
plt.xlabel('Barrio', fontsize=12)
plt.ylabel('Cantidad de delitos', fontsize=12)
plt.xticks(rotation=75, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
