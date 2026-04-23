import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar dataset consolidado
delitos_total = pd.read_csv('C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv')

# Contar delitos por franja
conteo_franjas = delitos_total['franja'].value_counts().sort_index()

# Convertir los labels del eje X a enteros
franjas_enteras = conteo_franjas.index.astype(float).astype(int).astype(str)

# Crear la escala de colores: azul (mínimo) → rojo (máximo)
norm = plt.Normalize(conteo_franjas.min(), conteo_franjas.max())
colors = plt.cm.coolwarm(norm(conteo_franjas.values))

# Graficar
plt.figure(figsize=(10,6))
sns.barplot(x=franjas_enteras, y=conteo_franjas.values, palette=colors)

plt.title('Cantidad de delitos por franja horaria', fontsize=14)
plt.xlabel('Franja horaria', fontsize=12)
plt.ylabel('Cantidad de delitos', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()
