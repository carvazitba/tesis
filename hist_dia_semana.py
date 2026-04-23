import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar dataset consolidado
delitos_total = pd.read_csv(
    'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv',
    low_memory=False
)

# Normalizar la columna 'dia' a minúsculas y quedarnos con las 3 primeras letras
delitos_total['dia'] = (
    delitos_total['dia']
    .astype(str)
    .str.strip()
    .str.lower()
    .str[:3]          # por si alguno viene como 'lunes', se queda en 'lun'
)

# Orden lógico de los días (abreviados en español)
orden_dias = ['lun', 'mar', 'mie', 'jue', 'vie', 'sab', 'dom']

# Contar delitos por día
conteo_dia = delitos_total['dia'].value_counts()

# Reordenar según el orden lógico y descartar posibles valores raros
conteo_dia = conteo_dia.reindex(orden_dias).dropna()

# Escala de colores azul → rojo según cantidad de delitos
norm = plt.Normalize(conteo_dia.min(), conteo_dia.max())
colors = plt.cm.coolwarm(norm(conteo_dia.values))

# Graficar
plt.figure(figsize=(12, 6))
sns.barplot(x=conteo_dia.index, y=conteo_dia.values, palette=colors)

plt.title('Cantidad de delitos por día de la semana', fontsize=16)
plt.xlabel('Día', fontsize=12)
plt.ylabel('Cantidad de delitos', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
