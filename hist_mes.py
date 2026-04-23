import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar dataset consolidado
delitos_total = pd.read_csv(
    'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv',
    low_memory=False
)

# Normalizar la columna 'mes'
delitos_total['mes'] = (
    delitos_total['mes']
    .astype(str)
    .str.strip()
    .str.lower()
    .str[:3]   # por si algún dato viene como 'enero', lo reduce a 'ene'
)

# Orden lógico de meses abreviados en español
orden_meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun',
               'jul', 'ago', 'sep', 'oct', 'nov', 'dic']

# Contar delitos por mes
conteo_mes = delitos_total['mes'].value_counts()

# Reordenar y quitar valores desconocidos
conteo_mes = conteo_mes.reindex(orden_meses).dropna()

# Escala: azul → rojo
norm = plt.Normalize(conteo_mes.min(), conteo_mes.max())
colors = plt.cm.coolwarm(norm(conteo_mes.values))

# Graficar
plt.figure(figsize=(14, 6))
sns.barplot(x=conteo_mes.index, y=conteo_mes.values, palette=colors)

plt.title('Cantidad de delitos por mes', fontsize=16)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Cantidad de delitos', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
