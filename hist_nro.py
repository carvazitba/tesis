import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar dataset
delitos_total = pd.read_csv(
    'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv',
    low_memory=False
)

# Convertir columna 'fecha' a datetime
delitos_total['fecha'] = pd.to_datetime(delitos_total['fecha'], errors='coerce')

# Eliminar fechas inválidas
delitos_total = delitos_total.dropna(subset=['fecha'])

# Extraer día del mes (1–31)
delitos_total['dia_mes'] = delitos_total['fecha'].dt.day

# Contar delitos por día del mes
conteo_dia_mes = delitos_total['dia_mes'].value_counts().sort_index()

# Escala azul → rojo según cantidad
norm = plt.Normalize(conteo_dia_mes.min(), conteo_dia_mes.max())
colors = plt.cm.coolwarm(norm(conteo_dia_mes.values))

# Graficar
plt.figure(figsize=(14, 6))
sns.barplot(x=conteo_dia_mes.index, y=conteo_dia_mes.values, palette=colors)

plt.title('Cantidad de delitos por día del mes', fontsize=16)
plt.xlabel('Día del mes', fontsize=12)
plt.ylabel('Cantidad de delitos', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
