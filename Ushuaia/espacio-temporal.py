import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 1. CARGA Y LIMPIEZA DE DATOS
# =========================================
ruta = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\delitos_total.csv'
print(f"Cargando dataset desde: {ruta}")
df = pd.read_csv(ruta, low_memory=False)

# Reemplazamos 'franja_horaria' por 'franja' y 'Dia' por 'dia'
# Limpiar franja (forzar numérico, eliminar nulos y pasar a entero)
df['franja'] = pd.to_numeric(df['franja'], errors='coerce')
df = df.dropna(subset=['franja', 'dia'])
df['franja'] = df['franja'].astype(int)

# Limpiar y normalizar la columna 'dia'
df['dia'] = df['dia'].astype(str).str.strip().str.lower().str[:3]

# Definir el orden lógico de la semana
orden_dias = ['lun', 'mar', 'mie', 'jue', 'vie', 'sab', 'dom']

# Filtrar para evitar cualquier valor atípico en la columna dia
df = df[df['dia'].isin(orden_dias)]

# =========================================
# 2. CREACIÓN DE LA MATRIZ BIVARIADA
# =========================================
print("Generando matriz de calor...")
matriz_calor = pd.crosstab(df['dia'], df['franja'])

# Reindexar las filas para que el gráfico empiece el lunes y termine el domingo
matriz_calor = matriz_calor.reindex(orden_dias)

# =========================================
# 3. VISUALIZACIÓN: HEATMAP
# =========================================
plt.figure(figsize=(14, 6))

ax = sns.heatmap(matriz_calor, 
                 cmap="YlOrRd", 
                 linewidths=.5, 
                 annot=False, 
                 cbar_kws={'label': 'Cantidad de Delitos'})

plt.title('Hotspots Temporales: Concentración de Delitos por Día y Hora', fontsize=16, pad=15)
plt.xlabel('Franja Horaria (00:00 - 23:00 hs)', fontsize=12)
plt.ylabel('Día de la Semana', fontsize=12)

# Rotar los días para que se lean de forma horizontal
plt.yticks(rotation=0) 

plt.tight_layout()
plt.show()