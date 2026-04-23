import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point

# ===============================
# CARGA DE DATOS
# ===============================
DELITOS_PATH = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\delitos_total.csv'
CAJEROS_PATH = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\cajeros-automaticos.csv'

print("Cargando datos...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
cajeros = pd.read_csv(CAJEROS_PATH)

# Normalizar columnas
delitos.columns = delitos.columns.str.strip().str.lower()
cajeros.columns = cajeros.columns.str.strip().str.lower()

# =========================================
# FILTRADO METODOLÓGICO: SOLO ROBO Y HURTO
# =========================================
col_tipo = 'tipo_delito' if 'tipo_delito' in delitos.columns else 'tipo'
if col_tipo in delitos.columns:
    delitos[col_tipo] = delitos[col_tipo].astype(str).str.strip().str.lower()
    delitos = delitos[delitos[col_tipo].isin(['robo', 'hurto'])]
    print(f"✅ Filtro aplicado: Analizando exclusivamente 'Robos' y 'Hurtos'.")

# Limpiar coordenadas
delitos = delitos.dropna(subset=['latitud', 'longitud']).copy()
cajeros = cajeros.dropna(subset=['lat', 'long']).copy()

delitos['latitud'] = pd.to_numeric(delitos['latitud'], errors='coerce')
delitos['longitud'] = pd.to_numeric(delitos['longitud'], errors='coerce')
cajeros['lat'] = pd.to_numeric(cajeros['lat'], errors='coerce')
cajeros['long'] = pd.to_numeric(cajeros['long'], errors='coerce')

# ===============================
# PASAR A GEO
# ===============================
delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos['longitud'], delitos['latitud']),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

cajeros_gdf = gpd.GeoDataFrame(
    cajeros,
    geometry=gpd.points_from_xy(cajeros['long'], cajeros['lat']),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

# ===============================
# GENERAR ANILLOS GLOBALES
# ===============================
print("Generando buffers y calculando áreas...")
distancias = [0, 50, 100, 150]
anillos = []

for idx, cajero in cajeros_gdf.iterrows():
    punto = cajero.geometry
    for i in range(len(distancias) - 1):
        r_in = distancias[i]
        r_out = distancias[i + 1]
        
        externo = punto.buffer(r_out)
        interno = punto.buffer(r_in)
        anillo = externo.difference(interno)
        
        anillos.append({
            'id_cajero': idx,
            'anillo': i + 1,
            'area_km2': anillo.area / 1e6,
            'geometry': anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# ===============================
# SPATIAL JOIN CON DEDUPLICACIÓN
# ===============================
print("Cruzando espacialmente y eliminando superposiciones (Nearest Allocation)...")
delitos_gdf['id_delito'] = range(len(delitos_gdf))

join = gpd.sjoin(delitos_gdf, anillos_gdf, how="inner", predicate="within")

# Si un delito cae en varios anillos, priorizar el anillo más cercano (1 > 2 > 3)
join = join.sort_values(by='anillo')
join = join.drop_duplicates(subset='id_delito', keep='first')

# Contar delitos por cajero y anillo
conteos = join.groupby(['id_cajero', 'anillo']).size().reset_index(name='cantidad')

# Unir conteos a la grilla de anillos y calcular densidad
anillos_gdf = anillos_gdf.merge(conteos, on=['id_cajero', 'anillo'], how='left')
anillos_gdf['cantidad'] = anillos_gdf['cantidad'].fillna(0)
anillos_gdf['densidad'] = anillos_gdf['cantidad'] / anillos_gdf['area_km2']

# Pivotear para tener 1 fila por cajero y 3 columnas de densidades
df_res = anillos_gdf.pivot(index='id_cajero', columns='anillo', values='densidad').reset_index()
df_res.columns = ['id_cajero', 'densidad_1', 'densidad_2', 'densidad_3']

# ===============================
# CLASIFICACIÓN
# ===============================
def clasificar(row):
    dens = [row['densidad_1'], row['densidad_2'], row['densidad_3']]
    
    # Excluir cajeros que no tienen ningún delito en 150 metros
    if sum(dens) == 0:
        return 'Sin delitos'
        
    max_idx = np.argmax(dens)
    if max_idx == 0:
        return 'A (0-50m)'
    elif max_idx == 1:
        return 'B (50-100m)'
    else:
        return 'C (100-150m)'

df_res['tipo'] = df_res.apply(clasificar, axis=1)

# ===============================
# RESUMEN (Filtrando los "Sin delitos")
# ===============================
df_activos = df_res[df_res['tipo'] != 'Sin delitos']
resumen = df_activos['tipo'].value_counts().reset_index()
resumen.columns = ['tipo', 'cantidad']
resumen['porcentaje'] = resumen['cantidad'] / resumen['cantidad'].sum() * 100

print("\n📊 Clasificación de cajeros con actividad delictiva:")
print(resumen)

# ===============================
# GRÁFICO PUBLICABLE
# ===============================
# Ordenamos categóricamente para que siempre salga A, B, C en ese orden
orden = ['A (0-50m)', 'B (50-100m)', 'C (100-150m)']
resumen['tipo'] = pd.Categorical(resumen['tipo'], categories=orden, ordered=True)
resumen = resumen.sort_values('tipo')

plt.figure(figsize=(9, 6))

ax = sns.barplot(data=resumen, x='tipo', y='porcentaje', palette='Reds_r')

# Agregar etiquetas de porcentaje sobre las barras
for i, row in resumen.iterrows():
    # Obtener el índice real de la barra en el gráfico
    idx_barra = orden.index(row['tipo'])
    plt.text(idx_barra, row['porcentaje'] + 1, f"{row['porcentaje']:.1f}%", 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Clasificación de Cajeros según su Anillo de Máxima Densidad (Robos y Hurtos)', fontsize=14, pad=15)
plt.xlabel('Zona de Mayor Riesgo', fontsize=12)
plt.ylabel('Porcentaje de Cajeros (%)', fontsize=12)

# Ajustar el límite Y para que entre el texto cómodamente
plt.ylim(0, resumen['porcentaje'].max() + 10)

# Estética general
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()

plt.tight_layout()
plt.show()