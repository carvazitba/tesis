import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from shapely.geometry import box

# =========================================
# RUTAS ABSOLUTAS
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DELITOS_PATH = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\delitos_total.csv'
# AQUÍ ESTÁ EL ARREGLO: Apuntamos al archivo limpio que vos generaste
ALOJ_PATH = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\alojamientos-geocodificados.csv'
OUTPUT_MATRIZ = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\grilla_maestra_ml.csv'

print("Cargando datasets...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
# Usamos separador de coma estándar para tu archivo generado
alojamientos = pd.read_csv(ALOJ_PATH, encoding='latin1', sep=',', low_memory=False) 

# Limpiar nombres de columnas (quita espacios invisibles y pasa a minúsculas)
delitos.columns = delitos.columns.str.strip().str.lower()
alojamientos.columns = alojamientos.columns.str.strip().str.lower()

# Forzar a numérico por si quedó algún texto y eliminar nulos
alojamientos['latitud'] = pd.to_numeric(alojamientos['latitud'], errors='coerce')
alojamientos['longitud'] = pd.to_numeric(alojamientos['longitud'], errors='coerce')

# Filtrar nulos usando tus columnas correctas
delitos = delitos.dropna(subset=['latitud', 'longitud'])
alojamientos = alojamientos.dropna(subset=['latitud', 'longitud'])

# =========================================
# PONDERACIÓN TEMPORAL
# =========================================
def asignar_peso(anio):
    if anio == 2023:
        return 1.0
    elif anio == 2022:
        return 0.75
    elif anio == 2021:
        return 0.50
    else:
        return 0.15

delitos['peso'] = delitos['anio'].apply(asignar_peso)

print("Proyectando a sistema métrico (EPSG:3857)...")
delitos_gdf = gpd.GeoDataFrame(delitos, geometry=gpd.points_from_xy(delitos['longitud'], delitos['latitud']), crs="EPSG:4326").to_crs("EPSG:3857")
# Usamos latitud y longitud del archivo geocodificado
aloj_gdf = gpd.GeoDataFrame(alojamientos, geometry=gpd.points_from_xy(alojamientos['longitud'], alojamientos['latitud']), crs="EPSG:4326").to_crs("EPSG:3857")

print("Creando grilla regular de 500x500 metros...")
xmin, ymin, xmax, ymax = delitos_gdf.total_bounds
tamaño_celda = 500

grid_cells = [box(x0, y0, x0 + tamaño_celda, y0 + tamaño_celda) 
              for x0 in np.arange(xmin, xmax, tamaño_celda) 
              for y0 in np.arange(ymin, ymax, tamaño_celda)]

grilla_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs="EPSG:3857")
grilla_gdf['id_celda'] = grilla_gdf.index
grilla_gdf['area_km2'] = grilla_gdf.area / 1e6 

print("Ejecutando cruces espaciales (Spatial Joins)...")
join_delitos = gpd.sjoin(delitos_gdf, grilla_gdf, how='inner', predicate='within')
delitos_ponderados = join_delitos.groupby('id_celda')['peso'].sum().reset_index(name='delitos_ponderados')

conteo_aloj = gpd.sjoin(aloj_gdf, grilla_gdf, how='inner', predicate='within').groupby('id_celda').size().reset_index(name='cant_alojamientos')

grilla_gdf = grilla_gdf.merge(delitos_ponderados, on='id_celda', how='left').merge(conteo_aloj, on='id_celda', how='left')
grilla_gdf[['delitos_ponderados', 'cant_alojamientos']] = grilla_gdf[['delitos_ponderados', 'cant_alojamientos']].fillna(0)

grilla_gdf['densidad_delitos'] = grilla_gdf['delitos_ponderados'] / grilla_gdf['area_km2']
grilla_gdf['densidad_alojamientos'] = grilla_gdf['cant_alojamientos'] / grilla_gdf['area_km2']

grilla_activa = grilla_gdf[(grilla_gdf['delitos_ponderados'] > 0) | (grilla_gdf['cant_alojamientos'] > 0)].copy()

grilla_activa.drop(columns=['geometry']).to_csv(OUTPUT_MATRIZ, index=False)
print(f"✅ Matriz Maestra guardada en: {OUTPUT_MATRIZ}")

print("Generando gráfico de dispersión...")
corr, p_value = stats.spearmanr(grilla_activa['densidad_alojamientos'], grilla_activa['densidad_delitos'])

plt.figure(figsize=(10, 6))
sns.regplot(data=grilla_activa, x='densidad_alojamientos', y='densidad_delitos', 
            scatter_kws={'alpha':0.5, 'color':'#3498db'}, line_kws={'color':'#e74c3c', 'linewidth':2})

plt.title(f'Relación entre Densidad de Alojamientos y Delitos\nCorrelación de Spearman: {corr:.2f} (p-value: {p_value:.3e})', fontsize=14)
plt.xlabel('Densidad de Alojamientos (por km²)', fontsize=12)
plt.ylabel('Densidad de Delitos (Ponderados por km²)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()