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

# NUEVA RUTA: Apuntamos al archivo de oferta gastronómica
GASTRO_PATH = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\oferta_gastronomica.xlsx'
OUTPUT_MATRIZ = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\grilla_maestra_gastro_ml.csv'

print("Cargando datasets...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)

# Leemos el archivo de Excel
gastronomia = pd.read_excel(GASTRO_PATH) 

# Limpiar nombres de columnas (quita espacios invisibles y pasa a minúsculas)
delitos.columns = delitos.columns.str.strip().str.lower()
gastronomia.columns = gastronomia.columns.str.strip().str.lower()

# Detectar dinámicamente cómo se llaman las columnas de coordenadas en el Excel
lat_col = 'latitud' if 'latitud' in gastronomia.columns else 'lat'
lon_col = 'longitud' if 'longitud' in gastronomia.columns else 'long'

# Forzar a numérico y eliminar nulos
gastronomia[lat_col] = pd.to_numeric(gastronomia[lat_col], errors='coerce')
gastronomia[lon_col] = pd.to_numeric(gastronomia[lon_col], errors='coerce')

delitos = delitos.dropna(subset=['latitud', 'longitud'])
gastronomia = gastronomia.dropna(subset=[lat_col, lon_col])

# =========================================
# PONDERACIÓN TEMPORAL DE DELITOS
# =========================================
def asignar_peso(anio):
    if anio == 2023: return 1.0
    elif anio == 2022: return 0.75
    elif anio == 2021: return 0.50
    else: return 0.15

delitos['peso'] = delitos['anio'].apply(asignar_peso)

# =========================================
# PROYECCIÓN ESPACIAL (EPSG:3857)
# =========================================
print("Proyectando a sistema métrico (EPSG:3857)...")
delitos_gdf = gpd.GeoDataFrame(
    delitos, geometry=gpd.points_from_xy(delitos['longitud'], delitos['latitud']), crs="EPSG:4326"
).to_crs("EPSG:3857")

gastro_gdf = gpd.GeoDataFrame(
    gastronomia, geometry=gpd.points_from_xy(gastronomia[lon_col], gastronomia[lat_col]), crs="EPSG:4326"
).to_crs("EPSG:3857")

# =========================================
# CREAR GRILLA REGULAR (500x500 metros)
# =========================================
print("Creando grilla regular de 500x500 metros...")
xmin, ymin, xmax, ymax = delitos_gdf.total_bounds
tamaño_celda = 500

grid_cells = [box(x0, y0, x0 + tamaño_celda, y0 + tamaño_celda) 
              for x0 in np.arange(xmin, xmax, tamaño_celda) 
              for y0 in np.arange(ymin, ymax, tamaño_celda)]

grilla_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs="EPSG:3857")
grilla_gdf['id_celda'] = grilla_gdf.index
grilla_gdf['area_km2'] = grilla_gdf.area / 1e6 

# =========================================
# CRUCES ESPACIALES (SPATIAL JOINS)
# =========================================
print("Ejecutando cruces espaciales...")
join_delitos = gpd.sjoin(delitos_gdf, grilla_gdf, how='inner', predicate='within')
delitos_ponderados = join_delitos.groupby('id_celda')['peso'].sum().reset_index(name='delitos_ponderados')

conteo_gastro = gpd.sjoin(gastro_gdf, grilla_gdf, how='inner', predicate='within').groupby('id_celda').size().reset_index(name='cant_gastronomia')

# Unir a la grilla maestra
grilla_gdf = grilla_gdf.merge(delitos_ponderados, on='id_celda', how='left').merge(conteo_gastro, on='id_celda', how='left')
grilla_gdf[['delitos_ponderados', 'cant_gastronomia']] = grilla_gdf[['delitos_ponderados', 'cant_gastronomia']].fillna(0)

# Calcular densidades
grilla_gdf['densidad_delitos'] = grilla_gdf['delitos_ponderados'] / grilla_gdf['area_km2']
grilla_gdf['densidad_gastronomia'] = grilla_gdf['cant_gastronomia'] / grilla_gdf['area_km2']

# Filtrar celdas "vacías"
grilla_activa = grilla_gdf[(grilla_gdf['delitos_ponderados'] > 0) | (grilla_gdf['cant_gastronomia'] > 0)].copy()

# Guardar matriz
grilla_activa.drop(columns=['geometry']).to_csv(OUTPUT_MATRIZ, index=False)
print(f"✅ Matriz Maestra guardada en: {OUTPUT_MATRIZ}")

# =========================================
# ANÁLISIS ESTADÍSTICO Y VISUALIZACIÓN
# =========================================
print("Generando gráfico de dispersión...")
corr, p_value = stats.spearmanr(grilla_activa['densidad_gastronomia'], grilla_activa['densidad_delitos'])

plt.figure(figsize=(10, 6))
sns.regplot(data=grilla_activa, x='densidad_gastronomia', y='densidad_delitos', 
            scatter_kws={'alpha':0.5, 'color':'#27ae60'}, line_kws={'color':'#c0392b', 'linewidth':2})

plt.title(f'Relación entre Densidad Gastronómica y Delitos\nCorrelación de Spearman: {corr:.2f} (p-value: {p_value:.3e})', fontsize=14)
plt.xlabel('Densidad de Oferta Gastronómica (por km²)', fontsize=12)
plt.ylabel('Densidad de Delitos (Ponderados por km²)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()