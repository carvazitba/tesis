# Importación de librerías necesarias
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium
from folium.plugins import HeatMap
import os

# Crear la carpeta de salida si no existe
output_dir = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/EDA'
os.makedirs(output_dir, exist_ok=True)

# Ruta corregida del archivo consolidado de delitos
delitos_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv'

# Carga del archivo consolidado de delitos con manejo de tipos mixtos
print(f"Cargando archivo consolidado de delitos desde: {delitos_path}")
delitos_total = pd.read_csv(delitos_path, low_memory=False)

# Limpieza y verificación de datos de delitos
delitos_total['latitud'] = pd.to_numeric(delitos_total['latitud'], errors='coerce')
delitos_total['longitud'] = pd.to_numeric(delitos_total['longitud'], errors='coerce')
delitos_total = delitos_total.dropna(subset=['latitud', 'longitud'])
delitos_total = delitos_total[(delitos_total['latitud'] != 0) & (delitos_total['longitud'] != 0)]
print(f"Total de registros de delitos después de la limpieza: {len(delitos_total)}")

# Asignación de ponderación a los delitos según el año
def asignar_peso(anio):
    if anio == 2023:
        return 1.0
    elif anio == 2022:
        return 0.75
    elif anio == 2021:
        return 0.50
    else:
        return 0.15

# Asignar el peso a cada delito
delitos_total['peso'] = delitos_total['anio'].apply(asignar_peso)

# Cargar los datos de alojamientos turísticos con manejo de encoding y delimitador
alojamientos_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/alojamientos-geocodificados.csv'
print(f"Cargando archivo de alojamientos turísticos desde: {alojamientos_path}")

try:
    alojamientos = pd.read_csv(alojamientos_path, encoding='latin1', delimiter=',')
    print(f"Total de registros de alojamientos (coma): {len(alojamientos)}")
    print("Columnas del archivo de alojamientos:")
    print(alojamientos.columns)
except Exception as e:
    print(f"Error al cargar el archivo de alojamientos: {e}")
    exit()

# Filtrar y limpiar datos de alojamientos
alojamientos['latitud'] = pd.to_numeric(alojamientos['latitud'], errors='coerce')
alojamientos['longitud'] = pd.to_numeric(alojamientos['longitud'], errors='coerce')
alojamientos = alojamientos.dropna(subset=['latitud', 'longitud'])
print(f"Total de registros de alojamientos después de la limpieza: {len(alojamientos)}")

# Clustering con DBSCAN para agrupar alojamientos cercanos
print("Realizando clustering de alojamientos con DBSCAN...")
coords = alojamientos[['latitud', 'longitud']].values
db = DBSCAN(eps=0.001, min_samples=5).fit(coords)
alojamientos['cluster'] = db.labels_
print(f"Cantidad de clústeres de alojamientos encontrados: {alojamientos['cluster'].nunique()}")

# Crear la grilla para calcular densidad delictiva
xmin, ymin, xmax, ymax = -58.6, -34.7, -58.3, -34.5
cell_size = 0.005
x_coords = np.arange(xmin, xmax, cell_size)
y_coords = np.arange(ymin, ymax, cell_size)

# Generar polígonos de la grilla
polygons = []
for x in x_coords:
    for y in y_coords:
        polygons.append(Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)]))

grid = gpd.GeoDataFrame({'geometry': polygons})

# Convertir puntos de delitos a geodataframe
delitos_gdf = gpd.GeoDataFrame(
    delitos_total, geometry=gpd.points_from_xy(delitos_total.longitud, delitos_total.latitud))

# Asegurar que el índice del geodataframe esté reseteado
grid = grid.reset_index()
grid = grid.rename(columns={'index': 'grid_id'})

# Unir la grilla con los delitos para contar la cantidad en cada celda ponderada
print("Calculando densidad delictiva ponderada en la grilla...")
joined = gpd.sjoin(grid, delitos_gdf, how="left", predicate="contains")

# Calcular densidad agrupando por la columna correcta y sumando el peso
if 'grid_id' in joined.columns:
    densidad = joined.groupby('grid_id')['peso'].sum().reset_index(name='densidad')
else:
    print("Error: No se encontró la columna 'grid_id' en el resultado del join.")
    exit()

# Añadir la densidad al geodataframe de la grilla
grid['densidad'] = 0
grid.loc[densidad['grid_id'], 'densidad'] = densidad['densidad']

# Asignar la densidad ponderada promedio a cada alojamiento
def calcular_densidad_promedio(lat, lon):
    punto = Point(lon, lat)
    celda = grid[grid.contains(punto)]
    if len(celda) > 0:
        return celda['densidad'].iloc[0]
    return 0

# Calcular la densidad ponderada para cada alojamiento
alojamientos['densidad'] = alojamientos.apply(lambda row: calcular_densidad_promedio(row['latitud'], row['longitud']), axis=1)

# Eliminar alojamientos con densidad igual a 0
alojamientos = alojamientos[alojamientos['densidad'] > 1]

# Clasificación de seguridad de alojamientos con 4 categorías
def clasificar_seguridad(densidad):
    if densidad < 250:
        return 'Muy Seguro'
    elif densidad < 500:
        return 'Seguro'
    elif densidad < 2500:
        return 'Moderado'
    else:
        return 'Riesgoso'

alojamientos['seguridad'] = alojamientos['densidad'].apply(clasificar_seguridad)

# Guardar el archivo con la columna de seguridad
output_csv_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/alojamientos-geocodificados-2.csv'
alojamientos.to_csv(output_csv_path, index=False)
print(f"Archivo con seguridad generado en: {output_csv_path}")

# Verificar el conteo de cada categoría
print("Conteo de alojamientos por categoría de seguridad:")
print(alojamientos['seguridad'].value_counts())
