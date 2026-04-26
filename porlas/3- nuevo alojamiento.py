# Importación de librerías necesarias
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
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

# Función para calcular densidad delictiva en una ubicación dada
def calcular_densidad_promedio(lat, lon):
    punto = Point(lon, lat)
    celda = grid[grid.contains(punto)]
    if len(celda) > 0:
        return celda['densidad'].iloc[0]
    return 0

# Cargar el modelo entrenado
model = lgb.Booster(model_file='modelo_lightgbm.txt')

# Predicción de un nuevo alojamiento con coordenadas dinámicas
latitud = float(input("Ingrese la latitud del nuevo alojamiento: "))
longitud = float(input("Ingrese la longitud del nuevo alojamiento: "))

# Calcular la densidad para las coordenadas dadas
densidad = calcular_densidad_promedio(latitud, longitud)

# Ejemplo de clúster, puedes cambiarlo si tienes un algoritmo de clustering diferente
cluster = 1

nuevo_alojamiento = np.array([[latitud, longitud, densidad, cluster]])

# Realizar la predicción
prediccion = model.predict(nuevo_alojamiento)
categoria_predicha = np.argmax(prediccion[0])
categoria = ['Muy Seguro', 'Seguro', 'Moderado', 'Riesgoso'][categoria_predicha]
print(f"\nCategoría de seguridad predicha para el nuevo alojamiento: {categoria}")
