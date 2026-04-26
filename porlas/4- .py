import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import os

# === Ruta y carga del archivo consolidado de delitos ===
delitos_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv'
print(f"Cargando archivo consolidado de delitos desde: {delitos_path}")
delitos_total = pd.read_csv(delitos_path, low_memory=False)

# === Limpieza de coordenadas ===
delitos_total['latitud'] = pd.to_numeric(delitos_total['latitud'], errors='coerce')
delitos_total['longitud'] = pd.to_numeric(delitos_total['longitud'], errors='coerce')
delitos_total = delitos_total.dropna(subset=['latitud', 'longitud'])
delitos_total = delitos_total[(delitos_total['latitud'] != 0) & (delitos_total['longitud'] != 0)]
print(f"Total de registros de delitos después de la limpieza: {len(delitos_total)}")

# === Asignar peso por año ===
def asignar_peso(anio):
    if anio == 2023:
        return 1.0
    elif anio == 2022:
        return 0.75
    elif anio == 2021:
        return 0.50
    else:
        return 0.15

delitos_total['peso'] = delitos_total['anio'].apply(asignar_peso)

# === Crear la grilla de CABA ===
xmin, ymin, xmax, ymax = -58.6, -34.7, -58.3, -34.5
cell_size = 0.005
x_coords = np.arange(xmin, xmax, cell_size)
y_coords = np.arange(ymin, ymax, cell_size)

polygons = []
for x in x_coords:
    for y in y_coords:
        polygons.append(Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)]))

grid = gpd.GeoDataFrame({'geometry': polygons})
grid = grid.reset_index().rename(columns={'index': 'grid_id'})

# === Convertir delitos a GeoDataFrame ===
delitos_gdf = gpd.GeoDataFrame(
    delitos_total, geometry=gpd.points_from_xy(delitos_total.longitud, delitos_total.latitud))
delitos_gdf.set_crs(epsg=4326, inplace=True)

# === Join espacial entre delitos y grilla ===
print("Calculando densidad delictiva ponderada en la grilla...")
grid.set_crs(epsg=4326, inplace=True)
joined = gpd.sjoin(grid, delitos_gdf, how="left", predicate="contains")

# === Calcular densidad delictiva por celda ===
if 'grid_id' in joined.columns:
    densidad = joined.groupby('grid_id')['peso'].sum().reset_index(name='densidad')
else:
    raise KeyError("No se encontró la columna 'grid_id' en el resultado del join.")

# === Agregar densidad al grid (convertir a float para evitar FutureWarning) ===
grid['densidad'] = 0.0
grid.loc[densidad['grid_id'], 'densidad'] = densidad['densidad'].astype(float)

# === Guardar la grilla como GeoJSON ===
grid.to_file("grid_densidad.geojson", driver='GeoJSON')
print("✅ Archivo 'grid_densidad.geojson' guardado correctamente.")
