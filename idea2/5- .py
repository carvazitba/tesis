import folium
import geopandas as gpd
import pandas as pd
import os

# === Rutas de archivos ===
grid_geojson_path = "grid_densidad.geojson"
alojamientos_csv_path = "C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/alojamientos-geocodificados.csv"
output_path = "mapa_densidad_delictiva.html"

# === Verificar existencia de archivos ===
if not os.path.exists(grid_geojson_path):
    raise FileNotFoundError(f"No se encuentra el archivo {grid_geojson_path}. Ejecutá primero el script que genera la grilla con densidad.")
if not os.path.exists(alojamientos_csv_path):
    raise FileNotFoundError(f"No se encuentra el archivo {alojamientos_csv_path}. Asegurate de que esté en la ruta correcta.")

# === Cargar grilla con densidad delictiva ===
grid = gpd.read_file(grid_geojson_path)

# === Crear mapa centrado en CABA ===
mapa = folium.Map(location=[-34.6, -58.45], zoom_start=12, tiles='cartodbpositron')

# === Capa de calor por densidad delictiva ===
folium.Choropleth(
    geo_data=grid,
    data=grid,
    columns=["grid_id", "densidad"],
    key_on="feature.properties.grid_id",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Densidad Delictiva Ponderada"
).add_to(mapa)

# === Cargar alojamientos turísticos ===
try:
    alojamientos = pd.read_csv(alojamientos_csv_path, encoding='latin1', sep=';')
except Exception as e:
    raise RuntimeError(f"Error al leer el CSV de alojamientos: {e}")

# === Mostrar columnas para verificar nombres ===
print("Columnas detectadas en el archivo de alojamientos:")
print(alojamientos.columns)

# === Asegurar que las columnas de ubicación existan ===
if 'lat' not in alojamientos.columns or 'lon' not in alojamientos.columns:
    raise KeyError("El archivo no contiene las columnas 'lat' y 'lon' necesarias para ubicar los alojamientos.")

# === Limpiar datos nulos ===
alojamientos = alojamientos.dropna(subset=['lat', 'lon'])

# === Agregar marcadores de alojamientos ===
for _, row in alojamientos.iterrows():
    nombre = row.get('establecimiento', 'Alojamiento')
    lat = row['lat']
    lon = row['lon']
    
    folium.Marker(
        location=[lat, lon],
        popup=nombre,
        icon=folium.Icon(color='green', icon='hotel', prefix='fa')
    ).add_to(mapa)

# === Guardar el mapa como HTML ===
mapa.save(output_path)
print(f"🗺️ Mapa generado correctamente: {output_path}")
