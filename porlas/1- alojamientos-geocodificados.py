from geopy.geocoders import Nominatim #lo voy a usar para convertir las direcciones en coordenadas
from geopy.extra.rate_limiter import RateLimiter #lo voy a usar para espaciar las consultas
import pandas as pd #lo voy a usar para leer el dataset
import time #atado a la segunda línea

# Crear el geocodificador con Nominatim
geolocator = Nominatim(user_agent="alojamientos_caba") #lo voy a usar para convertir las direcciones en coordenadas
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1) # Limitar a 1 consulta por segundo

# Ruta del archivo de alojamientos
alojamientos_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/alojamientos-turisticos.csv' # Ruta del archivo CSV, luego tengo que poner la URL
print(f"Cargando archivo de alojamientos turísticos desde: {alojamientos_path}") # Cargar el archivo CSV
alojamientos = pd.read_csv(alojamientos_path, encoding='latin1', delimiter=';') # Leer el archivo CSV

# Verificar las primeras filas
print("Primeras filas de alojamientos:") # Pongo el título para que en la siguiente línea se entienda qué se está mostrando
print(alojamientos.head()) # Mostrar las primeras filas del DataFrame

# Verificar el nombre de la columna que contiene la dirección
print("Columnas del archivo de alojamientos:") # Pongo el título para que en la siguiente línea se entienda qué se está mostrando
print(alojamientos.columns) # Mostrar las columnas del DataFrame

# Crear columnas de latitud y longitud
alojamientos['latitud'] = None # se crea la columna latitud
alojamientos['longitud'] = None # se crea la columna longitud

# Límites geográficos de CABA - esto lo hago para filtrar ubicaciones por fuera de CABA
lat_min, lat_max = -34.705, -34.534 
lon_min, lon_max = -58.531, -58.350

# Función para obtener coordenadas a partir de la dirección
def obtener_coordenadas(direccion):
    try:
        location = geolocator.geocode(direccion + ", Buenos Aires, Argentina")
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Error al geocodificar: {direccion} -> {e}")
    return None, None

# Iterar sobre el DataFrame para obtener coordenadas
for idx, row in alojamientos.iterrows():
    direccion = row['direccion']  # Ajusta el nombre si es diferente
    lat, lon = obtener_coordenadas(direccion)
    # Filtrar coordenadas que estén dentro de CABA
    if lat and lon and (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max):
        alojamientos.at[idx, 'latitud'] = lat
        alojamientos.at[idx, 'longitud'] = lon
        print(f"Geocodificado: {direccion} -> ({lat}, {lon}) [CABA]")
    else:
        print(f"Descartado: {direccion} -> ({lat}, {lon}) [Fuera de CABA]")
    time.sleep(1)  # Esperar para no saturar el servicio

# Eliminar filas que no lograron obtener coordenadas o están fuera de CABA
alojamientos = alojamientos.dropna(subset=['latitud', 'longitud'])
print(f"Total de alojamientos con coordenadas dentro de CABA: {len(alojamientos)}")

# Guardar el archivo obtenido
output_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/alojamientos-geocodificados.csv'
alojamientos.to_csv(output_path, index=False)
print(f"Archivo de alojamientos con coordenadas guardado en: {output_path}")
