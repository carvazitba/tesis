from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import time

# Crear el geocodificador con Nominatim
geolocator = Nominatim(user_agent="alojamientos_caba")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Ruta del archivo de alojamientos
alojamientos_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/alojamientos-turisticos.csv'
print(f"Cargando archivo de alojamientos turísticos desde: {alojamientos_path}")
alojamientos = pd.read_csv(alojamientos_path, encoding='latin1', delimiter=';')

# Verificar las primeras filas
print("Primeras filas de alojamientos:")
print(alojamientos.head())

# Verificar el nombre de la columna que contiene la dirección
print("Columnas del archivo de alojamientos:")
print(alojamientos.columns)

# Crear columnas de latitud y longitud
alojamientos['latitud'] = None
alojamientos['longitud'] = None

# Límites geográficos de CABA
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

# Guardar el archivo enriquecido
output_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/alojamientos-geocodificados.csv'
alojamientos.to_csv(output_path, index=False)
print(f"Archivo de alojamientos con coordenadas guardado en: {output_path}")
