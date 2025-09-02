# Análisis Exploratorio de Datos (EDA) con Folium
import pandas as pd
import folium
from folium.plugins import HeatMap
import os

# Crear la carpeta "EDA" si no existe
output_dir = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/EDA'
os.makedirs(output_dir, exist_ok=True)

# Cargar el archivo consolidado de delitos
delitos_total = pd.read_csv('C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv')

# Filtrar datos con coordenadas válidas
delitos_total['latitud'] = pd.to_numeric(delitos_total['latitud'], errors='coerce')
delitos_total['longitud'] = pd.to_numeric(delitos_total['longitud'], errors='coerce')
delitos_total = delitos_total.dropna(subset=['latitud', 'longitud'])
delitos_total = delitos_total[(delitos_total['latitud'] != 0) & (delitos_total['longitud'] != 0)]

# Obtener los tipos de delitos únicos
tipos_delitos = delitos_total['tipo'].unique()

# Generar un mapa de calor para cada tipo de delito
for tipo in tipos_delitos:
    print(f"Generando mapa de calor para el delito: {tipo}")

    # Filtrar el dataset por el tipo de delito
    df_tipo = delitos_total[delitos_total['tipo'] == tipo]

    # Verificar si hay datos suficientes para generar el mapa
    if df_tipo.empty:
        print(f"No hay datos para el tipo de delito: {tipo}")
        continue

    # Crear el mapa base
    mapa_delitos = folium.Map(location=[-34.6083, -58.3712], zoom_start=12)

    # Crear la lista de coordenadas para el mapa de calor
    heat_data = []
    for _, row in df_tipo.iterrows():
        try:
            lat = float(row['latitud'])
            lon = float(row['longitud'])
            if not (lat >= -34.7 and lat <= -34.5 and lon >= -58.6 and lon <= -58.3):
                continue
            heat_data.append([lat, lon])
        except (ValueError, TypeError):
            continue

    print(f"Cantidad de puntos en el mapa de calor ({tipo}): {len(heat_data)}")

    # Verificar si la lista de puntos está vacía
    if len(heat_data) == 0:
        print(f"No hay puntos válidos para el mapa de calor del tipo de delito: {tipo}")
        continue

    # Añadir el mapa de calor sin gradiente personalizado
    HeatMap(heat_data, radius=8, blur=15, max_zoom=12).add_to(mapa_delitos)

    # Guardar el mapa con el nombre del tipo de delito
    output_path = f'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/EDA/mapa_delitos_{tipo}.html'
    mapa_delitos.save(output_path)
    print(f"Mapa de calor guardado como {output_path}")

print("Mapas de calor generados para todos los tipos de delitos.")
