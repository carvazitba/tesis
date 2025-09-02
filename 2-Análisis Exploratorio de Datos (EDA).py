# Análisis Exploratorio de Datos (EDA)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import webbrowser
import os

# Configuración de gráficos
plt.style.use('ggplot')

# Crear la carpeta "EDA" si no existe
output_dir = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/EDA'
os.makedirs(output_dir, exist_ok=True)

# Cargar el archivo consolidado de delitos
delitos_total = pd.read_csv('C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv')

# Verificación rápida del DataFrame
print("Cantidad de registros en el DataFrame:", len(delitos_total))
print("Primeras filas del DataFrame:")
print(delitos_total.head())

# Verificar si el DataFrame tiene registros
if len(delitos_total) == 0:
    print("El DataFrame está vacío. Verifique los datos cargados.")
else:
    # Mapa de calor de densidad delictiva
    print("Generando el mapa de calor de densidad delictiva...")
    mapa_delitos = folium.Map(location=[-34.6083, -58.3712], zoom_start=12)

    # Usar una muestra de 60,000 puntos para optimizar el rendimiento
    sample_delitos = delitos_total.sample(min(60000, len(delitos_total)))

    # Filtrar datos con coordenadas válidas y convertirlas a flotante
    sample_delitos['latitud'] = pd.to_numeric(sample_delitos['latitud'], errors='coerce')
    sample_delitos['longitud'] = pd.to_numeric(sample_delitos['longitud'], errors='coerce')
    sample_delitos = sample_delitos.dropna(subset=['latitud', 'longitud'])
    sample_delitos = sample_delitos[(sample_delitos['latitud'] != 0) & (sample_delitos['longitud'] != 0)]

    # Filtrar el rango geográfico válido para CABA
    sample_delitos = sample_delitos[
        (sample_delitos['latitud'].between(-34.7, -34.5)) &
        (sample_delitos['longitud'].between(-58.6, -58.3))
    ]

    # Crear la lista de coordenadas para el mapa de calor
    heat_data = [[float(row['latitud']), float(row['longitud'])] for _, row in sample_delitos.iterrows()]

    print(f"Cantidad de puntos en el mapa de calor: {len(heat_data)}")

    # Verificar si la lista de puntos está vacía
    if len(heat_data) == 0:
        print("No hay puntos válidos para el mapa de calor.")
    else:
        # Definir el gradiente de colores en hexadecimal
        gradient = {
            0.1: '#0000FF',  # Azul
            0.5: '#00FF00',  # Verde
            1.0: '#FF0000'   # Rojo
        }

        # Añadir el mapa de calor con el gradiente personalizado
        HeatMap(heat_data, radius=8, blur=15, max_zoom=12, gradient=gradient).add_to(mapa_delitos)

        # Guardar el mapa
        output_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/EDA/mapa_delitos.html'
        mapa_delitos.save(output_path)
        print(f"Mapa de delitos guardado como {output_path}")

        # Abrir el mapa en el navegador automáticamente
        webbrowser.open(output_path)
