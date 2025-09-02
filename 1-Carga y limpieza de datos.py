# Importación de librerías necesarias
import pandas as pd
import numpy as np

# Función para corregir coordenadas geográficas
def corregir_coordenadas(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return np.nan, np.nan

    # Rango válido para CABA
    lat_min, lat_max = -34.7, -34.5
    lon_min, lon_max = -58.6, -58.3
    
    # Si la latitud o longitud están fuera de rango, intentar corregir
    if not (lat_min <= lat <= lat_max):
        lat = lat / 1e6 if abs(lat) > 90 else lat
    if not (lon_min <= lon <= lon_max):
        lon = lon / 1e6 if abs(lon) > 180 else lon
    
    # Si después de la corrección sigue fuera de rango, retornar NaN
    if not (lat_min <= lat <= lat_max) or not (lon_min <= lon <= lon_max):
        return np.nan, np.nan
    return lat, lon

# Lista de archivos de delitos
archivos_delitos = [
    'delitos_2016.xlsx', 'delitos_2017.xlsx', 'delitos_2018.xlsx',
    'delitos_2019.xlsx', 'delitos_2021.xlsx', 'delitos_2022.xlsx', 'delitos_2023.xlsx'
]

# Lista para almacenar los dataframes limpios
datasets_limpios = []

# Carga y limpieza de cada archivo
for archivo in archivos_delitos:
    print(f"Cargando y limpiando: {archivo}")
    df = pd.read_excel(f'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/{archivo}')  # Ruta corregida
    
    # Conversión explícita a float
    df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
    df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')
    
    # Corrección de coordenadas geográficas
    df[['latitud', 'longitud']] = df.apply(
        lambda row: corregir_coordenadas(row['latitud'], row['longitud']), axis=1, result_type="expand")
    
    # Eliminación de registros con coordenadas NaN
    df = df.dropna(subset=['latitud', 'longitud'])
    
    # Agregar el dataframe limpio a la lista
    datasets_limpios.append(df)

# Unificación de todos los datasets limpios en un solo DataFrame
delitos_total = pd.concat(datasets_limpios, ignore_index=True)

# Verificación del dataset final
print("Número total de delitos después de la limpieza:", delitos_total.shape[0])
print("Columnas del dataset final:", delitos_total.columns)

# Guardar el DataFrame limpio en un archivo CSV
delitos_total.to_csv('C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv', index=False)
print("Archivo CSV generado: delitos_total.csv")
