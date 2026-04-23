# Importación de librerías necesarias
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1) FUNCIÓN: corregir decimales mal puestos
#    Regla: si hay más de 2 dígitos antes del decimal,
#    se inserta el punto luego de los 2 primeros.
#    Ej: -3456789 -> -34.56789
#        -5865432 -> -58.65432
# ------------------------------------------------------------
def corregir_decimal(valor):
    """
    Si la coordenada viene sin punto (o con el punto corrido)
    y tiene más de 2 dígitos antes del decimal, inserta el punto
    después de los dos primeros dígitos (ignorando el signo).

    Devuelve float o NaN si no se puede convertir.
    """
    if pd.isna(valor):
        return np.nan

    s = str(valor).strip()

    # Normalizar coma decimal si viniera con coma
    s = s.replace(",", ".")

    # Detectar signo
    negativo = s.startswith('-')
    if negativo:
        s = s[1:]

    # Si tiene más de un punto raro, lo limpiamos
    # (dejamos solo dígitos y un posible punto)
    # pero si es todo dígitos, lo tratamos como entero largo
    s_clean = "".join(ch for ch in s if (ch.isdigit() or ch == '.'))

    # Si es tipo "3456789" (solo dígitos) o "3456.789" con punto mal,
    # contamos dígitos antes del punto:
    if "." in s_clean:
        parte_entera = s_clean.split(".")[0]
        parte_decimal = s_clean.split(".")[1]
        if len(parte_entera) > 2:
            # Re-armar corriendo punto: tomar primeros 2 dígitos como entero
            s_clean = parte_entera[:2] + "." + parte_entera[2:] + parte_decimal
    else:
        # No tiene punto: si tiene más de 2 dígitos, insertarlo
        if len(s_clean) > 2:
            s_clean = s_clean[:2] + "." + s_clean[2:]

    if negativo:
        s_clean = "-" + s_clean

    try:
        return float(s_clean)
    except:
        return np.nan


# ------------------------------------------------------------
# 2) FUNCIÓN: validar / corregir rango CABA
# ------------------------------------------------------------
def corregir_coordenadas(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except:
        return np.nan, np.nan

    # Rango válido para CABA
    lat_min, lat_max = -34.7, -34.5
    lon_min, lon_max = -58.6, -58.3

    # Si están fuera de rango, intentar corrección típica /1e6
    if not (lat_min <= lat <= lat_max):
        lat = lat / 1e6 if abs(lat) > 90 else lat
    if not (lon_min <= lon <= lon_max):
        lon = lon / 1e6 if abs(lon) > 180 else lon

    # Si todavía fuera de rango -> NaN
    if not (lat_min <= lat <= lat_max) or not (lon_min <= lon <= lon_max):
        return np.nan, np.nan

    return lat, lon


# ------------------------------------------------------------
# 3) ARCHIVOS
# ------------------------------------------------------------
archivos_delitos = [
    'delitos_2016.xlsx', 'delitos_2017.xlsx', 'delitos_2018.xlsx',
    'delitos_2019.xlsx', 'delitos_2021.xlsx', 'delitos_2022.xlsx', 'delitos_2023.xlsx'
]

datasets_limpios = []

# Resumen global
total_global_inicial = 0
total_global_final = 0
total_global_nan_numeric = 0
total_global_decimal_fix = 0
total_global_invalid_rango = 0


# ------------------------------------------------------------
# 4) LOOP DE CARGA + LIMPIEZA CON CONTADORES
# ------------------------------------------------------------
for archivo in archivos_delitos:

    print("\n===================================================")
    print(f"CARGANDO ARCHIVO: {archivo}")
    print("===================================================")

    df = pd.read_excel(f'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/{archivo}')
    n_inicial = len(df)
    total_global_inicial += n_inicial
    print(f"Registros iniciales: {n_inicial}")

    # ---------- Paso A: conversión a numérico (coerce) ----------
    lat_na_antes = df['latitud'].isna().sum()
    lon_na_antes = df['longitud'].isna().sum()

    df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
    df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')

    lat_na_despues = df['latitud'].isna().sum()
    lon_na_despues = df['longitud'].isna().sum()

    nuevos_nan_lat = lat_na_despues - lat_na_antes
    nuevos_nan_lon = lon_na_despues - lon_na_antes
    total_nan_numeric = nuevos_nan_lat + nuevos_nan_lon
    total_global_nan_numeric += total_nan_numeric

    print(f"[Paso A] Nuevos NaN por conversión numérica:")
    print(f"         latitud: {nuevos_nan_lat}")
    print(f"         longitud: {nuevos_nan_lon}")

    # ---------- Paso B: corrección de decimales ----------
    lat_antes_fix = df['latitud'].copy()
    lon_antes_fix = df['longitud'].copy()

    df['latitud'] = df['latitud'].apply(corregir_decimal)
    df['longitud'] = df['longitud'].apply(corregir_decimal)

    # Contar cuántos cambiaron por la corrección
    cambios_lat = (lat_antes_fix != df['latitud']) & ~(lat_antes_fix.isna() & df['latitud'].isna())
    cambios_lon = (lon_antes_fix != df['longitud']) & ~(lon_antes_fix.isna() & df['longitud'].isna())

    n_fix_lat = cambios_lat.sum()
    n_fix_lon = cambios_lon.sum()
    total_fix_decimal = n_fix_lat + n_fix_lon
    total_global_decimal_fix += total_fix_decimal

    print(f"[Paso B] Corrección automática de decimales:")
    print(f"         latitud corregidas: {n_fix_lat}")
    print(f"         longitud corregidas: {n_fix_lon}")

    # ---------- Paso C: validación/corrección por rango CABA ----------
    df[['lat_corr', 'lon_corr']] = df.apply(
        lambda row: corregir_coordenadas(row['latitud'], row['longitud']),
        axis=1, result_type="expand"
    )

    invalidos_rango = df['lat_corr'].isna().sum()
    total_global_invalid_rango += invalidos_rango

    print(f"[Paso C] Coordenadas fuera de rango (quedan NaN): {invalidos_rango}")

    # Reemplazar por las corregidas y filtrar NaN
    df['latitud'] = df['lat_corr']
    df['longitud'] = df['lon_corr']
    df.drop(columns=['lat_corr', 'lon_corr'], inplace=True)

    n_antes_drop = len(df)
    df = df.dropna(subset=['latitud', 'longitud'])
    n_final = len(df)

    filtrados_total = n_inicial - n_final
    total_global_final += n_final

    print(f"[Paso D] Drop NaN finales:")
    print(f"         registros antes drop: {n_antes_drop}")
    print(f"         registros finales: {n_final}")
    print(f"         filtrados totales en archivo: {filtrados_total}")

    datasets_limpios.append(df)


# ------------------------------------------------------------
# 5) UNIFICACIÓN FINAL + RESUMEN GLOBAL
# ------------------------------------------------------------
delitos_total = pd.concat(datasets_limpios, ignore_index=True)

print("\n\n===================================================")
print("RESUMEN GLOBAL")
print("===================================================")
print(f"Total registros iniciales (todos los archivos): {total_global_inicial}")
print(f"Total nuevos NaN por conversión numérica (Paso A): {total_global_nan_numeric}")
print(f"Total correcciones de decimales aplicadas (Paso B): {total_global_decimal_fix}")
print(f"Total fuera de rango luego de corrección (Paso C): {total_global_invalid_rango}")
print(f"Total registros finales limpios: {len(delitos_total)}")

print("\nColumnas del dataset final:")
print(list(delitos_total.columns))

# Guardar CSV final
delitos_total.to_csv(
    'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv',
    index=False
)
print("\nArchivo CSV generado: delitos_total.csv")
