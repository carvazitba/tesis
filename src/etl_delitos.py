# ============================================================
# Pipeline reproducible de limpieza y consolidación de delitos
# Lee archivos desde: tesis/data/raw/
# Guarda salida en: tesis/data/processed/delitos_total.csv
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1) RUTAS REPRODUCIBLES
# ============================================================

# Este script debe estar ubicado en: tesis/src/
# BASE_DIR apunta a la carpeta raíz del proyecto: tesis/
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_PROCESSED / "delitos_total.csv"

print("===================================================")
print("CONFIGURACIÓN DE RUTAS")
print("===================================================")
print(f"Carpeta base del proyecto: {BASE_DIR}")
print(f"Carpeta de datos crudos:   {DATA_RAW}")
print(f"Carpeta de salida:         {DATA_PROCESSED}")
print(f"Archivo de salida:         {OUTPUT_PATH}")


# ============================================================
# 2) FUNCIÓN: CORREGIR DECIMALES MAL FORMATEADOS
# ============================================================

def corregir_decimal(valor):
    """
    Corrige coordenadas que vienen sin punto decimal o con el punto mal ubicado.
    Ejemplos:
        -3456789  -> -34.56789
        -5865432  -> -58.65432

    Devuelve float o NaN si no se puede convertir.
    """

    if pd.isna(valor):
        return np.nan

    s = str(valor).strip()
    s = s.replace(",", ".")

    negativo = s.startswith("-")
    if negativo:
        s = s[1:]

    s_clean = "".join(ch for ch in s if ch.isdigit() or ch == ".")

    if "." in s_clean:
        partes = s_clean.split(".")
        parte_entera = partes[0]
        parte_decimal = "".join(partes[1:])

        if len(parte_entera) > 2:
            s_clean = parte_entera[:2] + "." + parte_entera[2:] + parte_decimal
    else:
        if len(s_clean) > 2:
            s_clean = s_clean[:2] + "." + s_clean[2:]

    if negativo:
        s_clean = "-" + s_clean

    try:
        return float(s_clean)
    except Exception:
        return np.nan


# ============================================================
# 3) FUNCIÓN: VALIDAR / CORREGIR RANGO CABA
# ============================================================

def corregir_coordenadas(lat, lon):
    """
    Valida que las coordenadas estén dentro del rango aproximado de CABA.
    Si detecta valores de magnitud excesiva, intenta corregir dividiendo por 1e6.
    """

    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return np.nan, np.nan

    lat_min, lat_max = -34.7, -34.5
    lon_min, lon_max = -58.6, -58.3

    if not (lat_min <= lat <= lat_max):
        lat = lat / 1e6 if abs(lat) > 90 else lat

    if not (lon_min <= lon <= lon_max):
        lon = lon / 1e6 if abs(lon) > 180 else lon

    if not (lat_min <= lat <= lat_max) or not (lon_min <= lon <= lon_max):
        return np.nan, np.nan

    return lat, lon


# ============================================================
# 4) ARCHIVOS DE ENTRADA
# ============================================================

archivos_delitos = [
    "delitos_2016.xlsx",
    "delitos_2017.xlsx",
    "delitos_2018.xlsx",
    "delitos_2019.xlsx",
    "delitos_2021.xlsx",
    "delitos_2022.xlsx",
    "delitos_2023.xlsx",
]


# ============================================================
# 5) CONTADORES GLOBALES
# ============================================================

datasets_limpios = []

total_global_inicial = 0
total_global_final = 0
total_global_nan_numeric = 0
total_global_decimal_fix = 0
total_global_invalid_rango = 0


# ============================================================
# 6) LOOP DE CARGA + LIMPIEZA
# ============================================================

for archivo in archivos_delitos:

    archivo_path = DATA_RAW / archivo

    print("\n===================================================")
    print(f"CARGANDO ARCHIVO: {archivo}")
    print("===================================================")

    if not archivo_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {archivo_path}")

    df = pd.read_excel(archivo_path)

    df.columns = df.columns.str.strip().str.lower()

    if "latitud" not in df.columns or "longitud" not in df.columns:
        raise ValueError(
            f"El archivo {archivo} no contiene columnas 'latitud' y 'longitud'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    n_inicial = len(df)
    total_global_inicial += n_inicial

    print(f"Registros iniciales: {n_inicial}")

    # --------------------------------------------------------
    # Paso A: conversión a numérico
    # --------------------------------------------------------

    lat_na_antes = df["latitud"].isna().sum()
    lon_na_antes = df["longitud"].isna().sum()

    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    lat_na_despues = df["latitud"].isna().sum()
    lon_na_despues = df["longitud"].isna().sum()

    nuevos_nan_lat = lat_na_despues - lat_na_antes
    nuevos_nan_lon = lon_na_despues - lon_na_antes
    total_nan_numeric = nuevos_nan_lat + nuevos_nan_lon

    total_global_nan_numeric += total_nan_numeric

    print("[Paso A] Nuevos NaN por conversión numérica:")
    print(f"         latitud:  {nuevos_nan_lat}")
    print(f"         longitud: {nuevos_nan_lon}")

    # --------------------------------------------------------
    # Paso B: corrección automática de decimales
    # --------------------------------------------------------

    lat_antes_fix = df["latitud"].copy()
    lon_antes_fix = df["longitud"].copy()

    df["latitud"] = df["latitud"].apply(corregir_decimal)
    df["longitud"] = df["longitud"].apply(corregir_decimal)

    cambios_lat = (lat_antes_fix != df["latitud"]) & ~(
        lat_antes_fix.isna() & df["latitud"].isna()
    )
    cambios_lon = (lon_antes_fix != df["longitud"]) & ~(
        lon_antes_fix.isna() & df["longitud"].isna()
    )

    n_fix_lat = int(cambios_lat.sum())
    n_fix_lon = int(cambios_lon.sum())
    total_fix_decimal = n_fix_lat + n_fix_lon

    total_global_decimal_fix += total_fix_decimal

    print("[Paso B] Corrección automática de decimales:")
    print(f"         latitud corregidas:  {n_fix_lat}")
    print(f"         longitud corregidas: {n_fix_lon}")

    # --------------------------------------------------------
    # Paso C: validación/corrección por rango CABA
    # --------------------------------------------------------

    df[["lat_corr", "lon_corr"]] = df.apply(
        lambda row: corregir_coordenadas(row["latitud"], row["longitud"]),
        axis=1,
        result_type="expand",
    )

    invalidos_rango = int(df["lat_corr"].isna().sum())
    total_global_invalid_rango += invalidos_rango

    print(f"[Paso C] Coordenadas fuera de rango: {invalidos_rango}")

    df["latitud"] = df["lat_corr"]
    df["longitud"] = df["lon_corr"]
    df.drop(columns=["lat_corr", "lon_corr"], inplace=True)

    # --------------------------------------------------------
    # Paso D: eliminar registros sin coordenadas válidas
    # --------------------------------------------------------

    n_antes_drop = len(df)

    df = df.dropna(subset=["latitud", "longitud"]).copy()

    n_final = len(df)
    filtrados_total = n_inicial - n_final

    total_global_final += n_final

    print("[Paso D] Drop NaN finales:")
    print(f"         registros antes drop: {n_antes_drop}")
    print(f"         registros finales:    {n_final}")
    print(f"         filtrados totales:    {filtrados_total}")

    datasets_limpios.append(df)


# ============================================================
# 7) UNIFICACIÓN FINAL
# ============================================================

if not datasets_limpios:
    raise ValueError("No se cargó ningún dataset válido.")

delitos_total = pd.concat(datasets_limpios, ignore_index=True)


# ============================================================
# 8) RESUMEN GLOBAL
# ============================================================

print("\n\n===================================================")
print("RESUMEN GLOBAL")
print("===================================================")
print(f"Total registros iniciales: {total_global_inicial}")
print(f"Total nuevos NaN por conversión numérica: {total_global_nan_numeric}")
print(f"Total correcciones de decimales aplicadas: {total_global_decimal_fix}")
print(f"Total coordenadas fuera de rango: {total_global_invalid_rango}")
print(f"Total registros finales limpios: {len(delitos_total)}")

print("\nColumnas del dataset final:")
print(list(delitos_total.columns))


# ============================================================
# 9) GUARDAR CSV FINAL
# ============================================================

OUTPUT_GZ = OUTPUT_PATH.with_suffix(".csv.gz")

delitos_total.to_csv(OUTPUT_GZ, index=False, compression='gzip')

print(f"\n📁 Archivo comprimido generado en: {OUTPUT_GZ}")

print("\n===================================================")
print("PROCESO FINALIZADO")
print("===================================================")
print(f"Archivo CSV generado en: {OUTPUT_PATH}")