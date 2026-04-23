# =========================================
# LIMPIEZA Y UNIFICACIÓN DE DELITOS
# LECTURA DESDE GITHUB + SALIDA LOCAL
# =========================================

import os
import pandas as pd
import numpy as np

# =========================
# CONFIGURACIÓN
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "delitos_total.csv")

# Base RAW de GitHub
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/carvazitba/tesis/main/dataset"

archivos_delitos = [
    'delitos_2016.xlsx',
    'delitos_2017.xlsx',
    'delitos_2018.xlsx',
    'delitos_2019.xlsx',
    'delitos_2021.xlsx',
    'delitos_2022.xlsx',
    'delitos_2023.xlsx'
]

# =========================
# FUNCIONES
# =========================

def corregir_decimal(valor):
    if pd.isna(valor):
        return np.nan

    s = str(valor).strip().replace(",", ".")

    negativo = s.startswith("-")
    if negativo:
        s = s[1:]

    s_clean = "".join(ch for ch in s if ch.isdigit() or ch == ".")

    if "." in s_clean:
        parte_entera, parte_decimal = s_clean.split(".", 1)
        if len(parte_entera) > 2:
            s_clean = parte_entera[:2] + "." + parte_entera[2:] + parte_decimal
    else:
        if len(s_clean) > 2:
            s_clean = s_clean[:2] + "." + s_clean[2:]

    if negativo:
        s_clean = "-" + s_clean

    try:
        return float(s_clean)
    except:
        return np.nan


def corregir_coordenadas(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except:
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


# =========================
# PROCESO
# =========================

datasets_limpios = []

for archivo in archivos_delitos:
    url = f"{GITHUB_RAW_BASE}/{archivo}"
    print(f"\n📂 Procesando desde GitHub: {url}")

    try:
        df = pd.read_excel(url)
    except Exception as e:
        print(f"⚠️ No se pudo cargar {archivo}: {e}")
        continue

    # Validación de columnas mínimas
    if "latitud" not in df.columns or "longitud" not in df.columns:
        print(f"⚠️ El archivo {archivo} no tiene columnas 'latitud' y 'longitud'.")
        continue

    # Paso A: conversión numérica
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    # Paso B: corrección decimal
    df["latitud"] = df["latitud"].apply(corregir_decimal)
    df["longitud"] = df["longitud"].apply(corregir_decimal)

    # Paso C: validación de rango
    df[["latitud", "longitud"]] = df.apply(
        lambda row: corregir_coordenadas(row["latitud"], row["longitud"]),
        axis=1,
        result_type="expand"
    )

    # Paso D: limpieza final
    df = df.dropna(subset=["latitud", "longitud"])

    print(f"✔ Registros válidos en {archivo}: {len(df)}")
    datasets_limpios.append(df)

# =========================
# UNIFICACIÓN FINAL
# =========================

if not datasets_limpios:
    raise ValueError("No se pudo cargar ningún archivo válido desde GitHub.")

delitos_total = pd.concat(datasets_limpios, ignore_index=True)

print("\n📊 DATASET FINAL")
print(f"Total registros: {len(delitos_total)}")
print("Columnas finales:")
print(list(delitos_total.columns))

# =========================
# GUARDADO LOCAL
# =========================

delitos_total.to_csv(OUTPUT_PATH, index=False)
print(f"\n💾 Archivo guardado en: {OUTPUT_PATH}")