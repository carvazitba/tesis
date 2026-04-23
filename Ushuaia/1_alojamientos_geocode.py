# =========================================
# 1 - CARGA, LIMPIEZA Y GEOCODIFICACIÓN
# =========================================

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os

# =========================
# RUTAS DINÁMICAS (CLAVE)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "alojamientos-turisticos.csv"))
OUTPUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "alojamientos-geocodificados.csv"))
TEMP_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "alojamientos-geocodificados_temp.csv"))
CABA_POLYGON_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset", "comunas.geojson"))

SAVE_EVERY = 50

# =========================
# GEOCODIFICADOR
# =========================

geolocator = Nominatim(user_agent="tesis_caba")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# =========================
# CARGA DE DATOS
# =========================

print(f"📂 Cargando alojamientos desde: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, encoding="latin1", delimiter=";")

# =========================
# CARGAR POLÍGONO DE CABA
# =========================

print("🗺️ Cargando polígono de CABA...")

caba = gpd.read_file(CABA_POLYGON_PATH)
caba = caba.to_crs(epsg=4326)

caba_union = caba.unary_union

# =========================
# REANUDACIÓN
# =========================

if os.path.exists(TEMP_PATH):
    print("♻️ Archivo temporal encontrado. Reanudando...")
    df = pd.read_csv(TEMP_PATH)
else:
    df["latitud"] = None
    df["longitud"] = None

# =========================
# FUNCIÓN DE GEOCODIFICACIÓN
# =========================

def geocodificar(direccion):
    try:
        location = geocode(f"{direccion}, Buenos Aires, Argentina")
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"⚠️ Error geocodificando {direccion}: {e}")
    return None, None

# =========================
# PROCESO PRINCIPAL
# =========================

procesados = 0

for idx, row in df.iterrows():

    # si ya tiene coordenadas, lo salto
    if pd.notna(row.get("latitud")):
        continue

    direccion = row.get("direccion", None)

    if pd.isna(direccion):
        continue

    lat, lon = geocodificar(direccion)

    if lat and lon:
        punto = Point(lon, lat)

        if caba_union.contains(punto):
            df.at[idx, "latitud"] = lat
            df.at[idx, "longitud"] = lon
            print(f"✅ {direccion} -> ({lat:.5f}, {lon:.5f})")
        else:
            print(f"🚫 Fuera de CABA: {direccion}")
    else:
        print(f"❌ No geocodificado: {direccion}")

    procesados += 1

    # guardado incremental
    if procesados % SAVE_EVERY == 0:
        df.to_csv(TEMP_PATH, index=False)
        print(f"💾 Guardado parcial ({procesados} registros)")

# =========================
# LIMPIEZA FINAL
# =========================

df = df.dropna(subset=["latitud", "longitud"])

print(f"📊 Total geocodificados válidos: {len(df)}")

# =========================
# GUARDADO FINAL
# =========================

df.to_csv(OUTPUT_PATH, index=False)

if os.path.exists(TEMP_PATH):
    os.remove(TEMP_PATH)

print(f"✅ Archivo final guardado en: {OUTPUT_PATH}")