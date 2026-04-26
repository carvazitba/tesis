# =========================================
# 1 - CARGA, LIMPIEZA Y GEOCODIFICACIÓN
# =========================================

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# =========================================
# RUTAS REPRODUCIBLES
# =========================================

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

INPUT_PATH = DATA_RAW / "alojamientos-turisticos.csv"
OUTPUT_PATH = DATA_PROCESSED / "alojamientos-geocodificados.csv"
TEMP_PATH = DATA_PROCESSED / "alojamientos-geocodificados_temp.csv"
CABA_POLYGON_PATH = DATA_RAW / "comunas.geojson"

SAVE_EVERY = 50

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"INPUT:   {INPUT_PATH}")
print(f"OUTPUT:  {OUTPUT_PATH}")
print(f"TEMP:    {TEMP_PATH}")
print(f"COMUNAS: {CABA_POLYGON_PATH}")
print(f"EXISTS INPUT:   {INPUT_PATH.exists()}")
print(f"EXISTS COMUNAS: {CABA_POLYGON_PATH.exists()}")

# =========================================
# VALIDACIÓN DE ARCHIVOS
# =========================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {INPUT_PATH}")

if not CABA_POLYGON_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {CABA_POLYGON_PATH}")

# =========================================
# GEOCODIFICADOR
# =========================================

geolocator = Nominatim(user_agent="tesis_caba")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# =========================================
# CARGA DE DATOS
# =========================================

print(f"\n📂 Cargando alojamientos desde: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, encoding="latin1", delimiter=";")

df.columns = df.columns.str.strip().str.lower()

# =========================================
# CARGAR POLÍGONO DE CABA
# =========================================

print("🗺️ Cargando polígono de CABA...")

caba = gpd.read_file(CABA_POLYGON_PATH)
caba = caba.to_crs(epsg=4326)

caba_union = caba.unary_union

# =========================================
# REANUDACIÓN
# =========================================

if TEMP_PATH.exists():
    print("♻️ Archivo temporal encontrado. Reanudando...")
    df = pd.read_csv(TEMP_PATH)
else:
    df["latitud"] = None
    df["longitud"] = None

# =========================================
# FUNCIÓN DE GEOCODIFICACIÓN
# =========================================

def geocodificar(direccion):
    try:
        location = geocode(f"{direccion}, Buenos Aires, Argentina")
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"⚠️ Error geocodificando {direccion}: {e}")

    return None, None

# =========================================
# PROCESO PRINCIPAL
# =========================================

procesados = 0

for idx, row in df.iterrows():

    if pd.notna(row.get("latitud")) and pd.notna(row.get("longitud")):
        continue

    direccion = row.get("direccion", None)

    if pd.isna(direccion):
        continue

    lat, lon = geocodificar(direccion)

    if lat is not None and lon is not None:
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

    if procesados % SAVE_EVERY == 0:
        df.to_csv(TEMP_PATH, index=False)
        print(f"💾 Guardado parcial ({procesados} registros)")

# =========================================
# LIMPIEZA FINAL
# =========================================

df = df.dropna(subset=["latitud", "longitud"]).copy()

print(f"\n📊 Total geocodificados válidos: {len(df)}")

# =========================================
# GUARDADO FINAL
# =========================================

df.to_csv(OUTPUT_PATH, index=False)

if TEMP_PATH.exists():
    TEMP_PATH.unlink()

print(f"✅ Archivo final guardado en: {OUTPUT_PATH}")