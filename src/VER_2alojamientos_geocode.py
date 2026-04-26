# =========================================
# GEOCODIFICACIÓN ROBUSTA DE ALOJAMIENTOS
# =========================================

from pathlib import Path
import time
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
CABA_POLYGON_PATH = DATA_RAW / "comunas.geojson"

OUTPUT_PATH = DATA_PROCESSED / "alojamientos-geocodificados.csv"
TEMP_PATH = DATA_PROCESSED / "alojamientos-geocodificados_temp.csv"

SAVE_EVERY = 25

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"INPUT:   {INPUT_PATH}")
print(f"COMUNAS: {CABA_POLYGON_PATH}")
print(f"OUTPUT:  {OUTPUT_PATH}")
print(f"TEMP:    {TEMP_PATH}")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {INPUT_PATH}")

if not CABA_POLYGON_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {CABA_POLYGON_PATH}")

# =========================================
# CARGA DE DATOS
# =========================================

if TEMP_PATH.exists():
    print("♻️ Archivo temporal encontrado. Reanudando desde temporal...")
    df = pd.read_csv(TEMP_PATH, low_memory=False)
else:
    print("📂 Cargando alojamientos originales...")
    df = pd.read_csv(INPUT_PATH, encoding="latin1", delimiter=";", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    if "latitud" not in df.columns:
        df["latitud"] = None
    if "longitud" not in df.columns:
        df["longitud"] = None
    if "estado_geocodificacion" not in df.columns:
        df["estado_geocodificacion"] = None

# =========================================
# CARGAR POLÍGONO CABA
# =========================================

print("🗺️ Cargando polígono de CABA...")
caba = gpd.read_file(CABA_POLYGON_PATH).to_crs(epsg=4326)
caba_union = caba.unary_union

# =========================================
# GEOCODIFICADOR ROBUSTO
# =========================================

geolocator = Nominatim(
    user_agent="tesis_caba_geocoder",
    timeout=10
)

geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1.5,
    max_retries=3,
    error_wait_seconds=5,
    swallow_exceptions=True
)

def geocodificar(direccion):
    if pd.isna(direccion) or str(direccion).strip() == "":
        return None, None, "sin_direccion"

    direccion_limpia = str(direccion).strip()
    consulta = f"{direccion_limpia}, Ciudad Autónoma de Buenos Aires, Argentina"

    try:
        location = geocode(consulta)

        if location is None:
            return None, None, "no_geocodificado"

        lat, lon = location.latitude, location.longitude
        punto = Point(lon, lat)

        if not caba_union.contains(punto):
            return lat, lon, "fuera_caba"

        return lat, lon, "ok"

    except Exception as e:
        return None, None, f"error: {e}"

# =========================================
# PROCESO PRINCIPAL
# =========================================

procesados = 0
total = len(df)

print(f"📊 Total de registros: {total}")

for idx, row in df.iterrows():

    ya_tiene_coord = pd.notna(row.get("latitud")) and pd.notna(row.get("longitud"))
    estado_previo = row.get("estado_geocodificacion")

    if ya_tiene_coord or estado_previo in ["fuera_caba", "no_geocodificado", "sin_direccion"]:
        continue

    direccion = row.get("direccion", None)

    lat, lon, estado = geocodificar(direccion)

    df.at[idx, "latitud"] = lat
    df.at[idx, "longitud"] = lon
    df.at[idx, "estado_geocodificacion"] = estado

    procesados += 1

    if estado == "ok":
        print(f"✅ {idx}/{total} | {direccion} -> ({lat:.5f}, {lon:.5f})")
    else:
        print(f"⚠️ {idx}/{total} | {direccion} -> {estado}")

    if procesados % SAVE_EVERY == 0:
        df.to_csv(TEMP_PATH, index=False)
        print(f"💾 Guardado parcial: {procesados} nuevos procesados")

# =========================================
# GUARDADO FINAL
# =========================================

df.to_csv(TEMP_PATH, index=False)

df_final = df[
    (pd.notna(df["latitud"])) &
    (pd.notna(df["longitud"])) &
    (df["estado_geocodificacion"] == "ok")
].copy()

df_final.to_csv(OUTPUT_PATH, index=False)

print("===================================================")
print("PROCESO FINALIZADO")
print("===================================================")
print(f"Total registros originales: {len(df)}")
print(f"Geocodificados válidos: {len(df_final)}")
print(f"Archivo final: {OUTPUT_PATH}")
print(f"Archivo temporal conservado: {TEMP_PATH}")