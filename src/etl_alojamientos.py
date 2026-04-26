from pathlib import Path
import pandas as pd

# =========================================
# RUTAS REPRODUCIBLES
# =========================================

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

ALOJ_FILE = DATA_RAW / "alojamientos_turisticos.csv"
AIRBNB_FILE = DATA_RAW / "listings.csv"

OUTPUT_FILE = DATA_PROCESSED / "alojamientos_unificados.csv"

# =========================================
# LECTURA
# =========================================

print(f"📂 Leyendo archivo: {ALOJ_FILE}")
alojamientos = pd.read_csv(
    ALOJ_FILE,
    sep=",",
    encoding="utf-8-sig",
    quotechar='"'
)

print(f"📂 Leyendo archivo: {AIRBNB_FILE}")
airbnb = pd.read_csv(
    AIRBNB_FILE,
    sep=",",
    encoding="utf-8",
    quotechar='"',
    low_memory=False
)

print("Columnas alojamientos:", alojamientos.columns.tolist())
print("Columnas Airbnb:", airbnb.columns.tolist())

# =========================================
# NORMALIZAR ALOJAMIENTOS GCBA
# =========================================

alojamientos = alojamientos.rename(columns={
    "lat": "lat",
    "long": "long",
    "Lat": "lat",
    "Long": "long",
    "latitude": "lat",
    "longitude": "long"
})

alojamientos_limpio = alojamientos[["id", "lat", "long"]].copy()
alojamientos_limpio["fuente"] = "alojamientos_turisticos"

# =========================================
# NORMALIZAR AIRBNB
# =========================================

airbnb = airbnb.rename(columns={
    "latitude": "lat",
    "longitude": "long",
    "Lat": "lat",
    "Long": "long"
})

airbnb_limpio = airbnb[["id", "lat", "long"]].copy()
airbnb_limpio["fuente"] = "airbnb_listings"

# =========================================
# UNIFICAR
# =========================================

df_final = pd.concat(
    [alojamientos_limpio, airbnb_limpio],
    ignore_index=True
)

# =========================================
# LIMPIEZA DE COORDENADAS
# =========================================

df_final["lat"] = pd.to_numeric(df_final["lat"], errors="coerce")
df_final["long"] = pd.to_numeric(df_final["long"], errors="coerce")

df_final = df_final.dropna(subset=["lat", "long"])
df_final = df_final[(df_final["lat"] != 0) & (df_final["long"] != 0)]

# Filtro aproximado CABA
df_final = df_final[
    df_final["lat"].between(-34.75, -34.50) &
    df_final["long"].between(-58.60, -58.30)
].copy()

# Eliminar duplicados exactos por coordenadas
df_final = df_final.drop_duplicates(subset=["lat", "long"]).reset_index(drop=True)

# Orden final
df_final = df_final[["id", "lat", "long", "fuente"]]

# =========================================
# EXPORTAR
# =========================================

df_final.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("✅ Proceso finalizado correctamente.")
print(f"📊 Total de registros: {len(df_final):,}")
print(f"💾 Archivo guardado en: {OUTPUT_FILE}")