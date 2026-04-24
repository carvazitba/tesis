from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm

# =========================================
# RUTAS REPRODUCIBLES
# =========================================

BASE_DIR = Path(__file__).resolve().parent  # tesis/
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELITOS_PATH = DATA_PROCESSED / "delitos_total.csv.gz"
COMISARIAS_PATH = DATA_RAW / "comisarias-policia-de-la-ciudad.xlsx"

OUTPUT_HTML = OUTPUT_DIR / "mapa_comisarias_anillos_densidad_relativa_p20_p80.html"
OUTPUT_CSV = OUTPUT_DIR / "anillos_comisarias_densidad_relativa.csv"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"DELITOS: {DELITOS_PATH}")
print(f"COMISARIAS: {COMISARIAS_PATH}")

# =========================================
# CARGA DE DATOS
# =========================================

if not DELITOS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {DELITOS_PATH}")

if not COMISARIAS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {COMISARIAS_PATH}")

delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
comisarias = pd.read_excel(COMISARIAS_PATH)

# =========================================
# LIMPIEZA
# =========================================

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()

comisarias["lat"] = pd.to_numeric(comisarias["lat"], errors="coerce")
comisarias["long"] = pd.to_numeric(comisarias["long"], errors="coerce")
comisarias = comisarias.dropna(subset=["lat", "long"]).copy()

# =========================================
# GEO DATAFRAMES
# =========================================

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
)

comisarias_gdf = gpd.GeoDataFrame(
    comisarias,
    geometry=gpd.points_from_xy(comisarias["long"], comisarias["lat"]),
    crs="EPSG:4326"
)

# =========================================
# PROYECCIÓN MÉTRICA
# =========================================

delitos_m = delitos_gdf.to_crs(epsg=3857)
comisarias_m = comisarias_gdf.to_crs(epsg=3857)

# =========================================
# CREAR ANILLOS (300 m)
# =========================================

distancias = [0, 300, 600, 900, 1200]
anillos = []

for _, row in comisarias_m.iterrows():
    punto = row.geometry

    for i in range(4):
        externo = punto.buffer(distancias[i+1])
        interno = punto.buffer(distancias[i])
        anillo = externo.difference(interno)

        anillos.append({
            "id": row["id"],
            "nombre": row["nombre"],
            "direccion": row.get("direccion", ""),
            "barrio": row.get("barrio"),
            "comuna": row.get("comuna"),
            "anillo": i + 1,
            "distancia": f"{distancias[i]}-{distancias[i+1]} m",
            "geometry": anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# =========================================
# SPATIAL JOIN
# =========================================

join = gpd.sjoin(delitos_m, anillos_gdf, how="inner", predicate="within")

conteos = (
    join.groupby(["id", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(conteos, on=["id", "anillo"], how="left")
anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)

# =========================================
# DENSIDAD
# =========================================

anillos_gdf["area_km2"] = anillos_gdf.geometry.area / 1e6
anillos_gdf["densidad"] = anillos_gdf["cantidad_delitos"] / anillos_gdf["area_km2"]

# Relativa respecto al anillo 1
base = anillos_gdf[anillos_gdf["anillo"] == 1][["id", "densidad"]]
base = base.rename(columns={"densidad": "base"})

anillos_gdf = anillos_gdf.merge(base, on="id", how="left")

anillos_gdf["dens_rel"] = anillos_gdf["densidad"] / anillos_gdf["base"]

# =========================================
# EXPORT CSV
# =========================================

anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

# =========================================
# MAPA
# =========================================

anillos_wgs84 = anillos_gdf.to_crs(epsg=4326)

mapa = folium.Map(location=[-34.61, -58.43], zoom_start=12)

val = anillos_wgs84["dens_rel"].dropna()
p20, p80 = val.quantile(0.2), val.quantile(0.8)

colormap = cm.LinearColormap(
    colors=["green", "yellow", "red"],
    vmin=p20,
    vmax=p80
)
colormap.add_to(mapa)

for _, row in anillos_wgs84.iterrows():
    val = row["dens_rel"]

    color = "#ccc" if pd.isna(val) else colormap(val)

    folium.GeoJson(
        row["geometry"],
        style_function=lambda x, color=color: {
            "fillColor": color,
            "color": color,
            "weight": 1,
            "fillOpacity": 0.6
        }
    ).add_to(mapa)

mapa.save(OUTPUT_HTML)

print(f"🗺️ Mapa generado en: {OUTPUT_HTML}")