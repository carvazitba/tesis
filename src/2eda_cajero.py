from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELITOS_PATH = DATA_PROCESSED / "delitos_total.csv.gz"
CAJEROS_PATH = DATA_RAW / "cajeros-automaticos.csv"

OUTPUT_HTML = OUTPUT_DIR / "mapa_cajeros_3anillos_50m.html"
OUTPUT_CSV = OUTPUT_DIR / "anillos_cajeros_3anillos_50m.csv"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"DELITOS: {DELITOS_PATH}")
print(f"CAJEROS: {CAJEROS_PATH}")
print(f"EXISTS DELITOS: {DELITOS_PATH.exists()}")
print(f"EXISTS CAJEROS: {CAJEROS_PATH.exists()}")

# CARGA DE DATOS

if not DELITOS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {DELITOS_PATH}")

if not CAJEROS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {CAJEROS_PATH}")

print("📂 Cargando delitos...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)

print("📂 Cargando cajeros...")
cajeros = pd.read_csv(CAJEROS_PATH, low_memory=False)

# LIMPIEZA Y FILTRADO

delitos.columns = delitos.columns.str.strip().str.lower()
cajeros.columns = cajeros.columns.str.strip().str.lower()

# Filtrar robos y hurtos
col_tipo = 'tipo_delito' if 'tipo_delito' in delitos.columns else 'tipo'

if col_tipo in delitos.columns:
    delitos[col_tipo] = delitos[col_tipo].astype(str).str.strip().str.lower()
    delitos = delitos[delitos[col_tipo].isin(['robo', 'hurto'])]
    print(f"✅ Filtro aplicado (robo/hurto): {len(delitos)} registros")

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"])

cajeros["lat"] = pd.to_numeric(cajeros["lat"], errors="coerce")
cajeros["long"] = pd.to_numeric(cajeros["long"], errors="coerce")
cajeros = cajeros.dropna(subset=["lat", "long"])

# GEO DATAFRAMES

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
)

cajeros_gdf = gpd.GeoDataFrame(
    cajeros,
    geometry=gpd.points_from_xy(cajeros["long"], cajeros["lat"]),
    crs="EPSG:4326"
)

delitos_m = delitos_gdf.to_crs(epsg=3857)
cajeros_m = cajeros_gdf.to_crs(epsg=3857)

# CREAR ANILLOS (3 de 50m)

distancias = [0, 50, 100, 150]
anillos = []

for _, row in cajeros_m.iterrows():
    for i in range(len(distancias) - 1):
        externo = row.geometry.buffer(distancias[i+1])
        interno = row.geometry.buffer(distancias[i])
        anillo = externo.difference(interno)

        anillos.append({
            "id": row["id"],
            "banco": row.get("banco", "S/D"),
            "barrio": row.get("barrio", "S/D"),
            "anillo": i + 1,
            "distancia": f"{distancias[i]}-{distancias[i+1]} m",
            "geometry": anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# SPATIAL JOIN (sin doble conteo)

print("📍 Cruzando delitos y asignando al cajero más cercano...")

delitos_m["id_delito"] = range(len(delitos_m))

join = gpd.sjoin(delitos_m, anillos_gdf, how="inner", predicate="within")

# Ordenar por anillo → prioriza cercanía
join = join.sort_values(by="anillo")
join = join.drop_duplicates(subset="id_delito", keep="first")

conteos = (
    join.groupby(["id", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(conteos, on=["id", "anillo"], how="left")
anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)

# DENSIDADES

anillos_gdf["area_km2"] = anillos_gdf.geometry.area / 1_000_000
anillos_gdf["densidad"] = anillos_gdf["cantidad_delitos"] / anillos_gdf["area_km2"]

base = anillos_gdf[anillos_gdf["anillo"] == 1][["id", "densidad"]]
base = base.rename(columns={"densidad": "base"})

anillos_gdf = anillos_gdf.merge(base, on="id", how="left")

anillos_gdf["densidad_relativa"] = np.where(
    anillos_gdf["base"] > 0,
    anillos_gdf["densidad"] / anillos_gdf["base"],
    np.nan
)

# EXPORT CSV

anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
print(f"💾 CSV guardado en: {OUTPUT_CSV}")

# MAPA

anillos_wgs = anillos_gdf.to_crs(epsg=4326)

mapa = folium.Map(location=[-34.61, -58.43], zoom_start=12)

vals = anillos_wgs["densidad_relativa"].dropna()

p20 = vals.quantile(0.2)
p80 = vals.quantile(0.8)

p20 = min(p20, 1)
p80 = max(p80, 1)

colormap = cm.LinearColormap(
    ["green", "yellow", "red"],
    vmin=p20,
    vmax=p80
)
colormap.add_to(mapa)

def clip(v):
    if pd.isna(v):
        return None
    return max(min(v, p80), p20)

for _, row in anillos_wgs.iterrows():
    val = clip(row["densidad_relativa"])
    color = "#cccccc" if val is None else colormap(val)

    tooltip = f"""
    Anillo: {row['distancia']}<br>
    Delitos: {int(row['cantidad_delitos'])}<br>
    Densidad: {row['densidad']:.2f}<br>
    Relativa: {row['densidad_relativa']:.2f}
    """

    folium.GeoJson(
        row["geometry"].__geo_interface__,
        style_function=lambda f, c=color: {
            "fillColor": c,
            "color": c,
            "weight": 1,
            "fillOpacity": 0.6
        },
        tooltip=tooltip
    ).add_to(mapa)

for _, row in cajeros.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=3,
        color="black",
        fill=True
    ).add_to(mapa)

mapa.save(OUTPUT_HTML)
print(f"🗺️ Mapa generado en: {OUTPUT_HTML}")

# RESUMEN

print("\n📊 RESUMEN ESTADÍSTICO")

resumen = anillos_gdf.groupby("anillo").agg(
    distancia=("distancia", "first"),
    delitos_totales=("cantidad_delitos", "sum"),
    densidad_promedio=("densidad", "mean")
).reset_index()

base_global = resumen.loc[resumen["anillo"] == 1, "densidad_promedio"].values[0]
resumen["densidad_relativa_global"] = resumen["densidad_promedio"] / base_global

print(resumen.to_string(index=False, float_format="%.2f"))