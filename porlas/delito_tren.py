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
ESTACIONES_PATH = DATA_RAW / "estaciones-de-ferrocarril.csv"
BARRIOS_PATH = DATA_RAW / "barrios.csv"

OUTPUT_HTML = OUTPUT_DIR / "mapa_ferrocarril_caba.html"
OUTPUT_CSV = OUTPUT_DIR / "anillos_ferrocarril_caba.csv"

# CARGA DE DATOS

print("📂 Cargando delitos...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)

print("📂 Cargando estaciones...")
estaciones = pd.read_csv(ESTACIONES_PATH, low_memory=False)

print("📂 Cargando barrios...")
barrios = pd.read_csv(BARRIOS_PATH)

# LIMPIEZA GENERAL

delitos.columns = delitos.columns.str.strip().str.lower()
estaciones.columns = estaciones.columns.str.strip().str.lower()
barrios.columns = barrios.columns.str.strip().str.lower()

# LIMPIEZA DELITOS

col_tipo = "tipo_delito" if "tipo_delito" in delitos.columns else "tipo"

if col_tipo in delitos.columns:
    delitos[col_tipo] = delitos[col_tipo].astype(str).str.lower().str.strip()
    delitos = delitos[delitos[col_tipo].isin(["robo", "hurto"])].copy()
    print(f"✅ Delitos filtrados robo/hurto: {len(delitos):,}")

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")

delitos = delitos.dropna(subset=["latitud", "longitud"])

delitos = delitos[
    delitos["latitud"].between(-34.75, -34.50) &
    delitos["longitud"].between(-58.60, -58.30)
].copy()

print(f"✅ Delitos con coordenadas válidas: {len(delitos):,}")

# LIMPIEZA ESTACIONES

estaciones["lat"] = pd.to_numeric(estaciones["lat"], errors="coerce")
estaciones["long"] = pd.to_numeric(estaciones["long"], errors="coerce")

estaciones = estaciones.dropna(subset=["lat", "long"]).copy()

# Crear identificador propio si no existe id
if "id" not in estaciones.columns:
    estaciones = estaciones.reset_index(drop=True)
    estaciones["id_estacion"] = estaciones.index + 1
else:
    estaciones["id_estacion"] = estaciones["id"]

print(f"✅ Estaciones con coordenadas válidas: {len(estaciones):,}")

# FILTRAR ESTACIONES DENTRO DE CABA

print("📍 Filtrando estaciones dentro del límite oficial de CABA...")

barrios["geometry"] = gpd.GeoSeries.from_wkt(barrios["geometry"])

barrios_gdf = gpd.GeoDataFrame(
    barrios,
    geometry="geometry",
    crs="EPSG:4326"
)

caba = barrios_gdf.dissolve()

estaciones_gdf = gpd.GeoDataFrame(
    estaciones,
    geometry=gpd.points_from_xy(estaciones["long"], estaciones["lat"]),
    crs="EPSG:4326"
)

estaciones_gdf = gpd.sjoin(
    estaciones_gdf,
    caba,
    how="inner",
    predicate="within"
)

if "index_right" in estaciones_gdf.columns:
    estaciones_gdf = estaciones_gdf.drop(columns=["index_right"])

print(f"✅ Estaciones dentro de CABA: {len(estaciones_gdf):,}")

# GEODATAFRAME DE DELITOS

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
)

# Pasar a CRS métrico para distancias y áreas
delitos_m = delitos_gdf.to_crs(epsg=3857)
estaciones_m = estaciones_gdf.to_crs(epsg=3857)

# CREACIÓN DE ANILLOS

print("🧩 Creando anillos alrededor de estaciones...")

distancias = [0, 100, 200, 300]

anillos = []

for _, row in estaciones_m.iterrows():
    for i in range(len(distancias) - 1):
        externo = row.geometry.buffer(distancias[i + 1])
        interno = row.geometry.buffer(distancias[i])
        anillo = externo.difference(interno)

        anillos.append({
            "id_estacion": row["id_estacion"],
            "nombre": row.get("nombre", "S/D"),
            "linea": row.get("linea", "S/D"),
            "ramal": row.get("ramal", "S/D"),
            "barrio": row.get("barrio", "S/D"),
            "comuna": row.get("comuna", "S/D"),
            "anillo": i + 1,
            "distancia": f"{distancias[i]}-{distancias[i + 1]} m",
            "geometry": anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

print(f"✅ Anillos generados: {len(anillos_gdf):,}")

# SPATIAL JOIN SIN DOBLE CONTEO

print("📍 Cruzando delitos con anillos de estaciones...")

delitos_m["id_delito"] = range(len(delitos_m))

join = gpd.sjoin(
    delitos_m,
    anillos_gdf,
    how="inner",
    predicate="within"
)

# Si un delito cae dentro de varios anillos, se asigna al más cercano
join = join.sort_values(by="anillo")
join = join.drop_duplicates(subset="id_delito", keep="first")

conteos = (
    join.groupby(["id_estacion", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(
    conteos,
    on=["id_estacion", "anillo"],
    how="left"
)

anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)

# CÁLCULO DE DENSIDADES

anillos_gdf["area_km2"] = anillos_gdf.geometry.area / 1_000_000
anillos_gdf["densidad"] = (
    anillos_gdf["cantidad_delitos"] / anillos_gdf["area_km2"]
)

base = anillos_gdf[anillos_gdf["anillo"] == 1][
    ["id_estacion", "densidad"]
].rename(columns={"densidad": "base"})

anillos_gdf = anillos_gdf.merge(
    base,
    on="id_estacion",
    how="left"
)

anillos_gdf["densidad_relativa"] = np.where(
    anillos_gdf["base"] > 0,
    anillos_gdf["densidad"] / anillos_gdf["base"],
    np.nan
)

# EXPORTAR CSV

anillos_gdf.drop(columns="geometry").to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(f"💾 CSV guardado en: {OUTPUT_CSV}")

# MAPA

print("🗺️ Generando mapa...")

anillos_wgs = anillos_gdf.to_crs(epsg=4326)

mapa = folium.Map(
    location=[-34.61, -58.43],
    zoom_start=12,
    tiles="CartoDB positron"
)

vals = anillos_wgs["densidad_relativa"].dropna()

if len(vals) > 0:
    p20 = vals.quantile(0.2)
    p80 = vals.quantile(0.8)

    p20 = min(p20, 1)
    p80 = max(p80, 1)

    if p20 == p80:
        p20 = 0
        p80 = max(vals.max(), 1)

    colormap = cm.LinearColormap(
        ["green", "yellow", "red"],
        vmin=p20,
        vmax=p80
    )
    colormap.caption = "Densidad relativa respecto del anillo 0-100 m"
    colormap.add_to(mapa)

    def clip(v):
        if pd.isna(v):
            return None
        return max(min(v, p80), p20)

else:
    colormap = None

    def clip(v):
        return None

for _, row in anillos_wgs.iterrows():
    val = clip(row["densidad_relativa"])
    color = "#cccccc" if val is None else colormap(val)

    relativa = (
        "S/D"
        if pd.isna(row["densidad_relativa"])
        else f"{row['densidad_relativa']:.2f}"
    )

    tooltip = f"""
    Estación: {row['nombre']}<br>
    Línea: {row['linea']}<br>
    Ramal: {row['ramal']}<br>
    Barrio: {row['barrio']}<br>
    Anillo: {row['distancia']}<br>
    Delitos: {int(row['cantidad_delitos'])}<br>
    Densidad: {row['densidad']:.2f} delitos/km²<br>
    Densidad relativa: {relativa}
    """

    folium.GeoJson(
        row["geometry"].__geo_interface__,
        style_function=lambda f, c=color: {
            "fillColor": c,
            "color": c,
            "weight": 1,
            "fillOpacity": 0.55
        },
        tooltip=tooltip
    ).add_to(mapa)

# Puntos de estaciones
estaciones_wgs = estaciones_gdf.to_crs(epsg=4326)

for _, row in estaciones_wgs.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=4,
        color="black",
        fill=True,
        fill_opacity=0.9,
        tooltip=f"{row.get('nombre', 'S/D')} - {row.get('linea', 'S/D')}"
    ).add_to(mapa)

mapa.save(OUTPUT_HTML)

print(f"🗺️ Mapa guardado en: {OUTPUT_HTML}")

# RESUMEN ESTADÍSTICO

print("\n📊 RESUMEN ESTADÍSTICO")

resumen = anillos_gdf.groupby("anillo").agg(
    distancia=("distancia", "first"),
    delitos_totales=("cantidad_delitos", "sum"),
    densidad_promedio=("densidad", "mean")
).reset_index()

base_global = resumen.loc[
    resumen["anillo"] == 1,
    "densidad_promedio"
].values[0]

resumen["densidad_relativa_global"] = np.where(
    base_global > 0,
    resumen["densidad_promedio"] / base_global,
    np.nan
)

print(resumen.to_string(index=False, float_format="%.2f"))

print("\n✅ Proceso finalizado correctamente.")