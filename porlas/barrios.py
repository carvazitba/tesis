from pathlib import Path
import pandas as pd
import geopandas as gpd
import folium
from shapely import wkt

# ==================================================
# RUTAS REPRODUCIBLES
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_RAW = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_RAW / "barrios.csv"
OUTPUT_MAP = OUTPUT_DIR / "mapa_barrios_caba.html"

print("=" * 60)
print("VISUALIZACIÓN DE BARRIOS DE CABA")
print("=" * 60)
print(f"📂 Cargando archivo desde: {INPUT_FILE}")
print(f"🗺️ Mapa de salida: {OUTPUT_MAP}")

# ==================================================
# CARGA DEL DATASET
# ==================================================

df = pd.read_csv(INPUT_FILE)

print("\nColumnas detectadas:")
print(df.columns.tolist())

# ==================================================
# CONVERSIÓN DE GEOMETRÍA
# ==================================================
# El dataset de barrios suele traer la geometría en formato WKT,
# por ejemplo: POLYGON ((...))

if "geometry" not in df.columns:
    raise ValueError("No se encontró la columna 'geometry' en el dataset.")

df["geometry"] = df["geometry"].apply(wkt.loads)

gdf = gpd.GeoDataFrame(
    df,
    geometry="geometry",
    crs="EPSG:4326"
)

# ==================================================
# LIMPIEZA BÁSICA
# ==================================================

# Normalizamos nombres de columnas por si vienen en mayúsculas/minúsculas
gdf.columns = gdf.columns.str.lower()

# Verificamos columnas esperadas
columnas_necesarias = ["nombre", "comuna", "geometry"]

for col in columnas_necesarias:
    if col not in gdf.columns:
        raise ValueError(f"No se encontró la columna necesaria: {col}")

# Convertimos comuna a texto para mostrar mejor en el popup
gdf["comuna"] = gdf["comuna"].astype(str)

print(f"\nCantidad de barrios cargados: {len(gdf)}")
print(f"CRS del GeoDataFrame: {gdf.crs}")

# ==================================================
# CENTRO DEL MAPA
# ==================================================

centro = gdf.geometry.union_all().centroid

mapa = folium.Map(
    location=[centro.y, centro.x],
    zoom_start=12,
    tiles="CartoDB positron"
)

# ==================================================
# CAPA DE BARRIOS
# ==================================================

folium.GeoJson(
    gdf,
    name="Barrios CABA",
    style_function=lambda feature: {
        "fillColor": "#3186cc",
        "color": "#1f4e79",
        "weight": 1,
        "fillOpacity": 0.25,
    },
    highlight_function=lambda feature: {
        "fillColor": "#ffcc00",
        "color": "#000000",
        "weight": 2,
        "fillOpacity": 0.55,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["nombre", "comuna"],
        aliases=["Barrio:", "Comuna:"],
        localize=True,
        sticky=True,
        labels=True
    ),
    popup=folium.GeoJsonPopup(
        fields=["nombre", "comuna"],
        aliases=["Barrio:", "Comuna:"],
        localize=True,
        labels=True
    )
).add_to(mapa)

# ==================================================
# ETIQUETAS CON NOMBRE DEL BARRIO
# ==================================================

for _, row in gdf.iterrows():
    punto = row.geometry.centroid

    folium.Marker(
        location=[punto.y, punto.x],
        icon=folium.DivIcon(
            html=f"""
            <div style="
                font-size: 9px;
                color: black;
                font-weight: bold;
                text-shadow: 1px 1px 2px white;
                ">
                {row['nombre']}
            </div>
            """
        )
    ).add_to(mapa)

# ==================================================
# CONTROL DE CAPAS Y GUARDADO
# ==================================================

folium.LayerControl().add_to(mapa)

mapa.save(OUTPUT_MAP)

print("\n✅ Mapa generado correctamente.")
print(f"📍 Abrir en navegador: {OUTPUT_MAP}")