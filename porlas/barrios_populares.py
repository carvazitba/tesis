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

INPUT_FILE = DATA_RAW / "barrios_populares_poligono.csv"
OUTPUT_MAP = OUTPUT_DIR / "mapa_barrios_populares_caba.html"

print("=" * 60)
print("VISUALIZACIÓN DE BARRIOS POPULARES DE CABA")
print("=" * 60)
print(f"📂 Cargando archivo desde: {INPUT_FILE}")
print(f"🗺️ Mapa de salida: {OUTPUT_MAP}")

# ==================================================
# CARGA DEL DATASET
# ==================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("\nColumnas detectadas:")
print(df.columns.tolist())

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()

# ==================================================
# CONVERSIÓN DE GEOMETRÍA
# ==================================================

posibles_columnas_geom = ["geometry", "geom", "wkt"]

col_geom = None
for col in posibles_columnas_geom:
    if col in df.columns:
        col_geom = col
        break

if col_geom is None:
    raise ValueError(
        "No se encontró una columna de geometría. "
        "Verificar si el archivo contiene una columna llamada 'geometry', 'geom' o 'wkt'."
    )

print(f"\n✅ Columna de geometría detectada: {col_geom}")

df["geometry"] = df[col_geom].apply(wkt.loads)

gdf = gpd.GeoDataFrame(
    df,
    geometry="geometry",
    crs="EPSG:4326"
)

# ==================================================
# COLUMNAS PARA TOOLTIP / POPUP
# ==================================================

columnas_tooltip = []

for col in ["nombre", "nom_mapa", "tipo_asent", "alias", "superficie"]:
    if col in gdf.columns:
        columnas_tooltip.append(col)

if len(columnas_tooltip) == 0:
    raise ValueError(
        "No se encontraron columnas descriptivas esperadas "
        "como 'nombre', 'nom_mapa', 'tipo_asent', 'alias' o 'superficie'."
    )

print(f"Columnas usadas en tooltip: {columnas_tooltip}")
print(f"Cantidad de barrios populares cargados: {len(gdf)}")
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
# CAPA DE POLÍGONOS
# ==================================================

aliases = {
    "nombre": "Nombre:",
    "nom_mapa": "Nombre en mapa:",
    "tipo_asent": "Tipo de asentamiento:",
    "alias": "Alias:",
    "superficie": "Superficie:"
}

tooltip_aliases = [aliases.get(col, f"{col}:") for col in columnas_tooltip]

folium.GeoJson(
    gdf,
    name="Barrios populares CABA",
    style_function=lambda feature: {
        "fillColor": "#d73027",
        "color": "#7f0000",
        "weight": 1.5,
        "fillOpacity": 0.35,
    },
    highlight_function=lambda feature: {
        "fillColor": "#ffcc00",
        "color": "#000000",
        "weight": 2,
        "fillOpacity": 0.60,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=columnas_tooltip,
        aliases=tooltip_aliases,
        localize=True,
        sticky=True,
        labels=True
    ),
    popup=folium.GeoJsonPopup(
        fields=columnas_tooltip,
        aliases=tooltip_aliases,
        localize=True,
        labels=True
    )
).add_to(mapa)

# ==================================================
# GUARDADO
# ==================================================

folium.LayerControl().add_to(mapa)

mapa.save(OUTPUT_MAP)

print("\n✅ Mapa generado correctamente.")
print(f"📍 Abrir en navegador: {OUTPUT_MAP}")