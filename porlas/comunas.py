from pathlib import Path
import geopandas as gpd
import folium

# ==================================================
# RUTAS REPRODUCIBLES
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_RAW = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_RAW / "comunas.geojson"
OUTPUT_MAP = OUTPUT_DIR / "mapa_comunas_caba.html"

print("=" * 60)
print("VISUALIZACIÓN DE COMUNAS DE CABA")
print("=" * 60)
print(f"📂 Cargando archivo desde: {INPUT_FILE}")
print(f"🗺️ Mapa de salida: {OUTPUT_MAP}")

# ==================================================
# CARGA DEL GEOJSON
# ==================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

gdf = gpd.read_file(INPUT_FILE)

print("\nColumnas detectadas:")
print(gdf.columns.tolist())

# Normalizar nombres de columnas
gdf.columns = gdf.columns.str.strip().str.lower()

# Asegurar CRS WGS84 para Folium
if gdf.crs is None:
    print("⚠️ El archivo no tiene CRS definido. Se asigna EPSG:4326.")
    gdf = gdf.set_crs("EPSG:4326")
elif gdf.crs.to_string() != "EPSG:4326":
    print(f"🔄 Reproyectando de {gdf.crs} a EPSG:4326...")
    gdf = gdf.to_crs("EPSG:4326")

# ==================================================
# VALIDACIONES
# ==================================================

if "geometry" not in gdf.columns:
    raise ValueError("No se encontró la columna de geometría.")

if "comuna" not in gdf.columns:
    raise ValueError("No se encontró la columna 'comuna' para crear las etiquetas.")

# ==================================================
# COLUMNAS PARA TOOLTIP / POPUP
# ==================================================

columnas_tooltip = []

for col in ["comuna", "barrios", "perimetro", "area"]:
    if col in gdf.columns:
        columnas_tooltip.append(col)

if len(columnas_tooltip) == 0:
    columnas_tooltip = ["comuna"]

print(f"Columnas usadas en tooltip: {columnas_tooltip}")
print(f"Cantidad de comunas cargadas: {len(gdf)}")
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
    "comuna": "Comuna:",
    "barrios": "Barrios:",
    "perimetro": "Perímetro:",
    "area": "Área:"
}

tooltip_aliases = [aliases.get(col, f"{col}:") for col in columnas_tooltip]

folium.GeoJson(
    gdf,
    name="Comunas CABA",
    style_function=lambda feature: {
        "fillColor": "#3186cc",
        "color": "#1f4e79",
        "weight": 1.5,
        "fillOpacity": 0.30,
    },
    highlight_function=lambda feature: {
        "fillColor": "#ffcc00",
        "color": "#000000",
        "weight": 2,
        "fillOpacity": 0.55,
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
# ETIQUETAS DE LAS COMUNAS
# ==================================================

for _, row in gdf.iterrows():
    punto = row.geometry.representative_point()

    folium.Marker(
        location=[punto.y, punto.x],
        icon=folium.DivIcon(
            html=f"""
            <div style="
                font-size: 11px;
                color: black;
                font-weight: bold;
                text-align: center;
                text-shadow: 1px 1px 2px white;
                white-space: nowrap;
            ">
                {row['comuna']}
            </div>
            """
        )
    ).add_to(mapa)

# ==================================================
# GUARDADO
# ==================================================

folium.LayerControl().add_to(mapa)

mapa.save(OUTPUT_MAP)

print("\n✅ Mapa generado correctamente.")
print(f"📍 Abrir en navegador: {OUTPUT_MAP}")