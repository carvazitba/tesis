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

INPUT_ESTACIONES = DATA_RAW / "estaciones-de-ferrocarril.csv"
INPUT_BARRIOS = DATA_RAW / "barrios.csv"
OUTPUT_MAP = OUTPUT_DIR / "mapa_estaciones_ferrocarril_caba.html"

print("=" * 60)
print("VISUALIZACIÓN DE ESTACIONES DE FERROCARRIL EN CABA")
print("=" * 60)
print(f"📂 Cargando estaciones desde: {INPUT_ESTACIONES}")
print(f"📂 Cargando barrios desde: {INPUT_BARRIOS}")
print(f"🗺️ Mapa de salida: {OUTPUT_MAP}")

# ==================================================
# CARGA DE DATASETS
# ==================================================

if not INPUT_ESTACIONES.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_ESTACIONES}")

if not INPUT_BARRIOS.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_BARRIOS}")

df_est = pd.read_csv(INPUT_ESTACIONES, low_memory=False)
df_barrios = pd.read_csv(INPUT_BARRIOS, low_memory=False)

print("\nColumnas detectadas en estaciones:")
print(df_est.columns.tolist())

print("\nColumnas detectadas en barrios:")
print(df_barrios.columns.tolist())

# Normalizar nombres de columnas
df_est.columns = df_est.columns.str.strip().str.lower()
df_barrios.columns = df_barrios.columns.str.strip().str.lower()

# ==================================================
# GEOMETRÍA DE BARRIOS
# ==================================================

if "geometry" not in df_barrios.columns:
    raise ValueError("No se encontró la columna 'geometry' en barrios.csv.")

df_barrios["geometry"] = df_barrios["geometry"].apply(wkt.loads)

gdf_barrios = gpd.GeoDataFrame(
    df_barrios,
    geometry="geometry",
    crs="EPSG:4326"
)

print(f"\nCantidad de barrios cargados: {len(gdf_barrios)}")

# ==================================================
# VALIDACIÓN DE COORDENADAS DE ESTACIONES
# ==================================================

posibles_lat = ["lat", "latitud", "latitude"]
posibles_lon = ["long", "lon", "lng", "longitud", "longitude"]

col_lat = next((col for col in posibles_lat if col in df_est.columns), None)
col_lon = next((col for col in posibles_lon if col in df_est.columns), None)

if col_lat is None or col_lon is None:
    raise ValueError(
        "No se encontraron columnas de coordenadas. "
        "Verificar si el dataset contiene columnas como 'lat'/'long' o 'latitud'/'longitud'."
    )

print(f"✅ Columna de latitud detectada: {col_lat}")
print(f"✅ Columna de longitud detectada: {col_lon}")

df_est[col_lat] = pd.to_numeric(df_est[col_lat], errors="coerce")
df_est[col_lon] = pd.to_numeric(df_est[col_lon], errors="coerce")

df_est = df_est.dropna(subset=[col_lat, col_lon]).copy()

# Filtrado básico para evitar coordenadas incorrectas
df_est = df_est[
    (df_est[col_lat].between(-35.0, -34.0)) &
    (df_est[col_lon].between(-59.0, -57.5))
].copy()

print(f"\nCantidad de estaciones con coordenadas válidas: {len(df_est)}")

# ==================================================
# GEO DATAFRAME DE ESTACIONES
# ==================================================

gdf_est = gpd.GeoDataFrame(
    df_est,
    geometry=gpd.points_from_xy(df_est[col_lon], df_est[col_lat]),
    crs="EPSG:4326"
)

# ==================================================
# FILTRO ESPACIAL: SOLO ESTACIONES DENTRO DE BARRIOS
# ==================================================

gdf_est_filtrado = gpd.sjoin(
    gdf_est,
    gdf_barrios[["geometry"]],
    how="inner",
    predicate="within"
).drop(columns=["index_right"])

print(f"✅ Estaciones dentro del polígono de barrios: {len(gdf_est_filtrado)}")

if gdf_est_filtrado.empty:
    raise ValueError(
        "No quedaron estaciones dentro del polígono de barrios. "
        "Revisar coordenadas o geometrías."
    )

# ==================================================
# CENTRO DEL MAPA
# ==================================================

centro = gdf_barrios.geometry.union_all().centroid

mapa = folium.Map(
    location=[centro.y, centro.x],
    zoom_start=12,
    tiles="CartoDB positron"
)

# ==================================================
# COLUMNAS PARA TOOLTIP / POPUP
# ==================================================

columnas_tooltip = []

for col in [
    "nombre",
    "estacion",
    "linea",
    "ramal",
    "servicio",
    "barrio",
    "comuna"
]:
    if col in gdf_est_filtrado.columns:
        columnas_tooltip.append(col)

aliases = {
    "nombre": "Nombre:",
    "estacion": "Estación:",
    "linea": "Línea:",
    "ramal": "Ramal:",
    "servicio": "Servicio:",
    "barrio": "Barrio:",
    "comuna": "Comuna:"
}

# ==================================================
# CAPA DE PUNTOS
# ==================================================

grupo_estaciones = folium.FeatureGroup(name="Estaciones de ferrocarril en CABA")

for _, row in gdf_est_filtrado.iterrows():

    texto_tooltip = "<br>".join(
        [
            f"{aliases.get(col, col + ':')} {row[col]}"
            for col in columnas_tooltip
            if pd.notna(row[col])
        ]
    )

    folium.CircleMarker(
        location=[row[col_lat], row[col_lon]],
        radius=5,
        color="#7f0000",
        fill=True,
        fill_color="#d73027",
        fill_opacity=0.80,
        weight=1,
        tooltip=folium.Tooltip(texto_tooltip) if texto_tooltip else None,
        popup=folium.Popup(texto_tooltip, max_width=300) if texto_tooltip else None
    ).add_to(grupo_estaciones)

grupo_estaciones.add_to(mapa)

# ==================================================
# GUARDADO
# ==================================================

folium.LayerControl().add_to(mapa)

mapa.save(OUTPUT_MAP)

print("\n✅ Mapa generado correctamente.")
print(f"📍 Abrir en navegador: {OUTPUT_MAP}")