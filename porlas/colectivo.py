from pathlib import Path
import pandas as pd
import geopandas as gpd
import folium

# ==================================================
# RUTAS REPRODUCIBLES
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_RAW = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PARADAS = DATA_RAW / "paradas-de-colectivo.xlsx"
OUTPUT_MAP = OUTPUT_DIR / "mapa_paradas_colectivo_caba.html"

print("=" * 60)
print("VISUALIZACIÓN DE PARADAS DE COLECTIVO EN CABA")
print("=" * 60)
print(f"📂 Cargando paradas desde: {INPUT_PARADAS}")
print(f"🗺️ Mapa de salida: {OUTPUT_MAP}")

# ==================================================
# CARGA DEL DATASET
# ==================================================

if not INPUT_PARADAS.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_PARADAS}")

df_paradas = pd.read_excel(INPUT_PARADAS)

print("\nColumnas detectadas en paradas:")
print(df_paradas.columns.tolist())

# Normalizar nombres de columnas
df_paradas.columns = df_paradas.columns.str.strip().str.lower()

print("\nColumnas normalizadas:")
print(df_paradas.columns.tolist())

# ==================================================
# VALIDACIÓN Y CONVERSIÓN DE COORDENADAS
# ==================================================

# En este dataset:
# coord_x = longitud, por ejemplo -58,3709946
# coord_y = latitud,  por ejemplo -34,62565880

col_lon = "coord_x"
col_lat = "coord_y"

if col_lat not in df_paradas.columns or col_lon not in df_paradas.columns:
    raise ValueError(
        "No se encontraron las columnas 'coord_x' y 'coord_y' en el Excel."
    )

print(f"\n✅ Columna de longitud detectada: {col_lon}")
print(f"✅ Columna de latitud detectada: {col_lat}")

# Convertir coma decimal a punto decimal
df_paradas[col_lat] = (
    df_paradas[col_lat]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

df_paradas[col_lon] = (
    df_paradas[col_lon]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

# Convertir a numérico
df_paradas[col_lat] = pd.to_numeric(df_paradas[col_lat], errors="coerce")
df_paradas[col_lon] = pd.to_numeric(df_paradas[col_lon], errors="coerce")

# Eliminar registros sin coordenadas válidas
df_paradas = df_paradas.dropna(subset=[col_lat, col_lon]).copy()

# Filtrado básico para evitar coordenadas claramente fuera de CABA
df_paradas = df_paradas[
    (df_paradas[col_lat].between(-35.0, -34.0)) &
    (df_paradas[col_lon].between(-59.0, -57.0))
].copy()

print(f"\nCantidad de paradas con coordenadas válidas: {len(df_paradas)}")

if df_paradas.empty:
    raise ValueError(
        "No quedaron paradas con coordenadas válidas. "
        "Revisar si coord_x corresponde a longitud y coord_y a latitud."
    )

# ==================================================
# GEO DATAFRAME
# ==================================================

gdf_paradas = gpd.GeoDataFrame(
    df_paradas,
    geometry=gpd.points_from_xy(
        df_paradas[col_lon],  # x = longitud
        df_paradas[col_lat]   # y = latitud
    ),
    crs="EPSG:4326"
)

print(f"CRS del GeoDataFrame: {gdf_paradas.crs}")

# ==================================================
# CENTRO DEL MAPA
# ==================================================

centro = gdf_paradas.geometry.union_all().centroid

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
    "fid",
    "calle",
    "alt plano",
    "direccion",
    "comuna",
    "barrio",
    "l1",
    "l1_sen",
    "l2",
    "l2_sen",
    "l3",
    "l3_sen",
    "l4",
    "l4_sen",
    "l5",
    "l5_sen",
    "l6",
    "l6_sen"
]:
    if col in gdf_paradas.columns:
        columnas_tooltip.append(col)

aliases = {
    "fid": "ID:",
    "calle": "Calle:",
    "alt plano": "Altura:",
    "direccion": "Dirección:",
    "comuna": "Comuna:",
    "barrio": "Barrio:",
    "l1": "Línea 1:",
    "l1_sen": "Sentido 1:",
    "l2": "Línea 2:",
    "l2_sen": "Sentido 2:",
    "l3": "Línea 3:",
    "l3_sen": "Sentido 3:",
    "l4": "Línea 4:",
    "l4_sen": "Sentido 4:",
    "l5": "Línea 5:",
    "l5_sen": "Sentido 5:",
    "l6": "Línea 6:",
    "l6_sen": "Sentido 6:"
}

# ==================================================
# CAPA DE PUNTOS
# ==================================================

grupo_paradas = folium.FeatureGroup(name="Paradas de colectivo")

for _, row in gdf_paradas.iterrows():

    texto_tooltip = "<br>".join(
        [
            f"{aliases.get(col, col + ':')} {row[col]}"
            for col in columnas_tooltip
            if pd.notna(row[col])
        ]
    )

    folium.CircleMarker(
        location=[
            row[col_lat],  # latitud
            row[col_lon]   # longitud
        ],
        radius=3,
        color="#1f4e79",
        fill=True,
        fill_color="#3186cc",
        fill_opacity=0.65,
        weight=1,
        tooltip=folium.Tooltip(texto_tooltip) if texto_tooltip else None,
        popup=folium.Popup(texto_tooltip, max_width=350) if texto_tooltip else None
    ).add_to(grupo_paradas)

grupo_paradas.add_to(mapa)

# ==================================================
# GUARDADO
# ==================================================

folium.LayerControl().add_to(mapa)

mapa.save(OUTPUT_MAP)

print("\n✅ Mapa generado correctamente.")
print(f"📍 Abrir en navegador: {OUTPUT_MAP}")