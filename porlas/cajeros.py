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

INPUT_FILE = DATA_RAW / "cajeros-automaticos.csv"
OUTPUT_MAP = OUTPUT_DIR / "mapa_cajeros_automaticos_caba.html"

print("=" * 60)
print("VISUALIZACIÓN DE CAJEROS AUTOMÁTICOS DE CABA")
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
# VALIDACIÓN DE COORDENADAS
# ==================================================

if "lat" not in df.columns or "long" not in df.columns:
    raise ValueError("No se encontraron las columnas 'lat' y 'long' en el dataset.")

df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["long"] = pd.to_numeric(df["long"], errors="coerce")

df = df.dropna(subset=["lat", "long"]).copy()

# Filtrado básico para evitar coordenadas incorrectas fuera de CABA
df = df[
    (df["lat"].between(-35.0, -34.0)) &
    (df["long"].between(-59.0, -57.5))
].copy()

print(f"\nCantidad de cajeros con coordenadas válidas: {len(df)}")

# ==================================================
# GEO DATAFRAME
# ==================================================

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["long"], df["lat"]),
    crs="EPSG:4326"
)

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
# COLUMNAS PARA TOOLTIP / POPUP
# ==================================================

columnas_tooltip = []

for col in ["banco", "red", "ubicacion", "domicilio", "barrio", "comuna"]:
    if col in gdf.columns:
        columnas_tooltip.append(col)

aliases = {
    "banco": "Banco:",
    "red": "Red:",
    "ubicacion": "Ubicación:",
    "domicilio": "Domicilio:",
    "barrio": "Barrio:",
    "comuna": "Comuna:"
}

# ==================================================
# CAPA DE PUNTOS
# ==================================================

grupo_cajeros = folium.FeatureGroup(name="Cajeros automáticos CABA")

for _, row in gdf.iterrows():

    texto_tooltip = "<br>".join(
        [
            f"{aliases.get(col, col + ':')} {row[col]}"
            for col in columnas_tooltip
            if pd.notna(row[col])
        ]
    )

    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=4,
        color="#1f4e79",
        fill=True,
        fill_color="#3186cc",
        fill_opacity=0.75,
        weight=1,
        tooltip=folium.Tooltip(texto_tooltip) if texto_tooltip else None,
        popup=folium.Popup(texto_tooltip, max_width=300) if texto_tooltip else None
    ).add_to(grupo_cajeros)

grupo_cajeros.add_to(mapa)

# ==================================================
# GUARDADO
# ==================================================

folium.LayerControl().add_to(mapa)

mapa.save(OUTPUT_MAP)

print("\n✅ Mapa generado correctamente.")
print(f"📍 Abrir en navegador: {OUTPUT_MAP}")