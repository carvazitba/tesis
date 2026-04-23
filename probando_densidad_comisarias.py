import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm

# =========================================
# CONFIGURACIÓN DE RUTAS
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DELITOS_PATH = os.path.join(DATA_DIR, "delitos_total.csv")
COMISARIAS_PATH = os.path.join(DATA_DIR, "comisarias-policia-de-la-ciudad.xlsx")

OUTPUT_HTML = os.path.join(OUTPUT_DIR, "mapa_comisarias_anillos_densidad_relativa_p20_p80.html")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "anillos_comisarias_densidad_relativa.csv")

# =========================================
# CARGA DE DATOS
# =========================================

print(f"📂 Cargando delitos desde: {DELITOS_PATH}")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)

print(f"📂 Cargando comisarías desde: {COMISARIAS_PATH}")
comisarias = pd.read_excel(COMISARIAS_PATH)

# =========================================
# LIMPIEZA BÁSICA
# =========================================

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")

if "comuna" in delitos.columns:
    delitos["comuna"] = pd.to_numeric(delitos["comuna"], errors="coerce")

delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()

comisarias["lat"] = pd.to_numeric(comisarias["lat"], errors="coerce")
comisarias["long"] = pd.to_numeric(comisarias["long"], errors="coerce")
comisarias = comisarias.dropna(subset=["lat", "long"]).copy()

# =========================================
# CREAR GEODATAFRAMES
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
# CREAR ANILLOS
# 4 anillos de 300 m
# =========================================

distancias = [0, 300, 600, 900, 1200]
anillos = []

for _, row in comisarias_m.iterrows():
    punto = row.geometry

    for i in range(4):
        r_in = distancias[i]
        r_out = distancias[i + 1]

        externo = punto.buffer(r_out)
        interno = punto.buffer(r_in)
        anillo = externo.difference(interno)

        anillos.append({
            "id": row["id"],
            "nombre": row["nombre"],
            "direccion": row["direccion"] if "direccion" in row else "",
            "barrio": row["barrio"] if "barrio" in row else None,
            "comuna": row["comuna"] if "comuna" in row else None,
            "anillo": i + 1,
            "distancia": f"{r_in}-{r_out} m",
            "radio_in": r_in,
            "radio_out": r_out,
            "geometry": anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# =========================================
# CALCULAR DELITOS POR ANILLO
# =========================================

print("📍 Calculando delitos por anillo...")
join = gpd.sjoin(delitos_m, anillos_gdf, how="inner", predicate="within")

conteos = (
    join.groupby(["id", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(
    conteos,
    on=["id", "anillo"],
    how="left"
)

anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)

# =========================================
# DENSIDAD ABSOLUTA
# =========================================

anillos_gdf["area_m2"] = anillos_gdf.geometry.area
anillos_gdf["area_km2"] = anillos_gdf["area_m2"] / 1_000_000
anillos_gdf["densidad_delitos_km2"] = anillos_gdf["cantidad_delitos"] / anillos_gdf["area_km2"]

# =========================================
# DENSIDAD RELATIVA RESPECTO AL ANILLO 1
# =========================================

densidad_anillo_1 = (
    anillos_gdf[anillos_gdf["anillo"] == 1][["id", "densidad_delitos_km2"]]
    .rename(columns={"densidad_delitos_km2": "densidad_anillo_1"})
)

anillos_gdf = anillos_gdf.merge(densidad_anillo_1, on="id", how="left")

anillos_gdf["densidad_relativa_a1"] = np.where(
    anillos_gdf["densidad_anillo_1"] > 0,
    anillos_gdf["densidad_delitos_km2"] / anillos_gdf["densidad_anillo_1"],
    np.nan
)

# =========================================
# GUARDAR TABLA
# =========================================

anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
print(f"💾 Tabla guardada en: {OUTPUT_CSV}")

# =========================================
# VOLVER A WGS84 PARA EL MAPA
# =========================================

anillos_wgs84 = anillos_gdf.to_crs(epsg=4326)

# =========================================
# CREAR MAPA BASE
# =========================================

mapa = folium.Map(
    location=[-34.61, -58.43],
    zoom_start=12,
    tiles="cartodbpositron"
)

# =========================================
# ESCALA DE COLOR BASADA EN PERCENTILES 20-80
# =========================================

valores_validos = anillos_wgs84["densidad_relativa_a1"].dropna()

if len(valores_validos) == 0:
    raise ValueError("No se pudieron calcular densidades relativas válidas.")

p20 = float(valores_validos.quantile(0.20))
p80 = float(valores_validos.quantile(0.80))

# Asegurar que el valor 1 quede contenido dentro de la escala
p20 = min(p20, 1.0)
p80 = max(p80, 1.0)

colormap = cm.LinearColormap(
    colors=["darkgreen", "yellow", "darkred"],
    vmin=p20,
    vmax=p80
)
colormap.caption = "Densidad relativa respecto al primer anillo (escala recortada P20–P80)"
colormap.add_to(mapa)

def valor_visual(x, vmin, vmax):
    if pd.isna(x):
        return np.nan
    if x < vmin:
        return vmin
    if x > vmax:
        return vmax
    return x

# =========================================
# AGREGAR ANILLOS AL MAPA
# =========================================

for _, row in anillos_wgs84.iterrows():
    valor_real = row["densidad_relativa_a1"]
    valor_mapa = valor_visual(valor_real, p20, p80)

    if pd.isna(valor_mapa):
        color = "#cccccc"
    else:
        color = colormap(valor_mapa)

    if pd.notna(row["densidad_relativa_a1"]):
        tooltip_html = (
            f"<b>Comisaría:</b> {row['nombre']}<br>"
            f"<b>Dirección:</b> {row['direccion']}<br>"
            f"<b>Barrio:</b> {row['barrio']}<br>"
            f"<b>Comuna:</b> {row['comuna']}<br>"
            f"<b>Anillo:</b> {row['distancia']}<br>"
            f"<b>Delitos:</b> {int(row['cantidad_delitos'])}<br>"
            f"<b>Densidad absoluta:</b> {row['densidad_delitos_km2']:.2f} delitos/km²<br>"
            f"<b>Densidad relativa:</b> {row['densidad_relativa_a1']:.2f}x"
        )
    else:
        tooltip_html = (
            f"<b>Comisaría:</b> {row['nombre']}<br>"
            f"<b>Anillo:</b> {row['distancia']}<br>"
            f"<b>Sin densidad relativa calculable</b>"
        )

    folium.GeoJson(
        data=row["geometry"].__geo_interface__,
        style_function=lambda feature, color=color: {
            "fillColor": color,
            "color": color,
            "weight": 1,
            "fillOpacity": 0.60
        },
        tooltip=folium.Tooltip(tooltip_html)
    ).add_to(mapa)

# =========================================
# AGREGAR COMISARÍAS
# =========================================

for _, row in comisarias_gdf.iterrows():
    tooltip_html = (
        f"<b>{row['nombre']}</b><br>"
        f"{row['direccion']}<br>"
        f"Barrio: {row['barrio']}<br>"
        f"Comuna: {row['comuna']}"
    )

    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=3,
        color="black",
        fill=True,
        fill_color="black",
        fill_opacity=1,
        tooltip=folium.Tooltip(tooltip_html)
    ).add_to(mapa)

# =========================================
# GUARDAR MAPA
# =========================================

mapa.save(OUTPUT_HTML)
print(f"🗺️ Mapa guardado en: {OUTPUT_HTML}")
print(f"Escala visual usada: percentil 20 = {p20:.2f}, percentil 80 = {p80:.2f}")