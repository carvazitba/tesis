import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm

# =========================================
# CONFIGURACIÓN DE RUTAS
# =========================================

DELITOS_PATH = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\delitos_total.csv'
CAJEROS_PATH = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\dataset\cajeros-automaticos.csv'

OUTPUT_DIR = r'C:\Users\digni\OneDrive\Documents\GitHub\tesis\outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_HTML = os.path.join(OUTPUT_DIR, "mapa_cajeros_3anillos_50m.html")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "anillos_cajeros_3anillos_50m.csv")

# =========================================
# CARGA DE DATOS
# =========================================

print("📂 Cargando delitos...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)

print("📂 Cargando cajeros...")
cajeros = pd.read_csv(CAJEROS_PATH, low_memory=False, encoding='utf-8')

# =========================================
# LIMPIEZA Y FILTRADO METODOLÓGICO
# =========================================

delitos.columns = delitos.columns.str.strip().str.lower()
cajeros.columns = cajeros.columns.str.strip().str.lower()

# ---> NUEVO: Filtrar SOLO "Robo" y "Hurto" <---
col_tipo = 'tipo_delito' if 'tipo_delito' in delitos.columns else 'tipo'

if col_tipo in delitos.columns:
    # Pasar a minúsculas para evitar errores tipográficos
    delitos[col_tipo] = delitos[col_tipo].astype(str).str.strip().str.lower()
    delitos = delitos[delitos[col_tipo].isin(['robo', 'hurto'])]
    print(f"✅ Filtro aplicado: Analizando exclusivamente 'Robos' y 'Hurtos' (Total: {len(delitos)} registros)")

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"])

cajeros["lat"] = pd.to_numeric(cajeros["lat"], errors="coerce")
cajeros["long"] = pd.to_numeric(cajeros["long"], errors="coerce")
cajeros = cajeros.dropna(subset=["lat", "long"])

# =========================================
# GEO DATAFRAMES Y PROYECCIÓN MÉTRICA
# =========================================

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

# =========================================
# CREAR 3 ANILLOS DE 50 METROS
# =========================================

distancias = [0, 50, 100, 150]
anillos = []

for _, row in cajeros_m.iterrows():
    punto = row.geometry

    for i in range(len(distancias) - 1):
        r_in = distancias[i]
        r_out = distancias[i + 1]

        externo = punto.buffer(r_out)
        interno = punto.buffer(r_in)
        anillo = externo.difference(interno)

        anillos.append({
            "id": row["id"],
            "banco": row.get("banco", "S/D"),
            "red": row.get("red", "S/D"),
            "ubicacion": row.get("ubicacion", ""),
            "barrio": row.get("barrio", "S/D"),
            "comuna": row.get("comuna", None),
            "anillo": i + 1,
            "distancia": f"{r_in}-{r_out} m",
            "geometry": anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# =========================================
# SPATIAL JOIN (CON DEDUPLICACIÓN)
# =========================================

print("📍 Cruzando delitos espaciales y eliminando superposiciones...")

# Asignar ID único a los robos/hurtos
delitos_m['id_delito'] = range(len(delitos_m))

# Hacer el cruce
join = gpd.sjoin(delitos_m, anillos_gdf, how="inner", predicate="within")

# Ordenar por proximidad al cajero y eliminar duplicados (Nearest Allocation)
join = join.sort_values(by='anillo')
join = join.drop_duplicates(subset='id_delito', keep='first')

conteos = (
    join.groupby(["id", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(conteos, on=["id", "anillo"], how="left")
anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)

# =========================================
# DENSIDADES
# =========================================

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

# =========================================
# EXPORTAR CSV
# =========================================

anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
print("💾 CSV guardado")

# =========================================
# MAPA INTERACTIVO
# =========================================

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
    Delitos (Robos/Hurtos): {int(row['cantidad_delitos'])}<br>
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
print("🗺️ Mapa generado")

# =========================================
# RESUMEN ESTADÍSTICO GLOBAL CORREGIDO
# =========================================
print("\n" + "="*60)
print("📊 RESUMEN ESTADÍSTICO PARA ANÁLISIS DE TESIS (ROBO/HURTO)")
print("="*60)

resumen = anillos_gdf.groupby("anillo").agg(
    distancia=("distancia", "first"),
    delitos_totales=("cantidad_delitos", "sum"),
    densidad_abs_promedio=("densidad", "mean")
).reset_index()

densidad_base_global = resumen.loc[resumen["anillo"] == 1, "densidad_abs_promedio"].values[0]
resumen["densidad_relativa_global"] = resumen["densidad_abs_promedio"] / densidad_base_global

print(resumen.to_string(index=False, float_format="%.2f"))
print("-" * 60)