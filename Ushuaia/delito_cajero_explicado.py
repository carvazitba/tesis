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
# LIMPIEZA
# =========================================

delitos.columns = delitos.columns.str.strip().str.lower()
cajeros.columns = cajeros.columns.str.strip().str.lower()

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"])

cajeros["lat"] = pd.to_numeric(cajeros["lat"], errors="coerce")
cajeros["long"] = pd.to_numeric(cajeros["long"], errors="coerce")
cajeros = cajeros.dropna(subset=["lat", "long"])

# =========================================
# GEO DATAFRAMES
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

# =========================================
# PROYECCIÓN MÉTRICA
# =========================================

delitos_m = delitos_gdf.to_crs(epsg=3857)
cajeros_m = cajeros_gdf.to_crs(epsg=3857)

# =========================================
# CREAR 3 ANILLOS DE 50 METROS
# =========================================

distancias = [0, 50, 100, 150]  # 🔥 clave
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
# SPATIAL JOIN
# =========================================

print("📍 Calculando delitos...")
join = gpd.sjoin(delitos_m, anillos_gdf, how="inner", predicate="within")

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

# =========================================
# DENSIDAD RELATIVA
# =========================================

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
# MAPA
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

# =========================================
# DIBUJAR ANILLOS
# =========================================

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

# =========================================
# PUNTOS CAJEROS
# =========================================

for _, row in cajeros.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=3,
        color="black",
        fill=True
    ).add_to(mapa)

# =========================================
# GUARDAR
# =========================================

mapa.save(OUTPUT_HTML)
print("🗺️ Mapa generado")  
# =========================================
# ANÁLISIS TEXTUAL AUTOMÁTICO
# =========================================

OUTPUT_TXT = os.path.join(OUTPUT_DIR, "analisis_cajeros_3anillos_50m.txt")

print("🧠 Generando análisis textual...")

# Promedio de densidad por anillo
resumen = (
    anillos_gdf.groupby("anillo")
    .agg({
        "densidad": "mean",
        "densidad_relativa": "mean",
        "cantidad_delitos": "mean"
    })
    .reset_index()
)

# Extraer valores
a1 = resumen.loc[resumen["anillo"] == 1, "densidad"].values[0]
a2 = resumen.loc[resumen["anillo"] == 2, "densidad"].values[0]
a3 = resumen.loc[resumen["anillo"] == 3, "densidad"].values[0]

r2 = resumen.loc[resumen["anillo"] == 2, "densidad_relativa"].values[0]
r3 = resumen.loc[resumen["anillo"] == 3, "densidad_relativa"].values[0]

# =========================================
# DETECTAR PATRÓN
# =========================================

if a1 > a2 > a3:
    patron = "gradiente_decreciente"
    interpretacion = "Se observa una concentración de delitos en el entorno inmediato del cajero, que disminuye progresivamente con la distancia. Esto sugiere un posible efecto focalizador del delito."
elif a1 < a2 < a3:
    patron = "gradiente_creciente"
    interpretacion = "La densidad de delitos aumenta con la distancia al cajero, lo que sugiere que los cajeros se ubican en zonas ya densamente delictivas, sin evidenciar un efecto directo."
else:
    patron = "sin_patron_claro"
    interpretacion = "No se observa un patrón monotónico claro en la distribución del delito. La relación entre cajeros y delito parece depender de factores adicionales."

# =========================================
# GENERAR TEXTO
# =========================================

texto = f"""
ANÁLISIS DE DELITO EN TORNO A CAJEROS AUTOMÁTICOS
==============================================

Promedios por anillo:

Anillo 1 (0-50m):
- Densidad: {a1:.2f}
- Delitos promedio: {resumen.loc[0, 'cantidad_delitos']:.2f}

Anillo 2 (50-100m):
- Densidad: {a2:.2f}
- Relativa: {r2:.2f}

Anillo 3 (100-150m):
- Densidad: {a3:.2f}
- Relativa: {r3:.2f}

----------------------------------------------

Patrón detectado: {patron}

Interpretación:
{interpretacion}

----------------------------------------------

Observación metodológica:
El análisis se basa en densidad de delitos por km² en anillos concéntricos, lo que permite evitar sesgos por superficie y analizar gradientes espaciales del delito.

"""

# Guardar archivo
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(texto)

print(f"📄 Análisis guardado en: {OUTPUT_TXT}")