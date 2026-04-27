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
CAJEROS_PATH = DATA_RAW / "cajeros-automaticos.csv"

OUTPUT_HTML = OUTPUT_DIR / "mapa_cajeros_3anillos_50m.html"
OUTPUT_CSV = OUTPUT_DIR / "anillos_cajeros_3anillos_50m.csv"
OUTPUT_TXT = OUTPUT_DIR / "analisis_cajeros_3anillos_50m.txt"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"DELITOS: {DELITOS_PATH}")
print(f"CAJEROS: {CAJEROS_PATH}")
print(f"EXISTS DELITOS: {DELITOS_PATH.exists()}")
print(f"EXISTS CAJEROS: {CAJEROS_PATH.exists()}")

if not DELITOS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {DELITOS_PATH}")

if not CAJEROS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {CAJEROS_PATH}")

# CARGA DE DATOS

print("📂 Cargando delitos...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)

print("📂 Cargando cajeros...")
cajeros = pd.read_csv(CAJEROS_PATH, low_memory=False)

# LIMPIEZA

delitos.columns = delitos.columns.str.strip().str.lower()
cajeros.columns = cajeros.columns.str.strip().str.lower()

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()

cajeros["lat"] = pd.to_numeric(cajeros["lat"], errors="coerce")
cajeros["long"] = pd.to_numeric(cajeros["long"], errors="coerce")
cajeros = cajeros.dropna(subset=["lat", "long"]).copy()

# GEO DATAFRAMES

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

# PROYECCIÓN MÉTRICA

delitos_m = delitos_gdf.to_crs(epsg=3857)
cajeros_m = cajeros_gdf.to_crs(epsg=3857)

# CREAR 3 ANILLOS DE 50 METROS

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

# SPATIAL JOIN

print("📍 Calculando delitos...")
join = gpd.sjoin(delitos_m, anillos_gdf, how="inner", predicate="within")

conteos = (
    join.groupby(["id", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(conteos, on=["id", "anillo"], how="left")
anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)

# DENSIDADES

anillos_gdf["area_km2"] = anillos_gdf.geometry.area / 1_000_000
anillos_gdf["densidad"] = anillos_gdf["cantidad_delitos"] / anillos_gdf["area_km2"]

base = anillos_gdf[anillos_gdf["anillo"] == 1][["id", "densidad"]].rename(
    columns={"densidad": "base"}
)

anillos_gdf = anillos_gdf.merge(base, on="id", how="left")

anillos_gdf["densidad_relativa"] = np.where(
    anillos_gdf["base"] > 0,
    anillos_gdf["densidad"] / anillos_gdf["base"],
    np.nan
)

# EXPORTAR CSV

anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
print(f"💾 CSV guardado en: {OUTPUT_CSV}")

# MAPA

anillos_wgs = anillos_gdf.to_crs(epsg=4326)

mapa = folium.Map(
    location=[-34.61, -58.43],
    zoom_start=12,
    tiles="cartodbpositron"
)

vals = anillos_wgs["densidad_relativa"].dropna()

if len(vals) == 0:
    p20, p80 = 0.8, 1.2
else:
    p20 = vals.quantile(0.2)
    p80 = vals.quantile(0.8)
    p20 = min(p20, 1)
    p80 = max(p80, 1)

colormap = cm.LinearColormap(
    ["green", "yellow", "red"],
    vmin=p20,
    vmax=p80
)
colormap.caption = "Densidad relativa respecto al primer anillo"
colormap.add_to(mapa)

def clip(v):
    if pd.isna(v):
        return None
    return max(min(v, p80), p20)

for _, row in anillos_wgs.iterrows():
    val = clip(row["densidad_relativa"])
    color = "#cccccc" if val is None else colormap(val)

    tooltip = f"""
    Banco: {row['banco']}<br>
    Red: {row['red']}<br>
    Ubicación: {row['ubicacion']}<br>
    Barrio: {row['barrio']}<br>
    Comuna: {row['comuna']}<br>
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
        tooltip=folium.Tooltip(tooltip)
    ).add_to(mapa)

for _, row in cajeros.iterrows():
    tooltip = f"""
    Banco: {row.get('banco', 'S/D')}<br>
    Red: {row.get('red', 'S/D')}<br>
    Ubicación: {row.get('ubicacion', '')}<br>
    Barrio: {row.get('barrio', 'S/D')}<br>
    Comuna: {row.get('comuna', 'S/D')}
    """

    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=3,
        color="black",
        fill=True,
        fill_color="black",
        fill_opacity=1,
        tooltip=folium.Tooltip(tooltip)
    ).add_to(mapa)

mapa.save(OUTPUT_HTML)
print(f"🗺️ Mapa generado en: {OUTPUT_HTML}")

# ANÁLISIS TEXTUAL AUTOMÁTICO

print("🧠 Generando análisis textual...")

resumen = (
    anillos_gdf.groupby("anillo")
    .agg(
        distancia=("distancia", "first"),
        delitos_totales=("cantidad_delitos", "sum"),
        densidad_promedio=("densidad", "mean"),
        densidad_relativa_promedio=("densidad_relativa", "mean"),
        delitos_promedio=("cantidad_delitos", "mean")
    )
    .reset_index()
)

a1 = resumen.loc[resumen["anillo"] == 1, "densidad_promedio"].values[0]
a2 = resumen.loc[resumen["anillo"] == 2, "densidad_promedio"].values[0]
a3 = resumen.loc[resumen["anillo"] == 3, "densidad_promedio"].values[0]

r2_global = a2 / a1 if a1 > 0 else np.nan
r3_global = a3 / a1 if a1 > 0 else np.nan

if a1 > a2 > a3:
    patron = "gradiente_decreciente"
    interpretacion = (
        "Se observa una concentración de delitos en el entorno inmediato del cajero, "
        "con una disminución progresiva de la densidad a medida que aumenta la distancia."
    )
elif a1 < a2 < a3:
    patron = "gradiente_creciente"
    interpretacion = (
        "La densidad de delitos aumenta con la distancia al cajero, lo que sugiere que "
        "los cajeros se ubican en zonas ya densamente delictivas."
    )
else:
    patron = "sin_patron_monotono"
    interpretacion = (
        "No se observa un patrón estrictamente monotónico, lo que sugiere heterogeneidad "
        "espacial y posible influencia del contexto urbano."
    )

texto = f"""
ANÁLISIS DE DELITO EN TORNO A CAJEROS AUTOMÁTICOS
=================================================

Promedios por anillo:

Anillo 1 (0-50 m):
- Delitos totales: {resumen.loc[resumen["anillo"] == 1, "delitos_totales"].values[0]:.0f}
- Densidad promedio: {a1:.2f} delitos/km²
- Densidad relativa global: 1.00

Anillo 2 (50-100 m):
- Delitos totales: {resumen.loc[resumen["anillo"] == 2, "delitos_totales"].values[0]:.0f}
- Densidad promedio: {a2:.2f} delitos/km²
- Densidad relativa global: {r2_global:.2f}

Anillo 3 (100-150 m):
- Delitos totales: {resumen.loc[resumen["anillo"] == 3, "delitos_totales"].values[0]:.0f}
- Densidad promedio: {a3:.2f} delitos/km²
- Densidad relativa global: {r3_global:.2f}

-------------------------------------------------

Patrón detectado: {patron}

Interpretación:
{interpretacion}

-------------------------------------------------

Observación metodológica:
El análisis se basa en densidad de delitos por km² en tres anillos concéntricos
de 50 metros alrededor de cajeros automáticos. La normalización por superficie
permite comparar zonas de distinto tamaño y evaluar gradientes espaciales del delito.

Archivo CSV generado:
{OUTPUT_CSV}

Mapa interactivo generado:
{OUTPUT_HTML}
"""

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(texto)

print(f"📄 Análisis guardado en: {OUTPUT_TXT}")