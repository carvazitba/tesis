# ============================================================
# MAPA DE DENSIDAD DELICTIVA POR GRILLA - CABA
# ============================================================
# Lee desde:
#   tesis/data/processed/delitos_total.csv.gz
#
# Genera:
#   tesis/outputs/mapa_densidad_delitos_grilla.html
#   tesis/outputs/resumen_mapa_densidad_delitos_grilla.txt
#
# Ejecutar desde la raíz del proyecto:
#   python src/eda_mapa_densidad_delitos_grilla.py
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
from shapely.geometry import box


# ============================================================
# 1) RUTAS REPRODUCIBLES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_HTML = OUTPUT_DIR / "mapa_densidad_delitos_grilla.html"
OUTPUT_TXT = OUTPUT_DIR / "resumen_mapa_densidad_delitos_grilla.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("===================================================")
print("CONFIGURACIÓN DE RUTAS")
print("===================================================")
print(f"BASE_DIR:    {BASE_DIR}")
print(f"INPUT_FILE:  {INPUT_FILE}")
print(f"OUTPUT_HTML: {OUTPUT_HTML}")
print(f"OUTPUT_TXT:  {OUTPUT_TXT}")
print(f"EXISTS:      {INPUT_FILE.exists()}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo:\n{INPUT_FILE}\n\n"
        "Verificá que el dataset esté en:\n"
        "data/processed/delitos_total.csv.gz"
    )


# ============================================================
# 2) CARGA DEL DATASET
# ============================================================

print("\n📂 Cargando dataset de delitos...")

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

print(f"Registros cargados: {len(df):,}")
print("Columnas:")
print(list(df.columns))

if "latitud" not in df.columns or "longitud" not in df.columns:
    raise ValueError(
        "El dataset debe tener columnas 'latitud' y 'longitud'. "
        f"Columnas encontradas: {list(df.columns)}"
    )


# ============================================================
# 3) LIMPIEZA DE COORDENADAS
# ============================================================

print("\n🧹 Limpiando coordenadas...")

n_inicial = len(df)

df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

df = df.dropna(subset=["latitud", "longitud"]).copy()

df = df[
    (df["latitud"] != 0) &
    (df["longitud"] != 0)
].copy()

# Filtro aproximado para CABA
df = df[
    df["latitud"].between(-34.75, -34.50) &
    df["longitud"].between(-58.60, -58.30)
].copy()

n_validos = len(df)
n_descartados = n_inicial - n_validos

print(f"Registros iniciales:              {n_inicial:,}")
print(f"Registros con coordenadas válidas: {n_validos:,}")
print(f"Registros descartados:             {n_descartados:,}")

if df.empty:
    raise ValueError("No quedaron registros válidos luego de limpiar coordenadas.")


# ============================================================
# 4) CONVERTIR A GEODATAFRAME
# ============================================================

print("\n🌎 Creando GeoDataFrame...")

delitos_gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
    crs="EPSG:4326"
)

# Proyección métrica para trabajar en metros.
# EPSG:32721 = WGS 84 / UTM zone 21S.
# Es adecuada para Buenos Aires y permite calcular áreas/distancias en metros.
delitos_m = delitos_gdf.to_crs("EPSG:32721")


# ============================================================
# 5) CREAR GRILLA REGULAR
# ============================================================

print("\n🧩 Creando grilla regular...")

# Tamaño de celda en metros.
# 250 = más detalle.
# 500 = más general y más liviano.
TAM_CELDA = 250

xmin, ymin, xmax, ymax = delitos_m.total_bounds

grid_cells = []

x_coords = np.arange(xmin, xmax + TAM_CELDA, TAM_CELDA)
y_coords = np.arange(ymin, ymax + TAM_CELDA, TAM_CELDA)

for x0 in x_coords:
    for y0 in y_coords:
        grid_cells.append(box(x0, y0, x0 + TAM_CELDA, y0 + TAM_CELDA))

grilla = gpd.GeoDataFrame(
    {"geometry": grid_cells},
    crs="EPSG:32721"
)

grilla["id_celda"] = grilla.index
grilla["area_km2"] = grilla.geometry.area / 1_000_000

print(f"Celdas generadas: {len(grilla):,}")


# ============================================================
# 6) CONTAR DELITOS POR CELDA
# ============================================================

print("\n📍 Calculando delitos por celda...")

join = gpd.sjoin(
    delitos_m[["geometry"]],
    grilla[["id_celda", "geometry"]],
    how="inner",
    predicate="within"
)

conteo = (
    join.groupby("id_celda")
    .size()
    .reset_index(name="cantidad_delitos")
)

grilla = grilla.merge(conteo, on="id_celda", how="left")
grilla["cantidad_delitos"] = grilla["cantidad_delitos"].fillna(0).astype(int)

grilla["densidad_delitos_km2"] = (
    grilla["cantidad_delitos"] / grilla["area_km2"]
)

grilla_activa = grilla[grilla["cantidad_delitos"] > 0].copy()

print(f"Celdas con al menos un delito: {len(grilla_activa):,}")

if grilla_activa.empty:
    raise ValueError("No hay celdas con delitos para graficar.")


# ============================================================
# 7) ESCALA DE COLORES MEJORADA
# ============================================================
# Se usa transformación logarítmica para comprimir valores extremos.
# Esto mejora el contraste visual entre zonas de baja, media y alta densidad.

print("\n🎨 Calculando escala visual...")

grilla_activa["log_densidad"] = np.log1p(grilla_activa["densidad_delitos_km2"])

# Recorte por percentiles para que valores extremos no saturen la escala
p05_log = grilla_activa["log_densidad"].quantile(0.05)
p95_log = grilla_activa["log_densidad"].quantile(0.95)

grilla_activa["log_densidad_clip"] = grilla_activa["log_densidad"].clip(
    lower=p05_log,
    upper=p95_log
)

print(f"Percentil 5 log-densidad:  {p05_log:.4f}")
print(f"Percentil 95 log-densidad: {p95_log:.4f}")


# ============================================================
# 8) ESTADÍSTICAS PARA RESUMEN
# ============================================================

total_delitos = int(grilla_activa["cantidad_delitos"].sum())

total_celdas = int(len(grilla))
celdas_activas = int(len(grilla_activa))
celdas_sin_delitos = total_celdas - celdas_activas

porcentaje_celdas_activas = celdas_activas / total_celdas * 100
porcentaje_celdas_sin_delitos = celdas_sin_delitos / total_celdas * 100

densidad_media = grilla_activa["densidad_delitos_km2"].mean()
densidad_mediana = grilla_activa["densidad_delitos_km2"].median()
densidad_max = grilla_activa["densidad_delitos_km2"].max()

p25 = grilla_activa["densidad_delitos_km2"].quantile(0.25)
p50 = grilla_activa["densidad_delitos_km2"].quantile(0.50)
p75 = grilla_activa["densidad_delitos_km2"].quantile(0.75)
p90 = grilla_activa["densidad_delitos_km2"].quantile(0.90)
p95_densidad = grilla_activa["densidad_delitos_km2"].quantile(0.95)
p99 = grilla_activa["densidad_delitos_km2"].quantile(0.99)

umbral_top_5 = p95_densidad

top_5 = grilla_activa[
    grilla_activa["densidad_delitos_km2"] >= umbral_top_5
].copy()

top_10_celdas = (
    grilla_activa
    .sort_values("densidad_delitos_km2", ascending=False)
    .head(10)
    .copy()
)


# ============================================================
# 9) PASAR A WGS84 PARA FOLIUM
# ============================================================

grilla_wgs = grilla_activa.to_crs("EPSG:4326")
top_5_wgs = top_5.to_crs("EPSG:4326")


# ============================================================
# 10) CREAR MAPA BASE
# ============================================================

print("\n🗺️ Generando mapa...")

CENTRO_CABA = [-34.6037, -58.3816]

mapa = folium.Map(
    location=CENTRO_CABA,
    zoom_start=12,
    tiles="cartodbpositron",
    control_scale=True
)


# ============================================================
# 11) CREAR ESCALA DE COLORES
# ============================================================

colormap = cm.LinearColormap(
    colors=[
        "#ffffcc",  # amarillo muy claro
        "#ffeda0",
        "#fed976",
        "#feb24c",
        "#fd8d3c",
        "#f03b20",
        "#bd0026"   # rojo oscuro
    ],
    vmin=p05_log,
    vmax=p95_log
)

colormap.caption = (
    "Densidad delictiva por celda "
    f"({TAM_CELDA} m x {TAM_CELDA} m) - escala logarítmica"
)

colormap.add_to(mapa)


# ============================================================
# 12) FUNCIÓN DE ESTILO
# ============================================================

def style_function(feature):
    valor = feature["properties"]["log_densidad_clip"]

    return {
        "fillColor": colormap(valor),
        "color": "#555555",
        "weight": 0.2,
        "fillOpacity": 0.70,
    }


# ============================================================
# 13) AGREGAR GRILLA AL MAPA
# ============================================================

tooltip = folium.GeoJsonTooltip(
    fields=[
        "id_celda",
        "cantidad_delitos",
        "densidad_delitos_km2",
    ],
    aliases=[
        "ID celda:",
        "Cantidad de delitos:",
        "Densidad delitos/km²:",
    ],
    localize=True,
    sticky=False,
    labels=True
)

folium.GeoJson(
    grilla_wgs,
    name="Densidad delictiva por grilla",
    style_function=style_function,
    tooltip=tooltip,
).add_to(mapa)


# ============================================================
# 14) RESALTAR TOP 5% DE CELDAS
# ============================================================

folium.GeoJson(
    top_5_wgs,
    name="Top 5% mayor densidad",
    style_function=lambda feature: {
        "fillColor": "none",
        "color": "black",
        "weight": 1.5,
        "fillOpacity": 0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "id_celda",
            "cantidad_delitos",
            "densidad_delitos_km2",
        ],
        aliases=[
            "ID celda:",
            "Cantidad de delitos:",
            "Densidad delitos/km²:",
        ],
        localize=True,
        sticky=False,
        labels=True
    )
).add_to(mapa)


# ============================================================
# 15) CONTROL DE CAPAS
# ============================================================

folium.LayerControl(collapsed=False).add_to(mapa)


# ============================================================
# 16) GENERAR RESUMEN TXT PARA TESIS
# ============================================================

print("\n📝 Generando resumen estadístico del mapa...")

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("===================================================\n")
    f.write("RESUMEN DEL MAPA DE DENSIDAD DELICTIVA POR GRILLA\n")
    f.write("===================================================\n\n")

    f.write("1. INFORMACIÓN GENERAL\n")
    f.write("----------------------\n")
    f.write(f"Archivo de entrada: {INPUT_FILE}\n")
    f.write(f"Archivo HTML generado: {OUTPUT_HTML}\n")
    f.write(f"Tamaño de celda utilizado: {TAM_CELDA} m x {TAM_CELDA} m\n")
    f.write("Sistema de referencia métrico utilizado: EPSG:32721\n")
    f.write("Sistema de referencia de salida para visualización: EPSG:4326\n")
    f.write("Transformación visual aplicada: log1p(densidad_delitos_km2)\n")
    f.write("Escala visual acotada por percentiles: 5% - 95%\n\n")

    f.write("2. REGISTROS PROCESADOS\n")
    f.write("-----------------------\n")
    f.write(f"Registros iniciales del dataset: {n_inicial:,}\n")
    f.write(f"Registros con coordenadas válidas en CABA: {n_validos:,}\n")
    f.write(f"Registros descartados por coordenadas: {n_descartados:,}\n")
    f.write(f"Total de delitos incorporados al análisis espacial: {total_delitos:,}\n\n")

    f.write("3. DISTRIBUCIÓN DE CELDAS\n")
    f.write("-------------------------\n")
    f.write(f"Total de celdas generadas: {total_celdas:,}\n")
    f.write(f"Celdas con al menos un delito: {celdas_activas:,}\n")
    f.write(f"Celdas sin delitos: {celdas_sin_delitos:,}\n")
    f.write(f"Porcentaje de celdas activas: {porcentaje_celdas_activas:.2f}%\n")
    f.write(f"Porcentaje de celdas sin delitos: {porcentaje_celdas_sin_delitos:.2f}%\n\n")

    f.write("4. ESTADÍSTICAS DE DENSIDAD DELICTIVA\n")
    f.write("-------------------------------------\n")
    f.write("La densidad se expresa como delitos por km² dentro de cada celda activa.\n\n")
    f.write(f"Densidad media: {densidad_media:,.2f} delitos/km²\n")
    f.write(f"Densidad mediana: {densidad_mediana:,.2f} delitos/km²\n")
    f.write(f"Densidad máxima: {densidad_max:,.2f} delitos/km²\n\n")

    f.write("Percentiles de densidad delictiva:\n")
    f.write(f"Percentil 25: {p25:,.2f} delitos/km²\n")
    f.write(f"Percentil 50: {p50:,.2f} delitos/km²\n")
    f.write(f"Percentil 75: {p75:,.2f} delitos/km²\n")
    f.write(f"Percentil 90: {p90:,.2f} delitos/km²\n")
    f.write(f"Percentil 95: {p95_densidad:,.2f} delitos/km²\n")
    f.write(f"Percentil 99: {p99:,.2f} delitos/km²\n\n")

    f.write("5. ESCALA VISUAL DEL MAPA\n")
    f.write("-------------------------\n")
    f.write("Para mejorar la interpretación visual, se aplicó una transformación logarítmica.\n")
    f.write("Esto reduce el peso de valores extremos y permite distinguir mejor zonas de densidad media.\n\n")
    f.write(f"Percentil 5 de log-densidad usado para la escala: {p05_log:.4f}\n")
    f.write(f"Percentil 95 de log-densidad usado para la escala: {p95_log:.4f}\n\n")

    f.write("6. UMBRAL DE CELDAS CRÍTICAS\n")
    f.write("----------------------------\n")
    f.write("Se consideraron críticas las celdas ubicadas en el 5% superior de densidad delictiva.\n")
    f.write(f"Umbral top 5%: {umbral_top_5:,.2f} delitos/km²\n")
    f.write(f"Cantidad de celdas en top 5%: {len(top_5):,}\n\n")

    f.write("7. TOP 10 CELDAS CON MAYOR DENSIDAD DELICTIVA\n")
    f.write("---------------------------------------------\n")

    for _, row in top_10_celdas.iterrows():
        f.write(
            f"ID celda: {int(row['id_celda'])} | "
            f"Cantidad delitos: {int(row['cantidad_delitos'])} | "
            f"Densidad: {row['densidad_delitos_km2']:,.2f} delitos/km²\n"
        )

    f.write("\n8. INTERPRETACIÓN PRELIMINAR\n")
    f.write("----------------------------\n")
    f.write(
        "El mapa permite observar la distribución espacial de la densidad delictiva "
        "a partir de una grilla regular. Las celdas con mayor densidad representan "
        "zonas de concentración relativa de delitos y pueden ser interpretadas como "
        "hotspots espaciales preliminares. La utilización de una escala logarítmica "
        "permite mejorar la visualización de diferencias internas, evitando que los "
        "valores extremos oculten la variabilidad existente en zonas de densidad media.\n"
    )

print(f"📝 Resumen TXT generado en: {OUTPUT_TXT}")


# ============================================================
# 17) GUARDAR MAPA
# ============================================================

mapa.save(OUTPUT_HTML)

print("\n===================================================")
print("PROCESO FINALIZADO")
print("===================================================")
print("✅ Mapa de densidad generado correctamente.")
print(f"🗺️ Archivo HTML guardado en: {OUTPUT_HTML}")
print(f"📝 Resumen TXT guardado en: {OUTPUT_TXT}")