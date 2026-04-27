from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from shapely.geometry import box

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELITOS_PATH = DATA_PROCESSED / "delitos_total.csv.gz"
ESTACIONES_PATH = DATA_RAW / "estaciones-de-ferrocarril.csv"
BARRIOS_PATH = DATA_RAW / "barrios.csv"

OUTPUT_MATRIZ = DATA_PROCESSED / "grilla_maestra_tren_ml.csv"
OUTPUT_GRAFICO = OUTPUT_DIR / "relacion_estaciones_tren_delito.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"DELITOS:    {DELITOS_PATH}")
print(f"ESTACIONES: {ESTACIONES_PATH}")
print(f"BARRIOS:    {BARRIOS_PATH}")
print(f"EXISTS DELITOS:    {DELITOS_PATH.exists()}")
print(f"EXISTS ESTACIONES: {ESTACIONES_PATH.exists()}")
print(f"EXISTS BARRIOS:    {BARRIOS_PATH.exists()}")

if not DELITOS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {DELITOS_PATH}")

if not ESTACIONES_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {ESTACIONES_PATH}")

if not BARRIOS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {BARRIOS_PATH}")

# CARGA DE DATOS

print("Cargando datasets...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
estaciones = pd.read_csv(ESTACIONES_PATH, low_memory=False)
barrios = pd.read_csv(BARRIOS_PATH)

# LIMPIEZA

delitos.columns = delitos.columns.str.strip().str.lower()
estaciones.columns = estaciones.columns.str.strip().str.lower()
barrios.columns = barrios.columns.str.strip().str.lower()

# Detectar columnas de coordenadas estaciones
lat_col = "latitud" if "latitud" in estaciones.columns else "lat"
lon_col = "longitud" if "longitud" in estaciones.columns else "long"

estaciones[lat_col] = pd.to_numeric(estaciones[lat_col], errors="coerce")
estaciones[lon_col] = pd.to_numeric(estaciones[lon_col], errors="coerce")

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")

delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()
estaciones = estaciones.dropna(subset=[lat_col, lon_col]).copy()

# Crear identificador de estación si no existe
if "id" not in estaciones.columns:
    estaciones = estaciones.reset_index(drop=True)
    estaciones["id_estacion"] = estaciones.index + 1
else:
    estaciones["id_estacion"] = estaciones["id"]

# FILTRAR ESTACIONES DENTRO DE CABA

print("Filtrando estaciones dentro de CABA...")

barrios["geometry"] = gpd.GeoSeries.from_wkt(barrios["geometry"])

barrios_gdf = gpd.GeoDataFrame(
    barrios,
    geometry="geometry",
    crs="EPSG:4326"
)

caba = barrios_gdf.dissolve()

estaciones_gdf_wgs = gpd.GeoDataFrame(
    estaciones,
    geometry=gpd.points_from_xy(estaciones[lon_col], estaciones[lat_col]),
    crs="EPSG:4326"
)

estaciones_gdf_wgs = gpd.sjoin(
    estaciones_gdf_wgs,
    caba,
    how="inner",
    predicate="within"
)

if "index_right" in estaciones_gdf_wgs.columns:
    estaciones_gdf_wgs = estaciones_gdf_wgs.drop(columns=["index_right"])

print(f"✅ Estaciones dentro de CABA: {len(estaciones_gdf_wgs)}")

# PONDERACIÓN TEMPORAL

def asignar_peso(anio):
    if anio == 2023:
        return 1.0
    elif anio == 2022:
        return 0.75
    elif anio == 2021:
        return 0.50
    else:
        return 0.15

delitos["anio"] = pd.to_numeric(delitos["anio"], errors="coerce")
delitos["peso"] = delitos["anio"].apply(asignar_peso)

# GEO DATAFRAMES

print("Proyectando a sistema métrico...")

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

estaciones_gdf = estaciones_gdf_wgs.to_crs("EPSG:3857")

# CREAR GRILLA

print("Creando grilla de 500m...")

xmin, ymin, xmax, ymax = delitos_gdf.total_bounds
tam_celda = 500

grid_cells = [
    box(x0, y0, x0 + tam_celda, y0 + tam_celda)
    for x0 in np.arange(xmin, xmax, tam_celda)
    for y0 in np.arange(ymin, ymax, tam_celda)
]

grilla_gdf = gpd.GeoDataFrame(
    grid_cells,
    columns=["geometry"],
    crs="EPSG:3857"
)

grilla_gdf["id_celda"] = grilla_gdf.index
grilla_gdf["area_km2"] = grilla_gdf.geometry.area / 1e6

# SPATIAL JOIN

print("Ejecutando cruces espaciales...")

join_delitos = gpd.sjoin(
    delitos_gdf,
    grilla_gdf,
    how="inner",
    predicate="within"
)

delitos_pond = (
    join_delitos.groupby("id_celda")["peso"]
    .sum()
    .reset_index(name="delitos_ponderados")
)

join_estaciones = gpd.sjoin(
    estaciones_gdf,
    grilla_gdf,
    how="inner",
    predicate="within"
)

conteo_estaciones = (
    join_estaciones.groupby("id_celda")
    .size()
    .reset_index(name="cant_estaciones_tren")
)

grilla_gdf = (
    grilla_gdf
    .merge(delitos_pond, on="id_celda", how="left")
    .merge(conteo_estaciones, on="id_celda", how="left")
)

grilla_gdf[["delitos_ponderados", "cant_estaciones_tren"]] = (
    grilla_gdf[["delitos_ponderados", "cant_estaciones_tren"]].fillna(0)
)

# DENSIDADES

grilla_gdf["densidad_delitos"] = (
    grilla_gdf["delitos_ponderados"] / grilla_gdf["area_km2"]
)

grilla_gdf["densidad_estaciones_tren"] = (
    grilla_gdf["cant_estaciones_tren"] / grilla_gdf["area_km2"]
)

grilla_activa = grilla_gdf[
    (grilla_gdf["delitos_ponderados"] > 0) |
    (grilla_gdf["cant_estaciones_tren"] > 0)
].copy()

# EXPORTAR MATRIZ

grilla_activa.drop(columns=["geometry"]).to_csv(
    OUTPUT_MATRIZ,
    index=False,
    encoding="utf-8-sig"
)

print(f"💾 Matriz guardada en: {OUTPUT_MATRIZ}")

# ANÁLISIS ESTADÍSTICO

print("Calculando correlación...")

corr, p_value = stats.spearmanr(
    grilla_activa["densidad_estaciones_tren"],
    grilla_activa["densidad_delitos"]
)

print(f"📊 Spearman: {corr:.3f} | p-value: {p_value:.3e}")

# GRÁFICO PUBLICABLE

plt.figure(figsize=(10, 6))

sns.regplot(
    data=grilla_activa,
    x="densidad_estaciones_tren",
    y="densidad_delitos",
    scatter_kws={"alpha": 0.5},
    line_kws={"linewidth": 2}
)

plt.title(
    f"Relación entre densidad de estaciones ferroviarias y delitos\n"
    f"Spearman: {corr:.2f} (p-value: {p_value:.3e})",
    fontsize=14
)

plt.xlabel("Densidad de estaciones ferroviarias (por km²)")
plt.ylabel("Densidad de delitos (ponderados por km²)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_GRAFICO, dpi=300)

print(f"📊 Gráfico guardado en: {OUTPUT_GRAFICO}")

plt.show()