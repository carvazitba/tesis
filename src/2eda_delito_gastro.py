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
GASTRO_PATH = DATA_RAW / "oferta_gastronomica.xlsx"

OUTPUT_MATRIZ = DATA_PROCESSED / "grilla_maestra_gastro_ml.csv"
OUTPUT_GRAFICO = OUTPUT_DIR / "relacion_gastronomia_delito.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"DELITOS: {DELITOS_PATH}")
print(f"GASTRO:  {GASTRO_PATH}")
print(f"EXISTS DELITOS: {DELITOS_PATH.exists()}")
print(f"EXISTS GASTRO:  {GASTRO_PATH.exists()}")

if not DELITOS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {DELITOS_PATH}")

if not GASTRO_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {GASTRO_PATH}")


# CARGA DE DATOS


print("Cargando datasets...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
gastronomia = pd.read_excel(GASTRO_PATH)


# LIMPIEZA


delitos.columns = delitos.columns.str.strip().str.lower()
gastronomia.columns = gastronomia.columns.str.strip().str.lower()

# Detectar columnas de coordenadas
lat_col = "latitud" if "latitud" in gastronomia.columns else "lat"
lon_col = "longitud" if "longitud" in gastronomia.columns else "long"

gastronomia[lat_col] = pd.to_numeric(gastronomia[lat_col], errors="coerce")
gastronomia[lon_col] = pd.to_numeric(gastronomia[lon_col], errors="coerce")

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")

delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()
gastronomia = gastronomia.dropna(subset=[lat_col, lon_col]).copy()

# PONDERACIÓN TEMPORAL

def asignar_peso(anio):
    if anio == 2023: return 1.0
    elif anio == 2022: return 0.75
    elif anio == 2021: return 0.50
    else: return 0.15

delitos["anio"] = pd.to_numeric(delitos["anio"], errors="coerce")
delitos["peso"] = delitos["anio"].apply(asignar_peso)

# GEO DATAFRAMES

print("Proyectando a sistema métrico...")

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

gastro_gdf = gpd.GeoDataFrame(
    gastronomia,
    geometry=gpd.points_from_xy(gastronomia[lon_col], gastronomia[lat_col]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

# CREAR GRILLA

print("Creando grilla de 500m...")

xmin, ymin, xmax, ymax = delitos_gdf.total_bounds
tam_celda = 500

grid_cells = [
    box(x0, y0, x0 + tam_celda, y0 + tam_celda)
    for x0 in np.arange(xmin, xmax, tam_celda)
    for y0 in np.arange(ymin, ymax, tam_celda)
]

grilla_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:3857")
grilla_gdf["id_celda"] = grilla_gdf.index
grilla_gdf["area_km2"] = grilla_gdf.geometry.area / 1e6

# SPATIAL JOIN

print("Ejecutando cruces espaciales...")

join_delitos = gpd.sjoin(delitos_gdf, grilla_gdf, how="inner", predicate="within")
delitos_pond = (
    join_delitos.groupby("id_celda")["peso"]
    .sum()
    .reset_index(name="delitos_ponderados")
)

join_gastro = gpd.sjoin(gastro_gdf, grilla_gdf, how="inner", predicate="within")
conteo_gastro = (
    join_gastro.groupby("id_celda")
    .size()
    .reset_index(name="cant_gastronomia")
)

grilla_gdf = (
    grilla_gdf
    .merge(delitos_pond, on="id_celda", how="left")
    .merge(conteo_gastro, on="id_celda", how="left")
)

grilla_gdf[["delitos_ponderados", "cant_gastronomia"]] = (
    grilla_gdf[["delitos_ponderados", "cant_gastronomia"]].fillna(0)
)


# DENSIDADES


grilla_gdf["densidad_delitos"] = grilla_gdf["delitos_ponderados"] / grilla_gdf["area_km2"]
grilla_gdf["densidad_gastronomia"] = grilla_gdf["cant_gastronomia"] / grilla_gdf["area_km2"]

grilla_activa = grilla_gdf[
    (grilla_gdf["delitos_ponderados"] > 0) |
    (grilla_gdf["cant_gastronomia"] > 0)
].copy()

# EXPORTAR MATRIZ

grilla_activa.drop(columns=["geometry"]).to_csv(OUTPUT_MATRIZ, index=False)
print(f"💾 Matriz guardada en: {OUTPUT_MATRIZ}")

# ANÁLISIS ESTADÍSTICO

print("Calculando correlación...")

corr, p_value = stats.spearmanr(
    grilla_activa["densidad_gastronomia"],
    grilla_activa["densidad_delitos"]
)

print(f"📊 Spearman: {corr:.3f} | p-value: {p_value:.3e}")

# GRÁFICO PUBLICABLE

plt.figure(figsize=(10, 6))

sns.regplot(
    data=grilla_activa,
    x="densidad_gastronomia",
    y="densidad_delitos",
    scatter_kws={"alpha": 0.5},
    line_kws={"linewidth": 2}
)

plt.title(
    f"Relación entre densidad gastronómica y delitos\n"
    f"Spearman: {corr:.2f} (p-value: {p_value:.3e})",
    fontsize=14
)

plt.xlabel("Densidad gastronómica (por km²)")
plt.ylabel("Densidad de delitos (ponderados por km²)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_GRAFICO, dpi=300)

print(f"📊 Gráfico guardado en: {OUTPUT_GRAFICO}")

plt.show()