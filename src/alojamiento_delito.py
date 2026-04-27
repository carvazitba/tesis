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
ALOJ_PATH = DATA_RAW / "alojamientos-geocodificados.csv"

OUTPUT_MATRIZ = DATA_PROCESSED / "grilla_maestra_ml.csv"
OUTPUT_GRAFICO = OUTPUT_DIR / "relacion_alojamientos_delitos.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"DELITOS: {DELITOS_PATH}")
print(f"ALOJAMIENTOS: {ALOJ_PATH}")
print(f"EXISTS DELITOS: {DELITOS_PATH.exists()}")
print(f"EXISTS ALOJAMIENTOS: {ALOJ_PATH.exists()}")

# CARGA DE DATOS

if not DELITOS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {DELITOS_PATH}")

if not ALOJ_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {ALOJ_PATH}")

print("Cargando datasets...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
alojamientos = pd.read_csv(ALOJ_PATH, encoding="latin1", sep=",", low_memory=False)

# LIMPIEZA

delitos.columns = delitos.columns.str.strip().str.lower()
alojamientos.columns = alojamientos.columns.str.strip().str.lower()

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")

alojamientos["latitud"] = pd.to_numeric(alojamientos["latitud"], errors="coerce")
alojamientos["longitud"] = pd.to_numeric(alojamientos["longitud"], errors="coerce")

delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()
alojamientos = alojamientos.dropna(subset=["latitud", "longitud"]).copy()

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

# GEODATAFRAMES

print("Proyectando a sistema métrico (EPSG:3857)...")

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

aloj_gdf = gpd.GeoDataFrame(
    alojamientos,
    geometry=gpd.points_from_xy(alojamientos["longitud"], alojamientos["latitud"]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

# CREAR GRILLA 500 x 500 m

print("Creando grilla regular de 500x500 metros...")

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

# SPATIAL JOINS

print("Ejecutando cruces espaciales...")

join_delitos = gpd.sjoin(delitos_gdf, grilla_gdf, how="inner", predicate="within")

delitos_ponderados = (
    join_delitos
    .groupby("id_celda")["peso"]
    .sum()
    .reset_index(name="delitos_ponderados")
)

join_aloj = gpd.sjoin(aloj_gdf, grilla_gdf, how="inner", predicate="within")

conteo_aloj = (
    join_aloj
    .groupby("id_celda")
    .size()
    .reset_index(name="cant_alojamientos")
)

grilla_gdf = (
    grilla_gdf
    .merge(delitos_ponderados, on="id_celda", how="left")
    .merge(conteo_aloj, on="id_celda", how="left")
)

grilla_gdf[["delitos_ponderados", "cant_alojamientos"]] = (
    grilla_gdf[["delitos_ponderados", "cant_alojamientos"]].fillna(0)
)

# DENSIDADES

grilla_gdf["densidad_delitos"] = grilla_gdf["delitos_ponderados"] / grilla_gdf["area_km2"]
grilla_gdf["densidad_alojamientos"] = grilla_gdf["cant_alojamientos"] / grilla_gdf["area_km2"]

grilla_activa = grilla_gdf[
    (grilla_gdf["delitos_ponderados"] > 0) |
    (grilla_gdf["cant_alojamientos"] > 0)
].copy()

# EXPORTAR MATRIZ

grilla_activa.drop(columns=["geometry"]).to_csv(OUTPUT_MATRIZ, index=False)
print(f"✅ Matriz Maestra guardada en: {OUTPUT_MATRIZ}")

# CORRELACIÓN + GRÁFICO

print("Generando gráfico de dispersión...")

corr, p_value = stats.spearmanr(
    grilla_activa["densidad_alojamientos"],
    grilla_activa["densidad_delitos"]
)

plt.figure(figsize=(10, 6))

sns.regplot(
    data=grilla_activa,
    x="densidad_alojamientos",
    y="densidad_delitos",
    scatter_kws={"alpha": 0.5},
    line_kws={"linewidth": 2}
)

plt.title(
    f"Relación entre Densidad de Alojamientos y Delitos\n"
    f"Correlación de Spearman: {corr:.2f} (p-value: {p_value:.3e})",
    fontsize=14
)
plt.xlabel("Densidad de Alojamientos (por km²)")
plt.ylabel("Densidad de Delitos (ponderados por km²)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(OUTPUT_GRAFICO, dpi=300)
print(f"📊 Gráfico guardado en: {OUTPUT_GRAFICO}")

plt.show()