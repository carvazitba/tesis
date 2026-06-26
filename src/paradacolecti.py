from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from shapely.geometry import box

# ─────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────

BASE_DIR       = Path(__file__).resolve().parents[1]
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR     = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELITOS_PATH   = DATA_PROCESSED / "delitos_total.csv.gz"
PARADAS_PATH   = DATA_RAW / "paradas-de-colectivo.xlsx"

OUTPUT_MATRIZ  = DATA_PROCESSED / "grilla_maestra_colectivos_ml.csv"
OUTPUT_GRAFICO = OUTPUT_DIR / "relacion_colectivos_delito.png"

for p in [DELITOS_PATH, PARADAS_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"No se encontró: {p}")

# ─────────────────────────────────────────
# CARGA Y LIMPIEZA
# ─────────────────────────────────────────

print("📂 Cargando datos...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
paradas = pd.read_excel(PARADAS_PATH)

delitos.columns = delitos.columns.str.strip().str.lower()
paradas.columns = paradas.columns.str.strip().str.lower()

print(f"✅ Paradas cargadas:  {len(paradas)}")
print(f"✅ Delitos cargados:  {len(delitos)}")

# Coordenadas de delitos
delitos["latitud"]  = pd.to_numeric(delitos["latitud"],  errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()

# Coordenadas de paradas: separador decimal es coma → reemplazar
paradas["coord_x"] = (
    paradas["coord_x"].astype(str).str.replace(",", ".", regex=False)
)
paradas["coord_y"] = (
    paradas["coord_y"].astype(str).str.replace(",", ".", regex=False)
)
paradas["coord_x"] = pd.to_numeric(paradas["coord_x"], errors="coerce")
paradas["coord_y"] = pd.to_numeric(paradas["coord_y"], errors="coerce")
paradas = paradas.dropna(subset=["coord_x", "coord_y"]).copy()

print(f"✅ Paradas con coordenadas válidas: {len(paradas)}")

# ─────────────────────────────────────────
# PONDERACIÓN TEMPORAL
# ─────────────────────────────────────────

def asignar_peso(anio):
    if anio == 2023:   return 1.00
    elif anio == 2022: return 0.75
    elif anio == 2021: return 0.50
    else:              return 0.15

delitos["anio"] = pd.to_numeric(delitos["anio"], errors="coerce")
delitos["peso"] = delitos["anio"].apply(asignar_peso)

# ─────────────────────────────────────────
# GEO DATAFRAMES  →  EPSG:3857
# ─────────────────────────────────────────

print("Proyectando a sistema métrico...")

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

# coord_x = longitud, coord_y = latitud
paradas_gdf = gpd.GeoDataFrame(
    paradas,
    geometry=gpd.points_from_xy(paradas["coord_x"], paradas["coord_y"]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

# ─────────────────────────────────────────
# GRILLA 250×250m
# — misma resolución que el modelo predictivo
# — apropiada para la densidad de paradas (~1 cada 100-150m)
# ─────────────────────────────────────────

print("Creando grilla de 250m...")

xmin, ymin, xmax, ymax = delitos_gdf.total_bounds
TAM_CELDA = 250

grid_cells = [
    box(x0, y0, x0 + TAM_CELDA, y0 + TAM_CELDA)
    for x0 in np.arange(xmin, xmax, TAM_CELDA)
    for y0 in np.arange(ymin, ymax, TAM_CELDA)
]

grilla_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:3857")
grilla_gdf["id_celda"] = grilla_gdf.index
grilla_gdf["area_km2"] = grilla_gdf.geometry.area / 1e6

# ─────────────────────────────────────────
# SPATIAL JOIN
# ─────────────────────────────────────────

print("Ejecutando cruces espaciales...")

# Delitos ponderados por celda
join_delitos = gpd.sjoin(delitos_gdf, grilla_gdf, how="inner", predicate="within")
delitos_pond = (
    join_delitos.groupby("id_celda")["peso"]
    .sum()
    .reset_index(name="delitos_ponderados")
)

# Paradas por celda
join_paradas = gpd.sjoin(paradas_gdf, grilla_gdf, how="inner", predicate="within")
conteo_paradas = (
    join_paradas.groupby("id_celda")
    .size()
    .reset_index(name="cant_paradas")
)

grilla_gdf = (
    grilla_gdf
    .merge(delitos_pond,   on="id_celda", how="left")
    .merge(conteo_paradas, on="id_celda", how="left")
)

grilla_gdf[["delitos_ponderados", "cant_paradas"]] = (
    grilla_gdf[["delitos_ponderados", "cant_paradas"]].fillna(0)
)

# ─────────────────────────────────────────
# DENSIDADES
# ─────────────────────────────────────────

grilla_gdf["densidad_delitos"] = grilla_gdf["delitos_ponderados"] / grilla_gdf["area_km2"]
grilla_gdf["densidad_paradas"] = grilla_gdf["cant_paradas"]       / grilla_gdf["area_km2"]

# Celdas activas: al menos un delito o una parada
grilla_activa = grilla_gdf[
    (grilla_gdf["delitos_ponderados"] > 0) |
    (grilla_gdf["cant_paradas"]       > 0)
].copy()

print(f"✅ Celdas activas: {len(grilla_activa)}")

# ─────────────────────────────────────────
# EXPORTAR MATRIZ
# ─────────────────────────────────────────

grilla_activa.drop(columns=["geometry"]).to_csv(OUTPUT_MATRIZ, index=False)
print(f"💾 Matriz guardada en: {OUTPUT_MATRIZ}")

# ─────────────────────────────────────────
# ANÁLISIS ESTADÍSTICO
# ─────────────────────────────────────────

print("Calculando correlación de Spearman...")

corr, p_value = stats.spearmanr(
    grilla_activa["densidad_paradas"],
    grilla_activa["densidad_delitos"]
)

print("===================================================")
print(f"📊 Spearman ρ: {corr:.4f} | p-value: {p_value:.3e}")
print(f"📊 Celdas analizadas: {len(grilla_activa)}")
print(f"📊 Tamaño de celda: {TAM_CELDA}m × {TAM_CELDA}m")
print("===================================================")

# ─────────────────────────────────────────
# GRÁFICO
# ─────────────────────────────────────────

plt.figure(figsize=(10, 6))

sns.regplot(
    data=grilla_activa,
    x="densidad_paradas",
    y="densidad_delitos",
    scatter_kws={"alpha": 0.4, "s": 10},
    line_kws={"linewidth": 2}
)

plt.title(
    f"Relación entre densidad de paradas de colectivo y delitos\n"
    f"Spearman ρ = {corr:.2f} (p-value: {p_value:.3e}) | "
    f"Grilla {TAM_CELDA}m × {TAM_CELDA}m",
    fontsize=13
)
plt.xlabel("Densidad de paradas de colectivo (por km²)")
plt.ylabel("Densidad de delitos (ponderados por km²)")
plt.grid(True, linestyle="--", alpha=0.5)
sns.despine()

plt.tight_layout()
plt.savefig(OUTPUT_GRAFICO, dpi=300)
print(f"📊 Gráfico guardado en: {OUTPUT_GRAFICO}")

plt.show()