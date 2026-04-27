from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELITOS_PATH = DATA_PROCESSED / "delitos_total.csv.gz"
CAJEROS_PATH = DATA_RAW / "cajeros-automaticos.csv"

OUTPUT_CSV = OUTPUT_DIR / "clasificacion_cajeros_anillos.csv"
OUTPUT_IMG = OUTPUT_DIR / "clasificacion_cajeros_anillos.png"

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

print("Cargando datos...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
cajeros = pd.read_csv(CAJEROS_PATH, low_memory=False)

delitos.columns = delitos.columns.str.strip().str.lower()
cajeros.columns = cajeros.columns.str.strip().str.lower()

# FILTRADO METODOLÓGICO: ROBO Y HURTO

col_tipo = "tipo_delito" if "tipo_delito" in delitos.columns else "tipo"

if col_tipo in delitos.columns:
    delitos[col_tipo] = delitos[col_tipo].astype(str).str.strip().str.lower()
    delitos = delitos[delitos[col_tipo].isin(["robo", "hurto"])].copy()
    print(f"✅ Filtro aplicado: Robos y Hurtos ({len(delitos)} registros)")

# LIMPIEZA DE COORDENADAS

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")

cajeros["lat"] = pd.to_numeric(cajeros["lat"], errors="coerce")
cajeros["long"] = pd.to_numeric(cajeros["long"], errors="coerce")

delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()
cajeros = cajeros.dropna(subset=["lat", "long"]).copy()

# GEO DATAFRAMES

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

cajeros_gdf = gpd.GeoDataFrame(
    cajeros,
    geometry=gpd.points_from_xy(cajeros["long"], cajeros["lat"]),
    crs="EPSG:4326"
).to_crs("EPSG:3857")

# GENERAR 3 ANILLOS DE 50 METROS

print("Generando anillos de 50 metros...")

distancias = [0, 50, 100, 150]
anillos = []

for idx, cajero in cajeros_gdf.iterrows():
    punto = cajero.geometry

    for i in range(len(distancias) - 1):
        r_in = distancias[i]
        r_out = distancias[i + 1]

        externo = punto.buffer(r_out)
        interno = punto.buffer(r_in)
        anillo = externo.difference(interno)

        anillos.append({
            "id_cajero": idx,
            "id_original": cajero.get("id", idx),
            "banco": cajero.get("banco", "S/D"),
            "red": cajero.get("red", "S/D"),
            "barrio": cajero.get("barrio", "S/D"),
            "comuna": cajero.get("comuna", None),
            "anillo": i + 1,
            "distancia": f"{r_in}-{r_out} m",
            "area_km2": anillo.area / 1_000_000,
            "geometry": anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# SPATIAL JOIN CON DEDUPLICACIÓN

print("Cruzando delitos y eliminando superposiciones...")

delitos_gdf["id_delito"] = range(len(delitos_gdf))

join = gpd.sjoin(delitos_gdf, anillos_gdf, how="inner", predicate="within")

# Si un delito cae en varios anillos, se asigna al anillo más cercano
join = join.sort_values(by="anillo")
join = join.drop_duplicates(subset="id_delito", keep="first")

conteos = (
    join.groupby(["id_cajero", "anillo"])
    .size()
    .reset_index(name="cantidad")
)

anillos_gdf = anillos_gdf.merge(conteos, on=["id_cajero", "anillo"], how="left")
anillos_gdf["cantidad"] = anillos_gdf["cantidad"].fillna(0)

anillos_gdf["densidad"] = anillos_gdf["cantidad"] / anillos_gdf["area_km2"]

# TABLA POR CAJERO

df_res = (
    anillos_gdf
    .pivot(index="id_cajero", columns="anillo", values="densidad")
    .reset_index()
)

df_res.columns = ["id_cajero", "densidad_1", "densidad_2", "densidad_3"]

# Agregar datos del cajero
meta_cajeros = (
    anillos_gdf[["id_cajero", "id_original", "banco", "red", "barrio", "comuna"]]
    .drop_duplicates("id_cajero")
)

df_res = df_res.merge(meta_cajeros, on="id_cajero", how="left")

# CLASIFICACIÓN

def clasificar(row):
    dens = [row["densidad_1"], row["densidad_2"], row["densidad_3"]]

    if sum(dens) == 0:
        return "Sin delitos"

    max_idx = np.argmax(dens)

    if max_idx == 0:
        return "A (0-50m)"
    elif max_idx == 1:
        return "B (50-100m)"
    else:
        return "C (100-150m)"

df_res["tipo"] = df_res.apply(clasificar, axis=1)

# EXPORTAR RESULTADOS

df_res.to_csv(OUTPUT_CSV, index=False)
print(f"💾 CSV guardado en: {OUTPUT_CSV}")

# RESUMEN

df_activos = df_res[df_res["tipo"] != "Sin delitos"].copy()

resumen = df_activos["tipo"].value_counts().reset_index()
resumen.columns = ["tipo", "cantidad"]
resumen["porcentaje"] = resumen["cantidad"] / resumen["cantidad"].sum() * 100

orden = ["A (0-50m)", "B (50-100m)", "C (100-150m)"]
resumen["tipo"] = pd.Categorical(resumen["tipo"], categories=orden, ordered=True)
resumen = resumen.sort_values("tipo")

print("\n📊 Clasificación de cajeros con actividad delictiva:")
print(resumen.to_string(index=False, float_format="%.2f"))

# GRÁFICO PUBLICABLE

plt.figure(figsize=(9, 6))

ax = sns.barplot(
    data=resumen,
    x="tipo",
    y="porcentaje"
)

for i, row in resumen.reset_index(drop=True).iterrows():
    ax.text(
        i,
        row["porcentaje"] + 1,
        f"{row['porcentaje']:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

plt.title(
    "Clasificación de cajeros según anillo de máxima densidad\nRobos y hurtos",
    fontsize=14,
    pad=15
)
plt.xlabel("Zona de mayor riesgo")
plt.ylabel("Porcentaje de cajeros (%)")
plt.ylim(0, resumen["porcentaje"].max() + 10)
plt.grid(axis="y", linestyle="--", alpha=0.6)
sns.despine()

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)

print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()