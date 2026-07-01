# ==================== 2eda_cajero.py ====================
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

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"DELITOS: {DELITOS_PATH}")
print(f"CAJEROS: {CAJEROS_PATH}")
print(f"EXISTS DELITOS: {DELITOS_PATH.exists()}")
print(f"EXISTS CAJEROS: {CAJEROS_PATH.exists()}")

# CARGA DE DATOS

if not DELITOS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {DELITOS_PATH}")

if not CAJEROS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {CAJEROS_PATH}")

print("📂 Cargando delitos...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)

print("📂 Cargando cajeros...")
cajeros = pd.read_csv(CAJEROS_PATH, low_memory=False)

# LIMPIEZA Y FILTRADO

delitos.columns = delitos.columns.str.strip().str.lower()
cajeros.columns = cajeros.columns.str.strip().str.lower()

# Filtrar robos y hurtos
col_tipo = 'tipo_delito' if 'tipo_delito' in delitos.columns else 'tipo'

if col_tipo in delitos.columns:
    delitos[col_tipo] = delitos[col_tipo].astype(str).str.strip().str.lower()
    delitos = delitos[delitos[col_tipo].isin(['robo', 'hurto'])]
    print(f"✅ Filtro aplicado (robo/hurto): {len(delitos)} registros")

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"])

cajeros["lat"] = pd.to_numeric(cajeros["lat"], errors="coerce")
cajeros["long"] = pd.to_numeric(cajeros["long"], errors="coerce")
cajeros = cajeros.dropna(subset=["lat", "long"])

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

delitos_m = delitos_gdf.to_crs(epsg=3857)
cajeros_m = cajeros_gdf.to_crs(epsg=3857)

# CREAR ANILLOS (3 de 50m)

distancias = [0, 50, 100, 150]
anillos = []

for _, row in cajeros_m.iterrows():
    for i in range(len(distancias) - 1):
        externo = row.geometry.buffer(distancias[i+1])
        interno = row.geometry.buffer(distancias[i])
        anillo = externo.difference(interno)

        anillos.append({
            "id": row["id"],
            "banco": row.get("banco", "S/D"),
            "barrio": row.get("barrio", "S/D"),
            "anillo": i + 1,
            "distancia": f"{distancias[i]}-{distancias[i+1]} m",
            "geometry": anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# SPATIAL JOIN (sin doble conteo)

print("📍 Cruzando delitos y asignando al cajero más cercano...")

delitos_m["id_delito"] = range(len(delitos_m))

join = gpd.sjoin(delitos_m, anillos_gdf, how="inner", predicate="within")

# Ordenar por anillo → prioriza cercanía
join = join.sort_values(by="anillo")
join = join.drop_duplicates(subset="id_delito", keep="first")

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

base = anillos_gdf[anillos_gdf["anillo"] == 1][["id", "densidad"]]
base = base.rename(columns={"densidad": "base"})

anillos_gdf = anillos_gdf.merge(base, on="id", how="left")

anillos_gdf["densidad_relativa"] = np.where(
    anillos_gdf["base"] > 0,
    anillos_gdf["densidad"] / anillos_gdf["base"],
    np.nan
)

# EXPORT CSV

anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
print(f"💾 CSV guardado en: {OUTPUT_CSV}")

# MAPA

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

for _, row in cajeros.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=3,
        color="black",
        fill=True
    ).add_to(mapa)

mapa.save(OUTPUT_HTML)
print(f"🗺️ Mapa generado en: {OUTPUT_HTML}")

# RESUMEN

print("\n📊 RESUMEN ESTADÍSTICO")

resumen = anillos_gdf.groupby("anillo").agg(
    distancia=("distancia", "first"),
    delitos_totales=("cantidad_delitos", "sum"),
    densidad_promedio=("densidad", "mean")
).reset_index()

base_global = resumen.loc[resumen["anillo"] == 1, "densidad_promedio"].values[0]
resumen["densidad_relativa_global"] = resumen["densidad_promedio"] / base_global

print(resumen.to_string(index=False, float_format="%.2f"))

# ==================== 2eda_clasificar_cajeros.py ====================
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

# ==================== 2eda_comisarias.py ====================
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parent  # tesis/
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELITOS_PATH = DATA_PROCESSED / "delitos_total.csv.gz"
COMISARIAS_PATH = DATA_RAW / "comisarias-policia-de-la-ciudad.xlsx"

OUTPUT_HTML = OUTPUT_DIR / "mapa_comisarias_anillos_densidad_relativa_p20_p80.html"
OUTPUT_CSV = OUTPUT_DIR / "anillos_comisarias_densidad_relativa.csv"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"DELITOS: {DELITOS_PATH}")
print(f"COMISARIAS: {COMISARIAS_PATH}")

# CARGA DE DATOS

if not DELITOS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {DELITOS_PATH}")

if not COMISARIAS_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {COMISARIAS_PATH}")

delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
comisarias = pd.read_excel(COMISARIAS_PATH)

# LIMPIEZA

delitos["latitud"] = pd.to_numeric(delitos["latitud"], errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()

comisarias["lat"] = pd.to_numeric(comisarias["lat"], errors="coerce")
comisarias["long"] = pd.to_numeric(comisarias["long"], errors="coerce")
comisarias = comisarias.dropna(subset=["lat", "long"]).copy()

# GEO DATAFRAMES

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

# PROYECCIÓN MÉTRICA

delitos_m = delitos_gdf.to_crs(epsg=3857)
comisarias_m = comisarias_gdf.to_crs(epsg=3857)

# CREAR ANILLOS (300 m)

distancias = [0, 300, 600, 900, 1200]
anillos = []

for _, row in comisarias_m.iterrows():
    punto = row.geometry

    for i in range(4):
        externo = punto.buffer(distancias[i+1])
        interno = punto.buffer(distancias[i])
        anillo = externo.difference(interno)

        anillos.append({
            "id": row["id"],
            "nombre": row["nombre"],
            "direccion": row.get("direccion", ""),
            "barrio": row.get("barrio"),
            "comuna": row.get("comuna"),
            "anillo": i + 1,
            "distancia": f"{distancias[i]}-{distancias[i+1]} m",
            "geometry": anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# SPATIAL JOIN

join = gpd.sjoin(delitos_m, anillos_gdf, how="inner", predicate="within")

conteos = (
    join.groupby(["id", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(conteos, on=["id", "anillo"], how="left")
anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)

# DENSIDAD

anillos_gdf["area_km2"] = anillos_gdf.geometry.area / 1e6
anillos_gdf["densidad"] = anillos_gdf["cantidad_delitos"] / anillos_gdf["area_km2"]

# Relativa respecto al anillo 1
base = anillos_gdf[anillos_gdf["anillo"] == 1][["id", "densidad"]]
base = base.rename(columns={"densidad": "base"})

anillos_gdf = anillos_gdf.merge(base, on="id", how="left")

anillos_gdf["dens_rel"] = anillos_gdf["densidad"] / anillos_gdf["base"]

# EXPORT CSV

anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

# MAPA

anillos_wgs84 = anillos_gdf.to_crs(epsg=4326)

mapa = folium.Map(location=[-34.61, -58.43], zoom_start=12)

val = anillos_wgs84["dens_rel"].dropna()
p20, p80 = val.quantile(0.2), val.quantile(0.8)

colormap = cm.LinearColormap(
    colors=["green", "yellow", "red"],
    vmin=p20,
    vmax=p80
)
colormap.add_to(mapa)

for _, row in anillos_wgs84.iterrows():
    val = row["dens_rel"]

    color = "#ccc" if pd.isna(val) else colormap(val)

    folium.GeoJson(
        row["geometry"],
        style_function=lambda x, color=color: {
            "fillColor": color,
            "color": color,
            "weight": 1,
            "fillOpacity": 0.6
        }
    ).add_to(mapa)

mapa.save(OUTPUT_HTML)

print(f"🗺️ Mapa generado en: {OUTPUT_HTML}")

# ==================== 2eda_delito_cajero.py ====================
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

# ==================== 2eda_delito_gastro.py ====================
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

# ==================== alojamiento_delito.py ====================
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

# ==================== eda_arma.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv"
OUTPUT_IMG = OUTPUT_DIR / "eda_uso_arma.png"

print(f"📂 Cargando archivo desde: {INPUT_FILE}")

# CARGA

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"

df = pd.read_csv(INPUT_FILE)

# LIMPIEZA

df['uso_arma'] = df['uso_arma'].astype(str).str.strip().str.upper()

# Nos aseguramos que solo tome SI/NO válidos
df = df[df['uso_arma'].isin(['SI', 'NO'])]

# CONTEO

conteo = df['uso_arma'].value_counts().reindex(['SI', 'NO']).fillna(0)

porcentajes = conteo / conteo.sum() * 100

# GRÁFICO

plt.figure(figsize=(6,5))
ax = sns.barplot(x=conteo.index, y=conteo.values)

for i, (valor, pct) in enumerate(zip(conteo.values, porcentajes)):
    ax.text(i, valor + valor*0.01, f"{pct:.1f}%", ha='center', va='bottom', fontsize=11)

plt.title('Uso de arma en delitos (SI / NO)', fontsize=14)
plt.xlabel('Uso de arma')
plt.ylabel('Cantidad de delitos')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# Guardar imagen
plt.savefig(OUTPUT_IMG, dpi=300)

print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== eda_barrio.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parent  # porque este script está en tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_IMG = OUTPUT_DIR / "hist_delitos_barrio.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"BASE_DIR:   {BASE_DIR}")
print(f"INPUT_FILE: {INPUT_FILE}")
print(f"EXISTS:     {INPUT_FILE.exists()}")

# CARGA

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# LIMPIEZA

df["barrio"] = df["barrio"].astype(str).str.strip()

# CONTEO

conteo = df["barrio"].value_counts().sort_values(ascending=False)

# COLORES

norm = plt.Normalize(conteo.min(), conteo.max())
colors = plt.cm.coolwarm(norm(conteo.values))

# GRÁFICO

plt.figure(figsize=(16, 8))
sns.barplot(x=conteo.index, y=conteo.values, palette=colors)

plt.title("Cantidad de delitos por barrio", fontsize=16)
plt.xlabel("Barrio")
plt.ylabel("Cantidad de delitos")
plt.xticks(rotation=75, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(OUTPUT_IMG, dpi=300)
print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== eda_bar_anio.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv"
OUTPUT_IMG = OUTPUT_DIR / "bar_delitos_por_anio.png"

print(f"📂 Cargando archivo desde: {INPUT_FILE}")

# CARGA

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"

df = pd.read_csv(INPUT_FILE)

# AGRUPACIÓN

conteo = df['anio'].value_counts().sort_index()
porcentajes = conteo / conteo.sum() * 100

# GRÁFICO

plt.figure(figsize=(8,5))
ax = sns.barplot(x=conteo.index.astype(str), y=conteo.values)

# Etiquetas %
for i, (valor, pct) in enumerate(zip(conteo.values, porcentajes)):
    ax.text(i, valor + valor*0.01, f"{pct:.1f}%", ha='center', va='bottom', fontsize=11)

plt.title('Cantidad de delitos por año', fontsize=14)
plt.xlabel('Año')
plt.ylabel('Cantidad de delitos')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# Guardar imagen
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== eda_delito_horario.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

# Como este script está en tesis/src/, subimos un nivel hasta tesis/
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_IMG = OUTPUT_DIR / "histograma_franja.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"BASE_DIR:   {BASE_DIR}")
print(f"INPUT_FILE: {INPUT_FILE}")
print(f"EXISTS:     {INPUT_FILE.exists()}")

# CARGA

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# LIMPIEZA

df["franja"] = pd.to_numeric(df["franja"], errors="coerce")
df = df.dropna(subset=["franja"]).copy()
df["franja"] = df["franja"].astype(int)

# Mantener solo horas válidas 0–23
df = df[df["franja"].between(0, 23)].copy()

# CONTEO

conteo = df["franja"].value_counts().sort_index()

# COLORES

norm = plt.Normalize(conteo.min(), conteo.max())
colors = plt.cm.coolwarm(norm(conteo.values))

# GRÁFICO

plt.figure(figsize=(12, 6))
sns.barplot(x=conteo.index.astype(str), y=conteo.values, palette=colors)

plt.title("Cantidad de delitos por franja horaria", fontsize=14)
plt.xlabel("Hora del día (0–23)")
plt.ylabel("Cantidad de delitos")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(OUTPUT_IMG, dpi=300)
print(f"\n📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== eda_dia_mes.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv"
OUTPUT_IMG = OUTPUT_DIR / "eda_delitos_por_dia_mes.png"

print(f"📂 Cargando archivo desde: {INPUT_FILE}")

# CARGA

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# LIMPIEZA FECHA

df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df = df.dropna(subset=["fecha"]).copy()

# Extraer día del mes
df["dia_mes"] = df["fecha"].dt.day

# CONTEO

conteo = df["dia_mes"].value_counts().sort_index()

# COLORES (azul → rojo)

norm = plt.Normalize(conteo.min(), conteo.max())
colors = plt.cm.coolwarm(norm(conteo.values))

# GRÁFICO

plt.figure(figsize=(14, 6))
sns.barplot(x=conteo.index, y=conteo.values, palette=colors)

plt.title("Cantidad de delitos por día del mes", fontsize=16)
plt.xlabel("Día del mes")
plt.ylabel("Cantidad de delitos")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# Guardar imagen
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== eda_dia_semana.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

# Como este script está en tesis/src/, subimos un nivel hasta tesis/
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_IMG = OUTPUT_DIR / "eda_delitos_por_dia.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"BASE_DIR:   {BASE_DIR}")
print(f"INPUT_FILE: {INPUT_FILE}")
print(f"EXISTS:     {INPUT_FILE.exists()}")

# CARGA

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# LIMPIEZA

df["dia"] = (
    df["dia"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str[:3]
)

orden_dias = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]

# CONTEO

conteo_dia = df["dia"].value_counts()
conteo_dia = conteo_dia.reindex(orden_dias).dropna()

# COLORES

norm = plt.Normalize(conteo_dia.min(), conteo_dia.max())
colors = plt.cm.coolwarm(norm(conteo_dia.values))

# GRÁFICO

plt.figure(figsize=(12, 6))
sns.barplot(x=conteo_dia.index, y=conteo_dia.values, palette=colors)

plt.title("Cantidad de delitos por día de la semana", fontsize=16)
plt.xlabel("Día")
plt.ylabel("Cantidad de delitos")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)

print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== eda_espacio-temporal.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_IMG = OUTPUT_DIR / "heatmap_dia_hora.png"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"INPUT_FILE: {INPUT_FILE}")
print(f"EXISTS:     {INPUT_FILE.exists()}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró: {INPUT_FILE}")

# CARGA Y LIMPIEZA DE DATOS

print("Cargando dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

df["franja"] = pd.to_numeric(df["franja"], errors="coerce")
df = df.dropna(subset=["franja", "dia"]).copy()
df["franja"] = df["franja"].astype(int)

# Mantener solo horas válidas
df = df[df["franja"].between(0, 23)].copy()

df["dia"] = (
    df["dia"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str[:3]
)

orden_dias = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]
df = df[df["dia"].isin(orden_dias)].copy()

# MATRIZ BIVARIADA DÍA x HORA

print("Generando matriz de calor...")

matriz_calor = pd.crosstab(df["dia"], df["franja"])
matriz_calor = matriz_calor.reindex(orden_dias)
matriz_calor = matriz_calor.reindex(columns=range(24), fill_value=0)

# HEATMAP

plt.figure(figsize=(14, 6))

sns.heatmap(
    matriz_calor,
    cmap="YlOrRd",
    linewidths=0.5,
    annot=False,
    cbar_kws={"label": "Cantidad de delitos"}
)

plt.title(
    "Hotspots temporales: concentración de delitos por día y hora",
    fontsize=16,
    pad=15
)
plt.xlabel("Franja horaria (00:00–23:00 hs)")
plt.ylabel("Día de la semana")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)

print(f"📊 Heatmap guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== eda_heatmap_delitos.py ====================
# ============================================================
# EDA - Mapa de calor de delitos
# Lee: tesis/data/processed/delitos_total.csv.gz
# Guarda: tesis/outputs/mapa_delitos_heatmap.html
# ============================================================

from pathlib import Path
import pandas as pd
import folium
from folium.plugins import HeatMap
import webbrowser

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"
OUTPUT_HTML = OUTPUT_DIR / "mapa_delitos_heatmap.html"

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"BASE_DIR:   {BASE_DIR}")
print(f"INPUT_FILE: {INPUT_FILE}")
print(f"EXISTS:     {INPUT_FILE.exists()}")

# CARGA

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Cantidad de registros: {len(df)}")
print(df.head())

if len(df) == 0:
    raise ValueError("El DataFrame está vacío.")

# LIMPIEZA / MUESTREO

sample = df.sample(min(60000, len(df)), random_state=42).copy()

sample["latitud"] = pd.to_numeric(sample["latitud"], errors="coerce")
sample["longitud"] = pd.to_numeric(sample["longitud"], errors="coerce")

sample = sample.dropna(subset=["latitud", "longitud"])
sample = sample[(sample["latitud"] != 0) & (sample["longitud"] != 0)]

sample = sample[
    (sample["latitud"].between(-34.7, -34.5)) &
    (sample["longitud"].between(-58.6, -58.3))
]

heat_data = sample[["latitud", "longitud"]].values.tolist()

print(f"Cantidad de puntos en el mapa de calor: {len(heat_data)}")

if len(heat_data) == 0:
    raise ValueError("No hay puntos válidos para el mapa de calor.")

# MAPA

mapa = folium.Map(
    location=[-34.6083, -58.3712],
    zoom_start=12,
    tiles="cartodbpositron"
)

gradient = {
    0.1: "#0000FF",
    0.5: "#00FF00",
    1.0: "#FF0000"
}

HeatMap(
    heat_data,
    radius=8,
    blur=15,
    max_zoom=12,
    gradient=gradient
).add_to(mapa)

# GUARDAR / ABRIR

mapa.save(OUTPUT_HTML)
print(f"🗺️ Mapa guardado en: {OUTPUT_HTML}")

webbrowser.open(str(OUTPUT_HTML))

# ==================== eda_mes.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv"
OUTPUT_IMG = OUTPUT_DIR / "eda_delitos_por_mes.png"

print(f"📂 Cargando archivo desde: {INPUT_FILE}")

# CARGA

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# LIMPIEZA

df['mes'] = (
    df['mes']
    .astype(str)
    .str.strip()
    .str.lower()
    .str[:3]
)

orden_meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun',
               'jul', 'ago', 'sep', 'oct', 'nov', 'dic']

# CONTEO

conteo_mes = df['mes'].value_counts()
conteo_mes = conteo_mes.reindex(orden_meses).dropna()

# COLORES (azul → rojo)

norm = plt.Normalize(conteo_mes.min(), conteo_mes.max())
colors = plt.cm.coolwarm(norm(conteo_mes.values))

# GRÁFICO

plt.figure(figsize=(14, 6))
sns.barplot(x=conteo_mes.index, y=conteo_mes.values, palette=colors)

plt.title('Cantidad de delitos por mes', fontsize=16)
plt.xlabel('Mes')
plt.ylabel('Cantidad de delitos')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

# Guardar imagen
plt.savefig(OUTPUT_IMG, dpi=300)

print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== eda_moto.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv"
OUTPUT_IMG = OUTPUT_DIR / "eda_uso_moto.png"

print(f"📂 Cargando archivo desde: {INPUT_FILE}")

# CARGA

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"

df = pd.read_csv(INPUT_FILE)

# LIMPIEZA

df['uso_moto'] = df['uso_moto'].astype(str).str.strip().str.upper()

# Filtrar solo valores válidos
df = df[df['uso_moto'].isin(['SI', 'NO'])]

# CONTEO

conteo = df['uso_moto'].value_counts().reindex(['SI', 'NO']).fillna(0)
porcentajes = conteo / conteo.sum() * 100

# GRÁFICO

plt.figure(figsize=(6,5))
ax = sns.barplot(x=conteo.index, y=conteo.values)

for i, (valor, pct) in enumerate(zip(conteo.values, porcentajes)):
    ax.text(i, valor + valor*0.01, f"{pct:.1f}%", ha='center', va='bottom', fontsize=11)

plt.title('Uso de moto en delitos (SI / NO)', fontsize=14)
plt.xlabel('Uso de moto')
plt.ylabel('Cantidad de delitos')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# Guardar imagen
plt.savefig(OUTPUT_IMG, dpi=300)

print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== eda_tipo_delito.py ====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv"
OUTPUT_IMG = OUTPUT_DIR / "eda_delitos_por_tipo.png"

print(f"📂 Cargando archivo desde: {INPUT_FILE}")

# CARGA

INPUT_FILE = DATA_PROCESSED / "delitos_total.csv.gz"

df = pd.read_csv(INPUT_FILE)

# LIMPIEZA

df["tipo"] = df["tipo"].astype(str).str.strip()

# Opcional: quedarte con tipos válidos (evita NaN/raros)
df = df[df["tipo"].notna() & (df["tipo"] != "")]

# CONTEO

conteo_tipo = df["tipo"].value_counts().sort_values(ascending=False)

# COLORES (azul → rojo)

norm = plt.Normalize(conteo_tipo.min(), conteo_tipo.max())
colors = plt.cm.coolwarm(norm(conteo_tipo.values))

# GRÁFICO

plt.figure(figsize=(12, 7))
ax = sns.barplot(x=conteo_tipo.index, y=conteo_tipo.values, palette=colors)

plt.title("Cantidad de delitos por tipo", fontsize=16)
plt.xlabel("Tipo de delito")
plt.ylabel("Cantidad de delitos")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Etiquetas numéricas
for i, v in enumerate(conteo_tipo.values):
    ax.text(i, v + (v * 0.01), str(v), ha="center", va="bottom", fontsize=10)

plt.tight_layout()

# Guardar imagen
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()

# ==================== etl_alojamientos.py ====================
from pathlib import Path
import pandas as pd

# RUTAS REPRODUCIBLES

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

ALOJ_FILE = DATA_RAW / "alojamientos_turisticos.csv"
AIRBNB_FILE = DATA_RAW / "listings.csv"

OUTPUT_FILE = DATA_PROCESSED / "alojamientos_unificados.csv"

# LECTURA

print(f"📂 Leyendo archivo: {ALOJ_FILE}")
alojamientos = pd.read_csv(
    ALOJ_FILE,
    sep=",",
    encoding="utf-8-sig",
    quotechar='"'
)

print(f"📂 Leyendo archivo: {AIRBNB_FILE}")
airbnb = pd.read_csv(
    AIRBNB_FILE,
    sep=",",
    encoding="utf-8",
    quotechar='"',
    low_memory=False
)

print("Columnas alojamientos:", alojamientos.columns.tolist())
print("Columnas Airbnb:", airbnb.columns.tolist())

# NORMALIZAR ALOJAMIENTOS GCBA

alojamientos = alojamientos.rename(columns={
    "lat": "lat",
    "long": "long",
    "Lat": "lat",
    "Long": "long",
    "latitude": "lat",
    "longitude": "long"
})

alojamientos_limpio = alojamientos[["id", "lat", "long"]].copy()
alojamientos_limpio["fuente"] = "alojamientos_turisticos"

# NORMALIZAR AIRBNB

airbnb = airbnb.rename(columns={
    "latitude": "lat",
    "longitude": "long",
    "Lat": "lat",
    "Long": "long"
})

airbnb_limpio = airbnb[["id", "lat", "long"]].copy()
airbnb_limpio["fuente"] = "airbnb_listings"

# UNIFICAR

df_final = pd.concat(
    [alojamientos_limpio, airbnb_limpio],
    ignore_index=True
)

# LIMPIEZA DE COORDENADAS

df_final["lat"] = pd.to_numeric(df_final["lat"], errors="coerce")
df_final["long"] = pd.to_numeric(df_final["long"], errors="coerce")

df_final = df_final.dropna(subset=["lat", "long"])
df_final = df_final[(df_final["lat"] != 0) & (df_final["long"] != 0)]

# Filtro aproximado CABA
df_final = df_final[
    df_final["lat"].between(-34.75, -34.50) &
    df_final["long"].between(-58.60, -58.30)
].copy()

# Eliminar duplicados exactos por coordenadas
df_final = df_final.drop_duplicates(subset=["lat", "long"]).reset_index(drop=True)

# Orden final
df_final = df_final[["id", "lat", "long", "fuente"]]

# EXPORTAR

df_final.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("✅ Proceso finalizado correctamente.")
print(f"📊 Total de registros: {len(df_final):,}")
print(f"💾 Archivo guardado en: {OUTPUT_FILE}")

# ==================== etl_delitos.py ====================
# Pipeline reproducible de limpieza y consolidación de delitos
# Lee archivos desde: tesis/data/raw/
# Guarda salida en: tesis/data/processed/delitos_total.csv

from pathlib import Path
import pandas as pd
import numpy as np


# 1) RUTAS REPRODUCIBLES

# Este script debe estar ubicado en: tesis/src/
# BASE_DIR apunta a la carpeta raíz del proyecto: tesis/
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_PROCESSED / "delitos_total.csv"

print("===================================================")
print("CONFIGURACIÓN DE RUTAS")
print("===================================================")
print(f"Carpeta base del proyecto: {BASE_DIR}")
print(f"Carpeta de datos crudos:   {DATA_RAW}")
print(f"Carpeta de salida:         {DATA_PROCESSED}")
print(f"Archivo de salida:         {OUTPUT_PATH}")


# 2) FUNCIÓN: CORREGIR DECIMALES MAL FORMATEADOS

def corregir_decimal(valor):
    """
    Corrige coordenadas que vienen sin punto decimal o con el punto mal ubicado.
    Ejemplos:
        -3456789  -> -34.56789
        -5865432  -> -58.65432

    Devuelve float o NaN si no se puede convertir.
    """

    if pd.isna(valor):
        return np.nan

    s = str(valor).strip()
    s = s.replace(",", ".")

    negativo = s.startswith("-")
    if negativo:
        s = s[1:]

    s_clean = "".join(ch for ch in s if ch.isdigit() or ch == ".")

    if "." in s_clean:
        partes = s_clean.split(".")
        parte_entera = partes[0]
        parte_decimal = "".join(partes[1:])

        if len(parte_entera) > 2:
            s_clean = parte_entera[:2] + "." + parte_entera[2:] + parte_decimal
    else:
        if len(s_clean) > 2:
            s_clean = s_clean[:2] + "." + s_clean[2:]

    if negativo:
        s_clean = "-" + s_clean

    try:
        return float(s_clean)
    except Exception:
        return np.nan


# 3) FUNCIÓN: VALIDAR / CORREGIR RANGO CABA

def corregir_coordenadas(lat, lon):
    """
    Valida que las coordenadas estén dentro del rango aproximado de CABA.
    Si detecta valores de magnitud excesiva, intenta corregir dividiendo por 1e6.
    """

    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return np.nan, np.nan

    lat_min, lat_max = -34.7, -34.5
    lon_min, lon_max = -58.6, -58.3

    if not (lat_min <= lat <= lat_max):
        lat = lat / 1e6 if abs(lat) > 90 else lat

    if not (lon_min <= lon <= lon_max):
        lon = lon / 1e6 if abs(lon) > 180 else lon

    if not (lat_min <= lat <= lat_max) or not (lon_min <= lon <= lon_max):
        return np.nan, np.nan

    return lat, lon


# 4) ARCHIVOS DE ENTRADA

# Busca automáticamente todos los archivos con formato delitos_(año).xlsx
# Ejemplo: delitos_2016.xlsx, delitos_2017.xlsx, delitos_2024.xlsx

archivos_delitos = sorted(DATA_RAW.glob("delitos_*.xlsx"))

if not archivos_delitos:
    raise FileNotFoundError(
        f"No se encontraron archivos con patrón 'delitos_*.xlsx' en: {DATA_RAW}"
    )

print("\n===================================================")
print("ARCHIVOS DE DELITOS DETECTADOS")
print("===================================================")

for archivo in archivos_delitos:
    print(f" - {archivo.name}")

# 5) CONTADORES GLOBALES

datasets_limpios = []

total_global_inicial = 0
total_global_final = 0
total_global_nan_numeric = 0
total_global_decimal_fix = 0
total_global_invalid_rango = 0


# 6) LOOP DE CARGA + LIMPIEZA

for archivo_path in archivos_delitos:

    print("\n===================================================")
    print(f"CARGANDO ARCHIVO: {archivo_path.name}")
    print("===================================================")

    df = pd.read_excel(archivo_path)

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower()

    # Validación mínima de columnas requeridas
    if "latitud" not in df.columns or "longitud" not in df.columns:
        raise ValueError(
            f"El archivo {archivo_path.name} no contiene columnas 'latitud' y 'longitud'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    n_inicial = len(df)
    total_global_inicial += n_inicial

    print(f"Registros iniciales: {n_inicial}")

    # --------------------------------------------------------
    # Paso A: conversión a numérico
    # --------------------------------------------------------

    lat_na_antes = df["latitud"].isna().sum()
    lon_na_antes = df["longitud"].isna().sum()

    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    lat_na_despues = df["latitud"].isna().sum()
    lon_na_despues = df["longitud"].isna().sum()

    nuevos_nan_lat = lat_na_despues - lat_na_antes
    nuevos_nan_lon = lon_na_despues - lon_na_antes
    total_nan_numeric = nuevos_nan_lat + nuevos_nan_lon

    total_global_nan_numeric += total_nan_numeric

    print("[Paso A] Nuevos NaN por conversión numérica:")
    print(f"         latitud:  {nuevos_nan_lat}")
    print(f"         longitud: {nuevos_nan_lon}")

    # --------------------------------------------------------
    # Paso B: corrección automática de decimales
    # --------------------------------------------------------

    lat_antes_fix = df["latitud"].copy()
    lon_antes_fix = df["longitud"].copy()

    df["latitud"] = df["latitud"].apply(corregir_decimal)
    df["longitud"] = df["longitud"].apply(corregir_decimal)

    cambios_lat = (lat_antes_fix != df["latitud"]) & ~(
        lat_antes_fix.isna() & df["latitud"].isna()
    )
    cambios_lon = (lon_antes_fix != df["longitud"]) & ~(
        lon_antes_fix.isna() & df["longitud"].isna()
    )

    n_fix_lat = int(cambios_lat.sum())
    n_fix_lon = int(cambios_lon.sum())
    total_fix_decimal = n_fix_lat + n_fix_lon

    total_global_decimal_fix += total_fix_decimal

    print("[Paso B] Corrección automática de decimales:")
    print(f"         latitud corregidas:  {n_fix_lat}")
    print(f"         longitud corregidas: {n_fix_lon}")

    # --------------------------------------------------------
    # Paso C: validación/corrección por rango CABA
    # --------------------------------------------------------

    df[["lat_corr", "lon_corr"]] = df.apply(
        lambda row: corregir_coordenadas(row["latitud"], row["longitud"]),
        axis=1,
        result_type="expand",
    )

    invalidos_rango = int(df["lat_corr"].isna().sum())
    total_global_invalid_rango += invalidos_rango

    print(f"[Paso C] Coordenadas fuera de rango: {invalidos_rango}")

    df["latitud"] = df["lat_corr"]
    df["longitud"] = df["lon_corr"]
    df.drop(columns=["lat_corr", "lon_corr"], inplace=True)

    # --------------------------------------------------------
    # Paso D: eliminar registros sin coordenadas válidas
    # --------------------------------------------------------

    n_antes_drop = len(df)

    df = df.dropna(subset=["latitud", "longitud"]).copy()

    n_final = len(df)
    filtrados_total = n_inicial - n_final

    total_global_final += n_final

    print("[Paso D] Drop NaN finales:")
    print(f"         registros antes drop: {n_antes_drop}")
    print(f"         registros finales:    {n_final}")
    print(f"         filtrados totales:    {filtrados_total}")

    datasets_limpios.append(df)


# 7) UNIFICACIÓN FINAL

if not datasets_limpios:
    raise ValueError("No se cargó ningún dataset válido.")

delitos_total = pd.concat(datasets_limpios, ignore_index=True)


# 8) RESUMEN GLOBAL

print("\n\n===================================================")
print("RESUMEN GLOBAL")
print("===================================================")
print(f"Total registros iniciales: {total_global_inicial}")
print(f"Total nuevos NaN por conversión numérica: {total_global_nan_numeric}")
print(f"Total correcciones de decimales aplicadas: {total_global_decimal_fix}")
print(f"Total coordenadas fuera de rango: {total_global_invalid_rango}")
print(f"Total registros finales limpios: {len(delitos_total)}")

print("\nColumnas del dataset final:")
print(list(delitos_total.columns))


# 9) GUARDAR CSV FINAL

OUTPUT_GZ = OUTPUT_PATH.with_suffix(".csv.gz")

delitos_total.to_csv(OUTPUT_GZ, index=False, compression='gzip')

print(f"\n📁 Archivo comprimido generado en: {OUTPUT_GZ}")

print("\n===================================================")
print("PROCESO FINALIZADO")
print("===================================================")
print(f"Archivo CSV generado en: {OUTPUT_PATH}")

# ==================== VER_2alojamientos_geocode.py ====================
# =========================================
# GEOCODIFICACIÓN ROBUSTA DE ALOJAMIENTOS
# =========================================

from pathlib import Path
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# =========================================
# RUTAS REPRODUCIBLES
# =========================================

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

INPUT_PATH = DATA_RAW / "alojamientos-turisticos.csv"
CABA_POLYGON_PATH = DATA_RAW / "comunas.geojson"

OUTPUT_PATH = DATA_PROCESSED / "alojamientos-geocodificados.csv"
TEMP_PATH = DATA_PROCESSED / "alojamientos-geocodificados_temp.csv"

SAVE_EVERY = 25

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"INPUT:   {INPUT_PATH}")
print(f"COMUNAS: {CABA_POLYGON_PATH}")
print(f"OUTPUT:  {OUTPUT_PATH}")
print(f"TEMP:    {TEMP_PATH}")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {INPUT_PATH}")

if not CABA_POLYGON_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {CABA_POLYGON_PATH}")

# =========================================
# CARGA DE DATOS
# =========================================

if TEMP_PATH.exists():
    print("♻️ Archivo temporal encontrado. Reanudando desde temporal...")
    df = pd.read_csv(TEMP_PATH, low_memory=False)
else:
    print("📂 Cargando alojamientos originales...")
    df = pd.read_csv(INPUT_PATH, encoding="latin1", delimiter=";", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    if "latitud" not in df.columns:
        df["latitud"] = None
    if "longitud" not in df.columns:
        df["longitud"] = None
    if "estado_geocodificacion" not in df.columns:
        df["estado_geocodificacion"] = None

# =========================================
# CARGAR POLÍGONO CABA
# =========================================

print("🗺️ Cargando polígono de CABA...")
caba = gpd.read_file(CABA_POLYGON_PATH).to_crs(epsg=4326)
caba_union = caba.unary_union

# =========================================
# GEOCODIFICADOR ROBUSTO
# =========================================

geolocator = Nominatim(
    user_agent="tesis_caba_geocoder",
    timeout=10
)

geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1.5,
    max_retries=3,
    error_wait_seconds=5,
    swallow_exceptions=True
)

def geocodificar(direccion):
    if pd.isna(direccion) or str(direccion).strip() == "":
        return None, None, "sin_direccion"

    direccion_limpia = str(direccion).strip()
    consulta = f"{direccion_limpia}, Ciudad Autónoma de Buenos Aires, Argentina"

    try:
        location = geocode(consulta)

        if location is None:
            return None, None, "no_geocodificado"

        lat, lon = location.latitude, location.longitude
        punto = Point(lon, lat)

        if not caba_union.contains(punto):
            return lat, lon, "fuera_caba"

        return lat, lon, "ok"

    except Exception as e:
        return None, None, f"error: {e}"

# =========================================
# PROCESO PRINCIPAL
# =========================================

procesados = 0
total = len(df)

print(f"📊 Total de registros: {total}")

for idx, row in df.iterrows():

    ya_tiene_coord = pd.notna(row.get("latitud")) and pd.notna(row.get("longitud"))
    estado_previo = row.get("estado_geocodificacion")

    if ya_tiene_coord or estado_previo in ["fuera_caba", "no_geocodificado", "sin_direccion"]:
        continue

    direccion = row.get("direccion", None)

    lat, lon, estado = geocodificar(direccion)

    df.at[idx, "latitud"] = lat
    df.at[idx, "longitud"] = lon
    df.at[idx, "estado_geocodificacion"] = estado

    procesados += 1

    if estado == "ok":
        print(f"✅ {idx}/{total} | {direccion} -> ({lat:.5f}, {lon:.5f})")
    else:
        print(f"⚠️ {idx}/{total} | {direccion} -> {estado}")

    if procesados % SAVE_EVERY == 0:
        df.to_csv(TEMP_PATH, index=False)
        print(f"💾 Guardado parcial: {procesados} nuevos procesados")

# =========================================
# GUARDADO FINAL
# =========================================

df.to_csv(TEMP_PATH, index=False)

df_final = df[
    (pd.notna(df["latitud"])) &
    (pd.notna(df["longitud"])) &
    (df["estado_geocodificacion"] == "ok")
].copy()

df_final.to_csv(OUTPUT_PATH, index=False)

print("===================================================")
print("PROCESO FINALIZADO")
print("===================================================")
print(f"Total registros originales: {len(df)}")
print(f"Geocodificados válidos: {len(df_final)}")
print(f"Archivo final: {OUTPUT_PATH}")
print(f"Archivo temporal conservado: {TEMP_PATH}")

# ==================== VER_alojamientos_geocode.py ====================
# =========================================
# 1 - CARGA, LIMPIEZA Y GEOCODIFICACIÓN
# =========================================

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# =========================================
# RUTAS REPRODUCIBLES
# =========================================

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

INPUT_PATH = DATA_RAW / "alojamientos-turisticos.csv"
OUTPUT_PATH = DATA_PROCESSED / "alojamientos-geocodificados.csv"
TEMP_PATH = DATA_PROCESSED / "alojamientos-geocodificados_temp.csv"
CABA_POLYGON_PATH = DATA_RAW / "comunas.geojson"

SAVE_EVERY = 50

print("===================================================")
print("DEBUG RUTAS")
print("===================================================")
print(f"INPUT:   {INPUT_PATH}")
print(f"OUTPUT:  {OUTPUT_PATH}")
print(f"TEMP:    {TEMP_PATH}")
print(f"COMUNAS: {CABA_POLYGON_PATH}")
print(f"EXISTS INPUT:   {INPUT_PATH.exists()}")
print(f"EXISTS COMUNAS: {CABA_POLYGON_PATH.exists()}")

# =========================================
# VALIDACIÓN DE ARCHIVOS
# =========================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {INPUT_PATH}")

if not CABA_POLYGON_PATH.exists():
    raise FileNotFoundError(f"No se encontró: {CABA_POLYGON_PATH}")

# =========================================
# GEOCODIFICADOR
# =========================================

geolocator = Nominatim(user_agent="tesis_caba")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# =========================================
# CARGA DE DATOS
# =========================================

print(f"\n📂 Cargando alojamientos desde: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, encoding="latin1", delimiter=";")

df.columns = df.columns.str.strip().str.lower()

# =========================================
# CARGAR POLÍGONO DE CABA
# =========================================

print("🗺️ Cargando polígono de CABA...")

caba = gpd.read_file(CABA_POLYGON_PATH)
caba = caba.to_crs(epsg=4326)

caba_union = caba.unary_union

# =========================================
# REANUDACIÓN
# =========================================

if TEMP_PATH.exists():
    print("♻️ Archivo temporal encontrado. Reanudando...")
    df = pd.read_csv(TEMP_PATH)
else:
    df["latitud"] = None
    df["longitud"] = None

# =========================================
# FUNCIÓN DE GEOCODIFICACIÓN
# =========================================

def geocodificar(direccion):
    try:
        location = geocode(f"{direccion}, Buenos Aires, Argentina")
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"⚠️ Error geocodificando {direccion}: {e}")

    return None, None

# =========================================
# PROCESO PRINCIPAL
# =========================================

procesados = 0

for idx, row in df.iterrows():

    if pd.notna(row.get("latitud")) and pd.notna(row.get("longitud")):
        continue

    direccion = row.get("direccion", None)

    if pd.isna(direccion):
        continue

    lat, lon = geocodificar(direccion)

    if lat is not None and lon is not None:
        punto = Point(lon, lat)

        if caba_union.contains(punto):
            df.at[idx, "latitud"] = lat
            df.at[idx, "longitud"] = lon
            print(f"✅ {direccion} -> ({lat:.5f}, {lon:.5f})")
        else:
            print(f"🚫 Fuera de CABA: {direccion}")
    else:
        print(f"❌ No geocodificado: {direccion}")

    procesados += 1

    if procesados % SAVE_EVERY == 0:
        df.to_csv(TEMP_PATH, index=False)
        print(f"💾 Guardado parcial ({procesados} registros)")

# =========================================
# LIMPIEZA FINAL
# =========================================

df = df.dropna(subset=["latitud", "longitud"]).copy()

print(f"\n📊 Total geocodificados válidos: {len(df)}")

# =========================================
# GUARDADO FINAL
# =========================================

df.to_csv(OUTPUT_PATH, index=False)

if TEMP_PATH.exists():
    TEMP_PATH.unlink()

print(f"✅ Archivo final guardado en: {OUTPUT_PATH}")

