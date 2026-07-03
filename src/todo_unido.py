from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────

BASE_DIR       = Path(__file__).resolve().parents[1]
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR     = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELITOS_PATH = DATA_PROCESSED / "delitos_total.csv.gz"
TREN_PATH    = DATA_RAW / "estaciones-de-ferrocarril.csv"

OUTPUT_HTML  = OUTPUT_DIR / "mapa_tren_anillos.html"
OUTPUT_CSV   = OUTPUT_DIR / "anillos_tren_densidad.csv"
OUTPUT_IMG   = OUTPUT_DIR / "mann_whitney_tren_anillos.png"

for p in [DELITOS_PATH, TREN_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"No se encontró: {p}")

# ─────────────────────────────────────────
# CARGA Y LIMPIEZA
# ─────────────────────────────────────────

print("📂 Cargando datos...")
delitos = pd.read_csv(DELITOS_PATH, low_memory=False)
tren    = pd.read_csv(TREN_PATH)

delitos.columns = delitos.columns.str.strip().str.lower()
tren.columns    = tren.columns.str.strip().str.lower()

delitos["latitud"]  = pd.to_numeric(delitos["latitud"],  errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()

tren["lat"]  = pd.to_numeric(tren["lat"],  errors="coerce")
tren["long"] = pd.to_numeric(tren["long"], errors="coerce")
tren = tren.dropna(subset=["lat", "long"]).copy()

# Solo estaciones dentro de CABA
tren_caba = tren[
    tren["comuna"].notna() & (tren["comuna"].astype(str).str.strip() != "")
].copy()

print(f"✅ Estaciones en CABA: {len(tren_caba)}")
print(f"✅ Delitos cargados:   {len(delitos)}")

# ─────────────────────────────────────────
# GEO DATAFRAMES  →  EPSG:3857
# ─────────────────────────────────────────

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

tren_gdf = gpd.GeoDataFrame(
    tren_caba,
    geometry=gpd.points_from_xy(tren_caba["long"], tren_caba["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

delitos_gdf["id_delito"] = range(len(delitos_gdf))

# ═══════════════════════════════════════════════════════════
# BLOQUE 1 — ANILLOS DE DENSIDAD (3 × 100 m)
# ═══════════════════════════════════════════════════════════

print("\n══ BLOQUE 1: Anillos de densidad ══")

distancias = [0, 100, 200, 300]
anillos = []

for _, row in tren_gdf.iterrows():
    punto = row.geometry
    for i in range(len(distancias) - 1):
        externo = punto.buffer(distancias[i + 1])
        interno = punto.buffer(distancias[i])
        anillo  = externo.difference(interno)

        anillos.append({
            "id":        row["id"],
            "nombre":    row.get("nombre", ""),
            "linea":     row.get("linea",  ""),
            "barrio":    row.get("barrio", ""),
            "comuna":    row.get("comuna", ""),
            "anillo":    i + 1,
            "distancia": f"{distancias[i]}-{distancias[i+1]} m",
            "geometry":  anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# Spatial join con deduplicación (cada delito al anillo más cercano)
join = gpd.sjoin(delitos_gdf, anillos_gdf, how="inner", predicate="within")
join = join.sort_values("anillo").drop_duplicates(subset="id_delito", keep="first")

conteos = (
    join.groupby(["id", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(conteos, on=["id", "anillo"], how="left")
anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)
anillos_gdf["area_km2"] = anillos_gdf.geometry.area / 1_000_000
anillos_gdf["densidad"] = anillos_gdf["cantidad_delitos"] / anillos_gdf["area_km2"]

# Densidad relativa respecto al anillo 1
base = (
    anillos_gdf[anillos_gdf["anillo"] == 1][["id", "densidad"]]
    .rename(columns={"densidad": "base"})
)
anillos_gdf = anillos_gdf.merge(base, on="id", how="left")
anillos_gdf["densidad_relativa"] = np.where(
    anillos_gdf["base"] > 0,
    anillos_gdf["densidad"] / anillos_gdf["base"],
    np.nan
)

# Export CSV
anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
print(f"💾 CSV guardado en: {OUTPUT_CSV}")

# Resumen estadístico
resumen = (
    anillos_gdf.groupby("anillo")
    .agg(
        distancia         =("distancia",        "first"),
        delitos_totales   =("cantidad_delitos",  "sum"),
        densidad_promedio =("densidad",          "mean"),
    )
    .reset_index()
)
base_global = resumen.loc[resumen["anillo"] == 1, "densidad_promedio"].values[0]
resumen["densidad_relativa_pct"] = resumen["densidad_promedio"] / base_global * 100

print("\n📊 RESUMEN POR ANILLO:")
print(resumen.to_string(index=False, float_format="%.2f"))

# Mapa interactivo
anillos_wgs = anillos_gdf.to_crs(epsg=4326)
mapa = folium.Map(location=[-34.61, -58.43], zoom_start=13)

vals = anillos_wgs["densidad_relativa"].dropna()
p20, p80 = vals.quantile(0.2), vals.quantile(0.8)
p20, p80 = min(p20, 1), max(p80, 1)

colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=p20, vmax=p80)
colormap.add_to(mapa)

def clip(v):
    if pd.isna(v): return None
    return max(min(v, p80), p20)

for _, row in anillos_wgs.iterrows():
    val   = clip(row["densidad_relativa"])
    color = "#cccccc" if val is None else colormap(val)
    tooltip = (
        f"<b>{row['nombre']}</b> ({row['linea']})<br>"
        f"Anillo: {row['distancia']}<br>"
        f"Delitos: {int(row['cantidad_delitos'])}<br>"
        f"Densidad: {row['densidad']:.1f} del/km²<br>"
        f"Relativa: {row['densidad_relativa']:.2f}"
    )
    folium.GeoJson(
        row["geometry"].__geo_interface__,
        style_function=lambda f, c=color: {
            "fillColor": c, "color": c,
            "weight": 1, "fillOpacity": 0.6
        },
        tooltip=tooltip
    ).add_to(mapa)

for _, row in tren_caba.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=5, color="black", fill=True,
        tooltip=f"{row['nombre']} — {row['linea']}"
    ).add_to(mapa)

mapa.save(OUTPUT_HTML)
print(f"🗺️ Mapa guardado en: {OUTPUT_HTML}")

# ═══════════════════════════════════════════════════════════
# BLOQUE 2 — MANN-WHITNEY U ENTRE ANILLOS
# ═══════════════════════════════════════════════════════════

print("\n══ BLOQUE 2: Mann-Whitney U entre anillos ══")

anillo1 = anillos_gdf[anillos_gdf["anillo"] == 1]["densidad"].values
anillo2 = anillos_gdf[anillos_gdf["anillo"] == 2]["densidad"].values
anillo3 = anillos_gdf[anillos_gdf["anillo"] == 3]["densidad"].values

pares = [
    ("Anillo 1 vs 2  (0-100m vs 100-200m)",   anillo1, anillo2),
    ("Anillo 1 vs 3  (0-100m vs 200-300m)",   anillo1, anillo3),
    ("Anillo 2 vs 3  (100-200m vs 200-300m)", anillo2, anillo3),
]

print(f"\n{'Comparación':<44} {'U-stat':>10} {'p-value':>12} {'Sig. (α=0.05)':>14}")
print("-" * 84)

resultados_mw = []
for label, a, b in pares:
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    sig = "✅ Sí" if p_val < 0.05 else "❌ No"
    print(f"{label:<44} {u_stat:>10.1f} {p_val:>12.4e} {sig:>14}")
    resultados_mw.append({
        "comparacion":  label,
        "u_stat":       u_stat,
        "p_value":      p_val,
        "significativo": p_val < 0.05
    })

# Gráfico: boxplot de densidad por anillo con p-values
fig, ax = plt.subplots(figsize=(9, 6))

orden_labels = ["0-100 m", "100-200 m", "200-300 m"]
data_plot = anillos_gdf[["anillo", "distancia", "densidad"]].copy()
data_plot["distancia"] = pd.Categorical(
    data_plot["distancia"], categories=orden_labels, ordered=True
)

sns.boxplot(
    data=data_plot,
    x="distancia",
    y="densidad",
    order=orden_labels,
    palette=["#d73027", "#fc8d59", "#91cf60"],
    ax=ax
)

# Anotar p-values sobre el gráfico
y_max = data_plot["densidad"].quantile(0.95)
comparaciones_plot = [
    (0, 1, resultados_mw[0]["p_value"], 0.05),
    (0, 2, resultados_mw[1]["p_value"], 0.18),
    (1, 2, resultados_mw[2]["p_value"], 0.32),
]

for x1, x2, p, offset in comparaciones_plot:
    y = y_max * (1 + offset)
    ax.plot([x1, x1, x2, x2], [y * 0.97, y, y, y * 0.97], lw=1, color="black")
    sig_label = f"p={p:.3f}" if p >= 0.001 else f"p={p:.2e}"
    ax.text((x1 + x2) / 2, y * 1.02, sig_label, ha="center", va="bottom", fontsize=9)

ax.set_title(
    "Densidad de delitos por anillo de distancia a estaciones de tren\n"
    "Mann-Whitney U entre anillos (α = 0.05)",
    fontsize=13
)
ax.set_xlabel("Distancia a la estación")
ax.set_ylabel("Densidad de delitos (del/km²)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
sns.despine()

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"\n📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()
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
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────

BASE_DIR       = Path(__file__).resolve().parents[1]
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR     = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELITOS_PATH    = DATA_PROCESSED / "delitos_total.csv.gz"
COMISARIAS_PATH = DATA_RAW / "comisarias-policia-de-la-ciudad.xlsx"

OUTPUT_HTML = OUTPUT_DIR / "mapa_comisarias_anillos.html"
OUTPUT_CSV  = OUTPUT_DIR / "anillos_comisarias_densidad.csv"
OUTPUT_IMG  = OUTPUT_DIR / "mann_whitney_comisarias_anillos.png"

for p in [DELITOS_PATH, COMISARIAS_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"No se encontró: {p}")

# ─────────────────────────────────────────
# CARGA Y LIMPIEZA
# ─────────────────────────────────────────

print("📂 Cargando datos...")
delitos    = pd.read_csv(DELITOS_PATH, low_memory=False)
comisarias = pd.read_excel(COMISARIAS_PATH)

delitos.columns    = delitos.columns.str.strip().str.lower()
comisarias.columns = comisarias.columns.str.strip().str.lower()

delitos["latitud"]  = pd.to_numeric(delitos["latitud"],  errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()

comisarias["lat"]  = pd.to_numeric(comisarias["lat"],  errors="coerce")
comisarias["long"] = pd.to_numeric(comisarias["long"], errors="coerce")
comisarias = comisarias.dropna(subset=["lat", "long"]).copy()

print(f"✅ Comisarías cargadas: {len(comisarias)}")
print(f"✅ Delitos cargados:    {len(delitos)}")

# ─────────────────────────────────────────
# GEO DATAFRAMES  →  EPSG:3857
# ─────────────────────────────────────────

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

comisarias_gdf = gpd.GeoDataFrame(
    comisarias,
    geometry=gpd.points_from_xy(comisarias["long"], comisarias["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

delitos_gdf["id_delito"] = range(len(delitos_gdf))

# ═══════════════════════════════════════════════════════════
# BLOQUE 1 — ANILLOS DE DENSIDAD (6 × 100 m)
# ═══════════════════════════════════════════════════════════

print("\n══ BLOQUE 1: Anillos de densidad ══")

distancias = [0, 100, 200, 300, 400, 500, 600]
anillos = []

for _, row in comisarias_gdf.iterrows():
    punto = row.geometry
    for i in range(len(distancias) - 1):
        externo = punto.buffer(distancias[i + 1])
        interno = punto.buffer(distancias[i])
        anillo  = externo.difference(interno)

        anillos.append({
            "id":        row["id"],
            "nombre":    row.get("nombre",    ""),
            "direccion": row.get("direccion", ""),
            "barrio":    row.get("barrio",    ""),
            "comuna":    row.get("comuna",    ""),
            "anillo":    i + 1,
            "distancia": f"{distancias[i]}-{distancias[i+1]} m",
            "geometry":  anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# Spatial join con deduplicación:
# cada delito se asigna una sola vez al anillo más cercano
# de la comisaría más próxima, evitando doble conteo
join = gpd.sjoin(delitos_gdf, anillos_gdf, how="inner", predicate="within")
join = join.sort_values("anillo").drop_duplicates(subset="id_delito", keep="first")

conteos = (
    join.groupby(["id", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(conteos, on=["id", "anillo"], how="left")
anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)
anillos_gdf["area_km2"] = anillos_gdf.geometry.area / 1_000_000
anillos_gdf["densidad"] = anillos_gdf["cantidad_delitos"] / anillos_gdf["area_km2"]

# Densidad relativa respecto al anillo 1 de cada comisaría
base = (
    anillos_gdf[anillos_gdf["anillo"] == 1][["id", "densidad"]]
    .rename(columns={"densidad": "base"})
)
anillos_gdf = anillos_gdf.merge(base, on="id", how="left")
anillos_gdf["densidad_relativa"] = np.where(
    anillos_gdf["base"] > 0,
    anillos_gdf["densidad"] / anillos_gdf["base"],
    np.nan
)

# Export CSV
anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
print(f"💾 CSV guardado en: {OUTPUT_CSV}")

# Resumen estadístico
resumen = (
    anillos_gdf.groupby("anillo")
    .agg(
        distancia         =("distancia",       "first"),
        delitos_totales   =("cantidad_delitos", "sum"),
        densidad_promedio =("densidad",         "mean"),
    )
    .reset_index()
)
base_global = resumen.loc[resumen["anillo"] == 1, "densidad_promedio"].values[0]
resumen["densidad_relativa_pct"] = resumen["densidad_promedio"] / base_global * 100

print("\n📊 RESUMEN POR ANILLO:")
print(resumen.to_string(index=False, float_format="%.2f"))

# Mapa interactivo
anillos_wgs = anillos_gdf.to_crs(epsg=4326)
mapa = folium.Map(location=[-34.61, -58.43], zoom_start=12)

vals = anillos_wgs["densidad_relativa"].dropna()
p20, p80 = vals.quantile(0.2), vals.quantile(0.8)
p20, p80 = min(p20, 1), max(p80, 1)

colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=p20, vmax=p80)
colormap.add_to(mapa)

def clip(v):
    if pd.isna(v): return None
    return max(min(v, p80), p20)

for _, row in anillos_wgs.iterrows():
    val   = clip(row["densidad_relativa"])
    color = "#cccccc" if val is None else colormap(val)
    tooltip = (
        f"<b>{row['nombre']}</b><br>"
        f"Barrio: {row['barrio']}<br>"
        f"Anillo: {row['distancia']}<br>"
        f"Delitos: {int(row['cantidad_delitos'])}<br>"
        f"Densidad: {row['densidad']:.1f} del/km²<br>"
        f"Relativa: {row['densidad_relativa']:.2f}"
    )
    folium.GeoJson(
        row["geometry"].__geo_interface__,
        style_function=lambda f, c=color: {
            "fillColor": c, "color": c,
            "weight": 1, "fillOpacity": 0.6
        },
        tooltip=tooltip
    ).add_to(mapa)

for _, row in comisarias.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=4, color="black", fill=True,
        tooltip=row.get("nombre", "Comisaría")
    ).add_to(mapa)

mapa.save(OUTPUT_HTML)
print(f"🗺️ Mapa guardado en: {OUTPUT_HTML}")

# ═══════════════════════════════════════════════════════════
# BLOQUE 2 — MANN-WHITNEY U ENTRE ANILLOS
# ═══════════════════════════════════════════════════════════

print("\n══ BLOQUE 2: Mann-Whitney U entre anillos ══")

grupos = {
    i: anillos_gdf[anillos_gdf["anillo"] == i]["densidad"].values
    for i in range(1, 7)
}

# Todos los pares posibles
pares = [
    (i, j)
    for i in range(1, 7)
    for j in range(i + 1, 7)
]

etiquetas = {
    1: "0-100m",
    2: "100-200m",
    3: "200-300m",
    4: "300-400m",
    5: "400-500m",
    6: "500-600m",
}

print(f"\n{'Comparación':<46} {'U-stat':>10} {'p-value':>12} {'Sig. (α=0.05)':>14}")
print("-" * 86)

resultados_mw = []
for i, j in pares:
    label = f"Anillo {i} vs {j}  ({etiquetas[i]} vs {etiquetas[j]})"
    u_stat, p_val = stats.mannwhitneyu(grupos[i], grupos[j], alternative="two-sided")
    sig = "✅ Sí" if p_val < 0.05 else "❌ No"
    print(f"{label:<46} {u_stat:>10.1f} {p_val:>12.4e} {sig:>14}")
    resultados_mw.append({
        "anillo_a":      i,
        "anillo_b":      j,
        "u_stat":        u_stat,
        "p_value":       p_val,
        "significativo": p_val < 0.05
    })

# Gráfico: boxplot con p-values de pares adyacentes
fig, ax = plt.subplots(figsize=(13, 6))

orden_labels = [f"{distancias[i]}-{distancias[i+1]} m" for i in range(len(distancias)-1)]
data_plot = anillos_gdf[["anillo", "distancia", "densidad"]].copy()
data_plot["distancia"] = pd.Categorical(
    data_plot["distancia"], categories=orden_labels, ordered=True
)

palette = ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#a6d96a", "#1a9850"]

sns.boxplot(
    data=data_plot,
    x="distancia",
    y="densidad",
    order=orden_labels,
    palette=palette,
    hue="distancia",
    legend=False,
    ax=ax
)

# Anotar solo pares adyacentes
adyacentes = [(i, i+1) for i in range(1, 6)]
y_max = data_plot["densidad"].quantile(0.95)

for idx, (i, j) in enumerate(adyacentes):
    r = next(r for r in resultados_mw if r["anillo_a"] == i and r["anillo_b"] == j)
    offset = 0.05 + idx * 0.13
    y = y_max * (1 + offset)
    x1, x2 = i - 1, j - 1
    ax.plot([x1, x1, x2, x2], [y * 0.97, y, y, y * 0.97], lw=1, color="black")
    p = r["p_value"]
    sig_label = f"p={p:.3f}" if p >= 0.001 else f"p={p:.2e}"
    ax.text((x1 + x2) / 2, y * 1.02, sig_label, ha="center", va="bottom", fontsize=8)

ax.set_title(
    "Densidad de delitos por anillo de distancia a comisarías\n"
    "Mann-Whitney U entre anillos adyacentes (α = 0.05)",
    fontsize=13
)
ax.set_xlabel("Distancia a la comisaría")
ax.set_ylabel("Densidad de delitos (del/km²)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
sns.despine()

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"\n📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────

BASE_DIR       = Path(__file__).resolve().parents[1]
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR     = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELITOS_PATH    = DATA_PROCESSED / "delitos_total.csv.gz"
COMISARIAS_PATH = DATA_RAW / "comisarias-policia-de-la-ciudad.xlsx"

OUTPUT_HTML = OUTPUT_DIR / "mapa_comisarias_anillos.html"
OUTPUT_CSV  = OUTPUT_DIR / "anillos_comisarias_densidad.csv"
OUTPUT_IMG  = OUTPUT_DIR / "mann_whitney_comisarias_anillos.png"

for p in [DELITOS_PATH, COMISARIAS_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"No se encontró: {p}")

# ─────────────────────────────────────────
# CARGA Y LIMPIEZA
# ─────────────────────────────────────────

print("📂 Cargando datos...")
delitos    = pd.read_csv(DELITOS_PATH, low_memory=False)
comisarias = pd.read_excel(COMISARIAS_PATH)

delitos.columns    = delitos.columns.str.strip().str.lower()
comisarias.columns = comisarias.columns.str.strip().str.lower()

delitos["latitud"]  = pd.to_numeric(delitos["latitud"],  errors="coerce")
delitos["longitud"] = pd.to_numeric(delitos["longitud"], errors="coerce")
delitos = delitos.dropna(subset=["latitud", "longitud"]).copy()

comisarias["lat"]  = pd.to_numeric(comisarias["lat"],  errors="coerce")
comisarias["long"] = pd.to_numeric(comisarias["long"], errors="coerce")
comisarias = comisarias.dropna(subset=["lat", "long"]).copy()

print(f"✅ Comisarías cargadas: {len(comisarias)}")
print(f"✅ Delitos cargados:    {len(delitos)}")

# ─────────────────────────────────────────
# GEO DATAFRAMES  →  EPSG:3857
# ─────────────────────────────────────────

delitos_gdf = gpd.GeoDataFrame(
    delitos,
    geometry=gpd.points_from_xy(delitos["longitud"], delitos["latitud"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

comisarias_gdf = gpd.GeoDataFrame(
    comisarias,
    geometry=gpd.points_from_xy(comisarias["long"], comisarias["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

delitos_gdf["id_delito"] = range(len(delitos_gdf))

# ═══════════════════════════════════════════════════════════
# BLOQUE 1 — ANILLOS DE DENSIDAD (4 × 300 m)
# ═══════════════════════════════════════════════════════════

print("\n══ BLOQUE 1: Anillos de densidad ══")

distancias = [0, 300, 600, 900, 1200]
anillos = []

for _, row in comisarias_gdf.iterrows():
    punto = row.geometry
    for i in range(len(distancias) - 1):
        externo = punto.buffer(distancias[i + 1])
        interno = punto.buffer(distancias[i])
        anillo  = externo.difference(interno)

        anillos.append({
            "id":        row["id"],
            "nombre":    row.get("nombre",    ""),
            "direccion": row.get("direccion", ""),
            "barrio":    row.get("barrio",    ""),
            "comuna":    row.get("comuna",    ""),
            "anillo":    i + 1,
            "distancia": f"{distancias[i]}-{distancias[i+1]} m",
            "geometry":  anillo
        })

anillos_gdf = gpd.GeoDataFrame(anillos, crs="EPSG:3857")

# Spatial join con deduplicación:
# cada delito se asigna al anillo más cercano (menor número de anillo)
# de la comisaría más próxima, evitando doble conteo
join = gpd.sjoin(delitos_gdf, anillos_gdf, how="inner", predicate="within")
join = join.sort_values("anillo").drop_duplicates(subset="id_delito", keep="first")

conteos = (
    join.groupby(["id", "anillo"])
    .size()
    .reset_index(name="cantidad_delitos")
)

anillos_gdf = anillos_gdf.merge(conteos, on=["id", "anillo"], how="left")
anillos_gdf["cantidad_delitos"] = anillos_gdf["cantidad_delitos"].fillna(0)
anillos_gdf["area_km2"] = anillos_gdf.geometry.area / 1_000_000
anillos_gdf["densidad"] = anillos_gdf["cantidad_delitos"] / anillos_gdf["area_km2"]

# Densidad relativa respecto al anillo 1 de cada comisaría
base = (
    anillos_gdf[anillos_gdf["anillo"] == 1][["id", "densidad"]]
    .rename(columns={"densidad": "base"})
)
anillos_gdf = anillos_gdf.merge(base, on="id", how="left")
anillos_gdf["densidad_relativa"] = np.where(
    anillos_gdf["base"] > 0,
    anillos_gdf["densidad"] / anillos_gdf["base"],
    np.nan
)

# Export CSV
anillos_gdf.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
print(f"💾 CSV guardado en: {OUTPUT_CSV}")

# Resumen estadístico
resumen = (
    anillos_gdf.groupby("anillo")
    .agg(
        distancia         =("distancia",       "first"),
        delitos_totales   =("cantidad_delitos", "sum"),
        densidad_promedio =("densidad",         "mean"),
    )
    .reset_index()
)
base_global = resumen.loc[resumen["anillo"] == 1, "densidad_promedio"].values[0]
resumen["densidad_relativa_pct"] = resumen["densidad_promedio"] / base_global * 100

print("\n📊 RESUMEN POR ANILLO:")
print(resumen.to_string(index=False, float_format="%.2f"))

# Mapa interactivo
anillos_wgs = anillos_gdf.to_crs(epsg=4326)
mapa = folium.Map(location=[-34.61, -58.43], zoom_start=12)

vals = anillos_wgs["densidad_relativa"].dropna()
p20, p80 = vals.quantile(0.2), vals.quantile(0.8)
p20, p80 = min(p20, 1), max(p80, 1)

colormap = cm.LinearColormap(["green", "yellow", "red"], vmin=p20, vmax=p80)
colormap.add_to(mapa)

def clip(v):
    if pd.isna(v): return None
    return max(min(v, p80), p20)

for _, row in anillos_wgs.iterrows():
    val   = clip(row["densidad_relativa"])
    color = "#cccccc" if val is None else colormap(val)
    tooltip = (
        f"<b>{row['nombre']}</b><br>"
        f"Barrio: {row['barrio']}<br>"
        f"Anillo: {row['distancia']}<br>"
        f"Delitos: {int(row['cantidad_delitos'])}<br>"
        f"Densidad: {row['densidad']:.1f} del/km²<br>"
        f"Relativa: {row['densidad_relativa']:.2f}"
    )
    folium.GeoJson(
        row["geometry"].__geo_interface__,
        style_function=lambda f, c=color: {
            "fillColor": c, "color": c,
            "weight": 1, "fillOpacity": 0.6
        },
        tooltip=tooltip
    ).add_to(mapa)

# Marcadores de comisarías
for _, row in comisarias.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=4, color="black", fill=True,
        tooltip=row.get("nombre", "Comisaría")
    ).add_to(mapa)

mapa.save(OUTPUT_HTML)
print(f"🗺️ Mapa guardado en: {OUTPUT_HTML}")

# ═══════════════════════════════════════════════════════════
# BLOQUE 2 — MANN-WHITNEY U ENTRE ANILLOS
# ═══════════════════════════════════════════════════════════

print("\n══ BLOQUE 2: Mann-Whitney U entre anillos ══")

grupos = {
    i: anillos_gdf[anillos_gdf["anillo"] == i]["densidad"].values
    for i in range(1, 5)
}

pares = [
    ("Anillo 1 vs 2  (0-300m vs 300-600m)",    1, 2),
    ("Anillo 1 vs 3  (0-300m vs 600-900m)",    1, 3),
    ("Anillo 1 vs 4  (0-300m vs 900-1200m)",   1, 4),
    ("Anillo 2 vs 3  (300-600m vs 600-900m)",  2, 3),
    ("Anillo 2 vs 4  (300-600m vs 900-1200m)", 2, 4),
    ("Anillo 3 vs 4  (600-900m vs 900-1200m)", 3, 4),
]

print(f"\n{'Comparación':<46} {'U-stat':>10} {'p-value':>12} {'Sig. (α=0.05)':>14}")
print("-" * 86)

resultados_mw = []
for label, i, j in pares:
    u_stat, p_val = stats.mannwhitneyu(grupos[i], grupos[j], alternative="two-sided")
    sig = "✅ Sí" if p_val < 0.05 else "❌ No"
    print(f"{label:<46} {u_stat:>10.1f} {p_val:>12.4e} {sig:>14}")
    resultados_mw.append({
        "comparacion":   label,
        "anillo_a":      i,
        "anillo_b":      j,
        "u_stat":        u_stat,
        "p_value":       p_val,
        "significativo": p_val < 0.05
    })

# Gráfico: boxplot con comparaciones adyacentes anotadas
fig, ax = plt.subplots(figsize=(11, 6))

orden_labels = ["0-300 m", "300-600 m", "600-900 m", "900-1200 m"]
data_plot = anillos_gdf[["anillo", "distancia", "densidad"]].copy()
data_plot["distancia"] = pd.Categorical(
    data_plot["distancia"], categories=orden_labels, ordered=True
)

sns.boxplot(
    data=data_plot,
    x="distancia",
    y="densidad",
    order=orden_labels,
    palette=["#d73027", "#fc8d59", "#fee08b", "#91cf60"],
    ax=ax,
    hue="distancia",
    legend=False
)

# Anotar solo comparaciones adyacentes sobre el gráfico
y_max = data_plot["densidad"].quantile(0.95)
adyacentes = [
    (0, 1, next(r for r in resultados_mw if r["anillo_a"]==1 and r["anillo_b"]==2)["p_value"], 0.05),
    (1, 2, next(r for r in resultados_mw if r["anillo_a"]==2 and r["anillo_b"]==3)["p_value"], 0.20),
    (2, 3, next(r for r in resultados_mw if r["anillo_a"]==3 and r["anillo_b"]==4)["p_value"], 0.35),
]

for x1, x2, p, offset in adyacentes:
    y = y_max * (1 + offset)
    ax.plot([x1, x1, x2, x2], [y * 0.97, y, y, y * 0.97], lw=1, color="black")
    sig_label = f"p={p:.3f}" if p >= 0.001 else f"p={p:.2e}"
    ax.text((x1 + x2) / 2, y * 1.02, sig_label, ha="center", va="bottom", fontsize=9)

ax.set_title(
    "Densidad de delitos por anillo de distancia a comisarías\n"
    "Mann-Whitney U entre anillos adyacentes (α = 0.05)",
    fontsize=13
)
ax.set_xlabel("Distancia a la comisaría")
ax.set_ylabel("Densidad de delitos (del/km²)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
sns.despine()

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"\n📊 Gráfico guardado en: {OUTPUT_IMG}")

plt.show()
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
"""
Script: 01_generar_grilla_caba_250m.py
Descripción:
Genera una grilla regular de 250x250 metros sobre la Ciudad Autónoma
de Buenos Aires, usando el dataset de barrios como límite espacial.

Salida:
outputs/grilla_caba_250m.geojson
"""

from pathlib import Path
import warnings

import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import box

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]  # tesis/

DATA_RAW = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

BARRIOS_PATH = DATA_RAW / "barrios.csv"
OUTPUT_GEOJSON = OUTPUT_DIR / "grilla_caba_250m.geojson"

CELL_SIZE = 250  # metros
CRS_ORIGINAL = "EPSG:4326"
CRS_METRICO = "EPSG:32721"  # UTM 21S - adecuado para Buenos Aires

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCIONES
# ============================================================

def cargar_barrios(path: Path) -> gpd.GeoDataFrame:
    """Carga barrios.csv y convierte la geometría WKT a GeoDataFrame."""

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    print(f"📂 Leyendo barrios desde: {path.relative_to(BASE_DIR)}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    if "geometry" not in df.columns:
        raise ValueError(
            "No se encontró la columna 'geometry' en barrios.csv. "
            "Verificá que el archivo tenga geometrías en formato WKT."
        )

    df["geometry"] = df["geometry"].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_ORIGINAL)

    # Reparar geometrías inválidas
    gdf["geometry"] = gdf["geometry"].make_valid()

    return gdf


def generar_grilla(caba_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Genera grilla regular 250x250m y conserva celdas que intersectan CABA."""

    print("🗺️  Unificando barrios en un único polígono de CABA...")
    caba_union = caba_gdf.dissolve()

    print(f"📐 Proyectando a CRS métrico {CRS_METRICO}...")
    caba_m = caba_union.to_crs(CRS_METRICO)

    caba_geom = caba_m.geometry.iloc[0]

    xmin, ymin, xmax, ymax = caba_m.total_bounds

    print(f"🔲 Generando grilla regular de {CELL_SIZE} x {CELL_SIZE} metros...")

    grid_cells = []

    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            cell = box(x, y, x + CELL_SIZE, y + CELL_SIZE)

            if cell.intersects(caba_geom):
                grid_cells.append(cell)

            y += CELL_SIZE
        x += CELL_SIZE

    grilla = gpd.GeoDataFrame(
        {"geometry": grid_cells},
        geometry="geometry",
        crs=CRS_METRICO
    )

    grilla = grilla.reset_index(drop=True)
    grilla["grid_id"] = grilla.index + 1

    # Área total de la celda regular
    grilla["area_celda_m2"] = CELL_SIZE * CELL_SIZE

    # Área real de la celda dentro de CABA
    grilla["area_interseccion_caba_m2"] = grilla.geometry.intersection(caba_geom).area

    # Porcentaje de la celda que cae dentro de CABA
    grilla["porcentaje_en_caba"] = (
        grilla["area_interseccion_caba_m2"] / grilla["area_celda_m2"]
    )

    return grilla


def exportar_grilla(grilla: gpd.GeoDataFrame) -> None:
    """Exporta la grilla a GeoJSON en EPSG:4326."""

    print("🌍 Reproyectando grilla a WGS84 para exportar...")
    grilla_wgs = grilla.to_crs(CRS_ORIGINAL)

    grilla_wgs = grilla_wgs[
        [
            "grid_id",
            "area_celda_m2",
            "area_interseccion_caba_m2",
            "porcentaje_en_caba",
            "geometry",
        ]
    ]

    if OUTPUT_GEOJSON.exists():
        OUTPUT_GEOJSON.unlink()

    print(f"💾 Guardando GeoJSON en: {OUTPUT_GEOJSON.relative_to(BASE_DIR)}")
    grilla_wgs.to_file(OUTPUT_GEOJSON, driver="GeoJSON")


def imprimir_resumen(grilla: gpd.GeoDataFrame) -> None:
    """Imprime estadísticas finales de la grilla."""

    area_total_celdas_km2 = grilla["area_celda_m2"].sum() / 1_000_000
    area_real_caba_km2 = grilla["area_interseccion_caba_m2"].sum() / 1_000_000

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("===================================================")
    print(f"Cantidad de celdas generadas:        {len(grilla):,}")
    print(f"Tamaño de celda:                     {CELL_SIZE} x {CELL_SIZE} metros")
    print(f"Superficie total de celdas:          {area_total_celdas_km2:,.2f} km²")
    print(f"Superficie real cubierta en CABA:    {area_real_caba_km2:,.2f} km²")
    print(f"Porcentaje promedio dentro de CABA:  {grilla['porcentaje_en_caba'].mean() * 100:.2f}%")
    print(f"Archivo exportado:                   outputs/{OUTPUT_GEOJSON.name}")
    print("===================================================")


# ============================================================
# MAIN
# ============================================================

def main():
    print("===================================================")
    print("GENERACIÓN DE GRILLA CABA 250m x 250m")
    print("===================================================")

    barrios_gdf = cargar_barrios(BARRIOS_PATH)
    grilla = generar_grilla(barrios_gdf)
    exportar_grilla(grilla)
    imprimir_resumen(grilla)


if __name__ == "__main__":
    main()
"""
Script: 02_delitos_a_grilla_mes_franja.py

Descripción:
Asigna delitos a la grilla CABA 250m x 250m y genera el dataset base
para clasificación de hotspots por celda + mes + franja horaria.

Unidad de análisis:
    grid_id + mes + franja

Franjas horarias:
    - Madrugada: [00, 06)
    - Mañana:    [06, 12)
    - Tarde:     [12, 18)
    - Noche:     [18, 24)

Entradas:
    - outputs/grilla_caba_250m.geojson
    - data/processed/delitos_total.csv.gz

Columnas del CSV de salida:
    - grid_id                 Identificador de celda
    - mes                     Período mensual (YYYY-MM)
    - franja                  Franja horaria
    - cantidad_delitos         Suma de delitos en la celda/mes/franja
    - umbral_p90_mes_franja   Percentil 90 calculado sobre positivos del grupo mes+franja
    - hotspot_exploratorio    1 si cantidad_delitos >= umbral_p90_mes_franja, 0 si no

Salida:
    - outputs/dataset_hotspots_base.csv

Nota metodológica — data leakage:
    El hotspot_exploratorio se calcula con todos los datos disponibles y sirve
    para análisis exploratorio (EDA, mapas, distribuciones).
    Para el modelo final con validación cruzada temporal (TimeSeriesSplit),
    el target DEBE recalcularse dentro de cada split usando exclusivamente
    los datos de entrenamiento, para evitar data leakage temporal.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR    = BASE_DIR / "outputs"

DELITOS_PATH = DATA_PROCESSED / "delitos_total.csv.gz"
GRILLA_PATH  = OUTPUT_DIR / "grilla_caba_250m.geojson"
OUTPUT_CSV   = OUTPUT_DIR / "dataset_hotspots_base.csv"

CRS_ORIGINAL = "EPSG:4326"

# Mismo CRS métrico que en 01_generar_grilla_caba_250m.py.
# Si cambiás este valor, cambiálo también en el script de grilla.
CRS_METRICO = "EPSG:5347"

FRANJAS = ["Madrugada", "Manana", "Tarde", "Noche"]

# Bounding box defensivo de CABA en grados decimales (WGS 84).
# Filtra coordenadas claramente fuera del territorio antes del join espacial.
CABA_LAT = (-34.75, -34.45)
CABA_LON = (-58.65, -58.25)


# ============================================================
# FUNCIONES
# ============================================================

def cargar_delitos() -> pd.DataFrame:
    print(f"📂 Cargando delitos desde: {DELITOS_PATH.relative_to(BASE_DIR)}")

    if not DELITOS_PATH.exists():
        raise FileNotFoundError(f"No se encontró: {DELITOS_PATH}")

    usecols = {"fecha", "franja", "latitud", "longitud", "cantidad"}

    delitos = pd.read_csv(
        DELITOS_PATH,
        usecols=lambda c: c.strip().lower() in usecols,
        dtype=str,
        low_memory=False,
    )

    delitos.columns = delitos.columns.str.strip().str.lower()

    print(f"  Registros crudos leídos: {len(delitos):,}")
    print(f"  Columnas:                {delitos.columns.tolist()}")
    print(f"  Valores únicos de franja detectados:")
    print(f"    {delitos['franja'].dropna().astype(str).str.strip().unique()[:30]}")

    return delitos


def clasificar_franja(valor) -> str | None:
    """
    Convierte distintos formatos de hora/franja a 4 franjas horarias.

    Acepta:
        - Strings descriptivos: "Madrugada", "Manana", "Tarde", "Noche"
        - Valores numéricos (int o float como string): hora del día 0–23

    Intervalos aplicados:
        Madrugada: [00, 06)
        Manana:    [06, 12)
        Tarde:     [12, 18)
        Noche:     [18, 24)
    """
    if pd.isna(valor):
        return None

    v = str(valor).strip().lower().replace(",", ".")

    if "madrugada" in v:
        return "Madrugada"
    if "mañana" in v or "manana" in v:
        return "Manana"
    if "tarde" in v:
        return "Tarde"
    if "noche" in v:
        return "Noche"

    try:
        hora = int(float(v))
    except ValueError:
        return None

    if 0 <= hora < 6:
        return "Madrugada"
    if 6 <= hora < 12:
        return "Manana"
    if 12 <= hora < 18:
        return "Tarde"
    if 18 <= hora < 24:
        return "Noche"

    return None


def preparar_delitos(delitos: pd.DataFrame) -> gpd.GeoDataFrame:
    print("🧹 Preparando delitos...")

    lat_col = "latitud"
    lon_col = "longitud"

    for col in [lat_col, lon_col, "cantidad", "fecha", "franja"]:
        if col not in delitos.columns:
            raise ValueError(f"Columna requerida no encontrada: '{col}'")

    # Normalizar separadores decimales
    for col in [lat_col, lon_col, "cantidad"]:
        delitos[col] = (
            delitos[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

    delitos[lat_col]   = pd.to_numeric(delitos[lat_col],   errors="coerce")
    delitos[lon_col]   = pd.to_numeric(delitos[lon_col],   errors="coerce")
    delitos["cantidad"] = pd.to_numeric(delitos["cantidad"], errors="coerce")

    # Fallback: si cantidad es nula, asumir 1 evento por registro
    delitos["cantidad"] = delitos["cantidad"].fillna(1)

    # Eliminar coordenadas nulas
    delitos = delitos.dropna(subset=[lat_col, lon_col]).copy()

    # Filtro defensivo: coordenadas fuera del bounding box de CABA
    n_antes = len(delitos)
    delitos = delitos[
        delitos[lat_col].between(*CABA_LAT)
        & delitos[lon_col].between(*CABA_LON)
    ].copy()
    n_filtrados = n_antes - len(delitos)
    if n_filtrados > 0:
        print(
            f"  ⚠️  Coordenadas fuera de CABA descartadas: "
            f"{n_filtrados:,} ({n_filtrados / n_antes * 100:.1f}%)"
        )

    # Fechas
    delitos["fecha"] = pd.to_datetime(delitos["fecha"], errors="coerce")
    delitos = delitos.dropna(subset=["fecha"]).copy()
    delitos["mes"] = delitos["fecha"].dt.to_period("M").astype(str)

    # Franjas
    delitos["franja"] = delitos["franja"].apply(clasificar_franja)
    print("  Distribución de franjas convertidas:")
    print(delitos["franja"].value_counts(dropna=False).to_string(index=True))

    delitos = delitos[delitos["franja"].isin(FRANJAS)].copy()

    print(f"  Delitos válidos preparados: {len(delitos):,}")
    print(f"  Suma de cantidad:           {delitos['cantidad'].sum():,.0f}")

    # Construir GeoDataFrame y reproyectar
    delitos_gdf = gpd.GeoDataFrame(
        delitos[["mes", "franja", "cantidad"]],
        geometry=gpd.points_from_xy(delitos[lon_col], delitos[lat_col]),
        crs=CRS_ORIGINAL,
    ).to_crs(CRS_METRICO)

    return delitos_gdf


def cargar_grilla() -> gpd.GeoDataFrame:
    print(f"📂 Cargando grilla desde: {GRILLA_PATH.relative_to(BASE_DIR)}")

    if not GRILLA_PATH.exists():
        raise FileNotFoundError(f"No se encontró: {GRILLA_PATH}")

    grilla = gpd.read_file(GRILLA_PATH).to_crs(CRS_METRICO)

    if "grid_id" not in grilla.columns:
        raise ValueError("La grilla no tiene columna 'grid_id'.")

    print(f"  Celdas cargadas: {len(grilla):,}")
    print(f"  CRS:             {grilla.crs}")

    return grilla


def asignar_delitos_a_grilla(
    delitos_gdf: gpd.GeoDataFrame,
    grilla: gpd.GeoDataFrame,
) -> pd.DataFrame:
    print("📍 Asignando delitos a celdas...")

    # Guardia: CRS debe coincidir antes del join espacial
    assert delitos_gdf.crs == grilla.crs, (
        f"CRS mismatch — delitos: {delitos_gdf.crs} | grilla: {grilla.crs}"
    )

    delitos_gdf = delitos_gdf.reset_index(drop=True).copy()
    delitos_gdf["_id_delito"] = delitos_gdf.index

    join = gpd.sjoin(
        delitos_gdf,
        grilla[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    # Punto en borde exacto puede tocar más de una celda → conservar una sola
    join = (
        join
        .sort_values(["_id_delito", "grid_id"])
        .drop_duplicates(subset=["_id_delito"], keep="first")
    )

    n_asignados  = len(join)
    n_total      = len(delitos_gdf)
    n_no_asignados = n_total - n_asignados
    print(f"  Delitos asignados:     {n_asignados:,}")
    if n_no_asignados > 0:
        print(
            f"  ⚠️  Delitos no asignados: "
            f"{n_no_asignados:,} ({n_no_asignados / n_total * 100:.1f}%)"
        )

    conteos = (
        join
        .groupby(["grid_id", "mes", "franja"], as_index=False)
        .agg(cantidad_delitos=("cantidad", "sum"))
    )
    conteos["cantidad_delitos"] = conteos["cantidad_delitos"].round().astype(int)

    return conteos


def completar_combinaciones(
    conteos: pd.DataFrame,
    grilla: gpd.GeoDataFrame,
) -> pd.DataFrame:
    print("🧩 Completando combinaciones grid_id + mes + franja...")

    if conteos.empty:
        raise ValueError(
            "No se asignó ningún delito a la grilla. "
            "Revisá coordenadas, CRS, geometrías y franjas."
        )

    meses  = sorted(conteos["mes"].unique())
    grids  = sorted(grilla["grid_id"].unique())
    n_comb = len(grids) * len(meses) * len(FRANJAS)

    print(f"  {len(grids):,} celdas × {len(meses)} meses × {len(FRANJAS)} franjas = {n_comb:,} filas")

    base = pd.MultiIndex.from_product(
        [grids, meses, FRANJAS],
        names=["grid_id", "mes", "franja"],
    ).to_frame(index=False)

    df = base.merge(conteos, on=["grid_id", "mes", "franja"], how="left")
    df["cantidad_delitos"] = df["cantidad_delitos"].fillna(0).astype(int)

    return df


def crear_hotspot_exploratorio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea el target exploratorio: hotspot = 1 si la celda supera el
    percentil 90 de su grupo (mes + franja), calculado solo sobre
    celdas con al menos 1 delito.

    ⚠️ Para modelado con TimeSeriesSplit, recalcular el target dentro
    de cada fold usando exclusivamente datos de entrenamiento.
    """
    print("🔥 Creando hotspot exploratorio por mes + franja...")

    def p90_sobre_positivos(x: pd.Series) -> float:
        positivos = x[x > 0]
        return positivos.quantile(0.90) if len(positivos) > 0 else 0.0

    df["umbral_p90_mes_franja"] = (
        df.groupby(["mes", "franja"])["cantidad_delitos"]
        .transform(p90_sobre_positivos)
    )

    # hotspot = 1 solo si el grupo tiene delitos positivos Y la celda supera el umbral
    df["hotspot_exploratorio"] = (
        (df["umbral_p90_mes_franja"] > 0)
        & (df["cantidad_delitos"] >= df["umbral_p90_mes_franja"])
    ).astype(int)

    tasa = df["hotspot_exploratorio"].mean() * 100
    print(f"  Hotspots: {df['hotspot_exploratorio'].sum():,}  ({tasa:.2f}% del total)")

    return df


def exportar_dataset(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Dataset exportado: {OUTPUT_CSV.relative_to(BASE_DIR)}")
    print(f"   Tamaño:            {OUTPUT_CSV.stat().st_size / 1024:.1f} KB")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 55)
    print("GENERACIÓN DATASET BASE HOTSPOTS")
    print("Celda 250m x 250m + Mes + Franja horaria")
    print("=" * 55)

    delitos    = cargar_delitos()
    delitos_gdf = preparar_delitos(delitos)
    grilla     = cargar_grilla()

    conteos  = asignar_delitos_a_grilla(delitos_gdf, grilla)
    df_base  = completar_combinaciones(conteos, grilla)
    df_base  = crear_hotspot_exploratorio(df_base)

    exportar_dataset(df_base)

    print()
    print("=" * 55)
    print("✅ PROCESO COMPLETADO")
    print("=" * 55)
    print(f"  Filas dataset final:         {len(df_base):,}")
    print(f"  Celdas:                      {df_base['grid_id'].nunique():,}")
    print(f"  Meses:                       {df_base['mes'].nunique():,}")
    print(f"  Franjas:                     {df_base['franja'].nunique()}")
    print(f"  Suma cantidad_delitos:       {df_base['cantidad_delitos'].sum():,}")
    print(f"  Hotspots exploratorios:      {df_base['hotspot_exploratorio'].sum():,}")
    print(f"  Archivo generado:            outputs/{OUTPUT_CSV.name}")
    print("=" * 55)


if __name__ == "__main__":
    main()
"""
Script: ML_03_features_urbanas.py

Descripción:
Calcula e incorpora variables espaciales de infraestructura urbana al dataset base
de hotspots. Todas las features se calculan contra la grilla oficial de 250m x 250m.

Unidad espacial:
    grid_id

Entradas:
    - outputs/dataset_hotspots_base.csv
    - outputs/grilla_caba_250m.geojson

    - data/processed/alojamientos_unificados.csv

    - data/raw/cajeros-automaticos.csv
    - data/raw/comisarias-policia-de-la-ciudad.xlsx
    - data/raw/paradas-de-colectivo.xlsx
    - data/raw/estaciones-de-ferrocarril.csv
    - data/raw/oferta_gastronomica.xlsx

Salida:
    - outputs/dataset_ml_features.csv
"""

from pathlib import Path
import warnings

import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

GRILLA_PATH = OUTPUT_DIR / "grilla_caba_250m.geojson"
DATASET_BASE_PATH = OUTPUT_DIR / "dataset_hotspots_base.csv"
OUTPUT_CSV = OUTPUT_DIR / "dataset_ml_features.csv"

CAJEROS_PATH = DATA_RAW / "cajeros-automaticos.csv"
COMISARIAS_PATH = DATA_RAW / "comisarias-policia-de-la-ciudad.xlsx"
COLECTIVOS_PATH = DATA_RAW / "paradas-de-colectivo.xlsx"
TREN_PATH = DATA_RAW / "estaciones-de-ferrocarril.csv"
GASTRO_PATH = DATA_RAW / "oferta_gastronomica.xlsx"

ALOJAMIENTOS_PATH = DATA_PROCESSED / "alojamientos_unificados.csv"

CRS_ORIGINAL = "EPSG:4326"
CRS_METRICO = "EPSG:32721"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def verificar_archivo(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")


def leer_archivo(path: Path) -> pd.DataFrame:
    """
    Lee CSV o Excel según extensión.
    """
    verificar_archivo(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Extensión no soportada: {path.suffix}")

    df.columns = df.columns.str.strip().str.lower()
    return df


def convertir_numero_serie(s: pd.Series) -> pd.Series:
    """
    Convierte números que pueden venir con coma decimal o caracteres extraños.
    """
    return pd.to_numeric(
        s.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )


def buscar_columna(df: pd.DataFrame, posibles: list[str]) -> str:
    """
    Busca una columna dentro de un conjunto de nombres posibles.
    """
    cols = set(df.columns)

    for col in posibles:
        if col in cols:
            return col

    raise ValueError(
        f"No se encontró ninguna columna entre {posibles}. "
        f"Columnas disponibles: {df.columns.tolist()}"
    )


def construir_gdf_puntos(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    nombre_dataset: str,
) -> gpd.GeoDataFrame:
    """
    Convierte un DataFrame con lat/lon en GeoDataFrame métrico.
    """
    df = df.copy()

    df[lat_col] = convertir_numero_serie(df[lat_col])
    df[lon_col] = convertir_numero_serie(df[lon_col])

    n_antes = len(df)
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    n_descartados = n_antes - len(df)

    if n_descartados > 0:
        print(f"    ⚠️ {nombre_dataset}: coordenadas nulas/invalidas descartadas: {n_descartados:,}")

    if df.empty:
        print(f"    ⚠️ {nombre_dataset}: no quedaron puntos válidos.")
        return gpd.GeoDataFrame(df, geometry=[], crs=CRS_METRICO)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=CRS_ORIGINAL
    ).to_crs(CRS_METRICO)

    return gdf


def contar_puntos_por_grilla(
    grilla: gpd.GeoDataFrame,
    puntos: gpd.GeoDataFrame,
    nombre_feature: str,
) -> pd.DataFrame:
    """
    Calcula cantidad de puntos dentro de cada celda.
    """
    if puntos.empty:
        return pd.DataFrame({
            "grid_id": grilla["grid_id"],
            nombre_feature: 0
        })

    join = gpd.sjoin(
        puntos,
        grilla[["grid_id", "geometry"]],
        how="inner",
        predicate="within"
    )

    conteos = (
        join.groupby("grid_id")
        .size()
        .reset_index(name=nombre_feature)
    )

    base = grilla[["grid_id"]].copy()
    base = base.merge(conteos, on="grid_id", how="left")
    base[nombre_feature] = base[nombre_feature].fillna(0).astype(int)

    return base


def distancia_minima_a_puntos(
    grilla: gpd.GeoDataFrame,
    puntos: gpd.GeoDataFrame,
    nombre_feature: str,
) -> pd.DataFrame:
    """
    Calcula distancia desde el centroide de cada celda al punto más cercano.
    """
    centroides = grilla[["grid_id", "geometry"]].copy()
    centroides["geometry"] = centroides.geometry.centroid

    if puntos.empty:
        centroides[nombre_feature] = pd.NA
        return centroides[["grid_id", nombre_feature]]

    nearest = gpd.sjoin_nearest(
        centroides,
        puntos[["geometry"]],
        how="left",
        distance_col=nombre_feature
    )

    nearest = nearest[["grid_id", nombre_feature]].drop_duplicates("grid_id")
    nearest[nombre_feature] = nearest[nombre_feature].round(2)

    return nearest


def agregar_features_puntos(
    df_base: pd.DataFrame,
    grilla: gpd.GeoDataFrame,
    path: Path,
    nombre: str,
    prefijo: str,
    posibles_lat: list[str],
    posibles_lon: list[str],
    calcular_distancia: bool = True,
) -> pd.DataFrame:
    """
    Lee un dataset de puntos, lo cruza con la grilla y agrega:
        - cant_<prefijo>
        - dist_min_<prefijo>_m
    """
    print(f"\n📍 Procesando {nombre}...")

    df_puntos = leer_archivo(path)

    lat_col = buscar_columna(df_puntos, posibles_lat)
    lon_col = buscar_columna(df_puntos, posibles_lon)

    print(f"    Columnas usadas: lat='{lat_col}' | lon='{lon_col}'")

    puntos = construir_gdf_puntos(
        df=df_puntos,
        lat_col=lat_col,
        lon_col=lon_col,
        nombre_dataset=nombre
    )

    cant_col = f"cant_{prefijo}"

    conteos = contar_puntos_por_grilla(
        grilla=grilla,
        puntos=puntos,
        nombre_feature=cant_col
    )

    df_base = df_base.merge(conteos, on="grid_id", how="left")
    df_base[cant_col] = df_base[cant_col].fillna(0).astype(int)

    if calcular_distancia:
        dist_col = f"dist_min_{prefijo}_m"

        distancias = distancia_minima_a_puntos(
            grilla=grilla,
            puntos=puntos,
            nombre_feature=dist_col
        )

        df_base = df_base.merge(distancias, on="grid_id", how="left")

    print(f"    Feature agregada: {cant_col} | suma={df_base[cant_col].sum():,}")

    return df_base


def validar_merge(df: pd.DataFrame, filas_originales: int, etapa: str) -> None:
    if len(df) != filas_originales:
        raise ValueError(
            f"Error en {etapa}: el merge cambió la cantidad de filas. "
            f"Antes={filas_originales:,}, después={len(df):,}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 60)
    print("FEATURE ENGINEERING URBANO UNIFICADO - ML")
    print("=" * 60)

    print("\n📂 Cargando grilla y dataset base...")

    verificar_archivo(GRILLA_PATH)
    verificar_archivo(DATASET_BASE_PATH)

    grilla = gpd.read_file(GRILLA_PATH).to_crs(CRS_METRICO)

    if "grid_id" not in grilla.columns:
        raise ValueError("La grilla no tiene columna 'grid_id'.")

    df_final = pd.read_csv(DATASET_BASE_PATH)
    filas_originales = len(df_final)

    print(f"  Celdas grilla:          {grilla['grid_id'].nunique():,}")
    print(f"  Filas dataset base:     {filas_originales:,}")
    print(f"  Columnas dataset base:  {len(df_final.columns):,}")

    # ========================================================
    # FEATURES URBANAS DESDE DATASETS DE PUNTOS
    # ========================================================

    df_final = agregar_features_puntos(
        df_base=df_final,
        grilla=grilla,
        path=CAJEROS_PATH,
        nombre="cajeros automáticos",
        prefijo="cajeros",
        posibles_lat=["lat", "latitude", "latitud"],
        posibles_lon=["long", "lon", "lng", "longitude", "longitud"],
        calcular_distancia=True,
    )
    validar_merge(df_final, filas_originales, "cajeros")

    df_final = agregar_features_puntos(
        df_base=df_final,
        grilla=grilla,
        path=ALOJAMIENTOS_PATH,
        nombre="alojamientos turísticos / Airbnb",
        prefijo="alojamientos",
        posibles_lat=["lat", "latitude", "latitud"],
        posibles_lon=["long", "lon", "lng", "longitude", "longitud"],
        calcular_distancia=True,
    )
    validar_merge(df_final, filas_originales, "alojamientos")

    df_final = agregar_features_puntos(
        df_base=df_final,
        grilla=grilla,
        path=COMISARIAS_PATH,
        nombre="comisarías",
        prefijo="comisarias",
        posibles_lat=["lat", "latitude", "latitud"],
        posibles_lon=["long", "lon", "lng", "longitude", "longitud"],
        calcular_distancia=True,
    )
    validar_merge(df_final, filas_originales, "comisarías")

    df_final = agregar_features_puntos(
        df_base=df_final,
        grilla=grilla,
        path=COLECTIVOS_PATH,
        nombre="paradas de colectivo",
        prefijo="colectivos",
        posibles_lat=["coord_y", "lat", "latitude", "latitud", "y"],
        posibles_lon=["coord_x", "long", "lon", "lng", "longitude", "longitud", "x"],
        calcular_distancia=True,
    )
    validar_merge(df_final, filas_originales, "colectivos")

    df_final = agregar_features_puntos(
        df_base=df_final,
        grilla=grilla,
        path=TREN_PATH,
        nombre="estaciones de ferrocarril",
        prefijo="estaciones_tren",
        posibles_lat=["lat", "latitude", "latitud"],
        posibles_lon=["long", "lon", "lng", "longitude", "longitud"],
        calcular_distancia=True,
    )
    validar_merge(df_final, filas_originales, "estaciones de ferrocarril")

    df_final = agregar_features_puntos(
        df_base=df_final,
        grilla=grilla,
        path=GASTRO_PATH,
        nombre="oferta gastronómica",
        prefijo="gastronomia",
        posibles_lat=["lat", "latitude", "latitud"],
        posibles_lon=["long", "lon", "lng", "longitude", "longitud"],
        calcular_distancia=True,
    )
    validar_merge(df_final, filas_originales, "gastronomía")

    # ========================================================
    # ATRIBUTOS DE LA GRILLA
    # ========================================================

    print("\n📐 Agregando atributos propios de la grilla...")

    cols_grilla = [
        "grid_id",
        "area_celda_m2",
        "area_interseccion_caba_m2",
        "porcentaje_en_caba"
    ]

    cols_grilla = [c for c in cols_grilla if c in grilla.columns]

    df_final = df_final.merge(
        grilla[cols_grilla],
        on="grid_id",
        how="left"
    )

    validar_merge(df_final, filas_originales, "atributos de grilla")

    # ========================================================
    # LIMPIEZA FINAL
    # ========================================================

    feature_count_cols = [c for c in df_final.columns if c.startswith("cant_")]
    for col in feature_count_cols:
        df_final[col] = df_final[col].fillna(0).astype(int)

    dist_cols = [c for c in df_final.columns if c.startswith("dist_min_")]
    for col in dist_cols:
        df_final[col] = df_final[col].fillna(df_final[col].max())

    # ========================================================
    # EXPORTACIÓN
    # ========================================================

    print(f"\n💾 Exportando dataset enriquecido a: {OUTPUT_CSV.relative_to(BASE_DIR)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("===================================================")
    print(f"  Filas dataset resultante:    {len(df_final):,}")
    print(f"  Columnas dataset resultante: {len(df_final.columns):,}")
    print(f"  Archivo generado:            outputs/{OUTPUT_CSV.name}")

    print("\n  Nuevas variables predictoras:")
    nuevas_cols = [
        c for c in df_final.columns
        if c.startswith("cant_") or c.startswith("dist_min_")
    ]

    for col in nuevas_cols:
        print(
            f"    - {col}: "
            f"min={df_final[col].min():,.2f} | "
            f"max={df_final[col].max():,.2f} | "
            f"mean={df_final[col].mean():,.2f}"
        )

    print("===================================================")


if __name__ == "__main__":
    main()
"""
Script: ML_04_features_temporales.py

Agrega variables temporales e históricas al dataset con features urbanas.
Evita data leakage usando solo información pasada.
"""

from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_CSV = OUTPUT_DIR / "dataset_ml_features.csv"
OUTPUT_CSV = OUTPUT_DIR / "dataset_ml_features_temporales.csv"

TARGET_COL = "cantidad_delitos"
HOTSPOT_COL = "hotspot_exploratorio"


# ============================================================
# FUNCIONES
# ============================================================

def exportar_csv_seguro(df: pd.DataFrame, output_path: Path) -> None:
    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError:
            raise PermissionError(
                f"No se puede sobrescribir {output_path}. "
                "Cerrá el archivo si está abierto."
            )

    df.to_csv(output_path, index=False)


def agregar_features_historicas(df, group_cols, sufijo):
    df = df.sort_values(group_cols + ["fecha_mes"]).copy()

    col_lag = f"delitos_mes_anterior_{sufijo}"
    col_roll3 = f"promedio_3_meses_{sufijo}"
    col_roll6 = f"promedio_6_meses_{sufijo}"
    col_hist_mean = f"promedio_historico_{sufijo}"
    col_hist_sum = f"delitos_historicos_{sufijo}"

    df[col_lag] = (
        df.groupby(group_cols)[TARGET_COL]
        .shift(1)
        .fillna(0)
    )

    df[col_roll3] = (
        df.groupby(group_cols)[col_lag]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .fillna(0)
    )

    df[col_roll6] = (
        df.groupby(group_cols)[col_lag]
        .transform(lambda x: x.rolling(6, min_periods=1).mean())
        .fillna(0)
    )

    df[col_hist_mean] = (
        df.groupby(group_cols)[col_lag]
        .transform(lambda x: x.expanding(min_periods=1).mean())
        .fillna(0)
    )

    df[col_hist_sum] = (
        df.groupby(group_cols)[col_lag]
        .transform(lambda x: x.expanding(min_periods=1).sum())
        .fillna(0)
    )

    return df


def agregar_promedios_estacionales(df):
    df = df.sort_values(["grid_id", "franja", "fecha_mes"]).copy()

    grupos = [
        (["grid_id", "mes_num"], "promedio_historico_mes_calendario_grid"),
        (["grid_id", "trimestre"], "promedio_historico_trimestre_grid"),
        (["grid_id", "franja", "mes_num"], "promedio_historico_mes_calendario_franja"),
        (["grid_id", "franja", "trimestre"], "promedio_historico_trimestre_franja"),
    ]

    for group_cols, new_col in grupos:
        lag_temp = f"__lag_{new_col}"

        df = df.sort_values(group_cols + ["fecha_mes"]).copy()

        df[lag_temp] = (
            df.groupby(group_cols)[TARGET_COL]
            .shift(1)
            .fillna(0)
        )

        df[new_col] = (
            df.groupby(group_cols)[lag_temp]
            .transform(lambda x: x.expanding(min_periods=1).mean())
            .fillna(0)
        )

        df = df.drop(columns=[lag_temp])

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING TEMPORAL - ML")
    print("=" * 60)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No se encontró: {INPUT_CSV}")

    print(f"\n📂 Cargando dataset desde: {INPUT_CSV.relative_to(BASE_DIR)}")
    df = pd.read_csv(INPUT_CSV)

    filas_originales = len(df)

    print(f"  Filas iniciales:    {len(df):,}")
    print(f"  Columnas iniciales: {len(df.columns):,}")

    # Validaciones básicas
    requeridas = ["grid_id", "mes", "franja", TARGET_COL]
    faltantes = [c for c in requeridas if c not in df.columns]

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {faltantes}\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    # Fecha mensual
    print("\n🧭 Preparando fecha mensual...")
    df["fecha_mes"] = pd.to_datetime(df["mes"].astype(str) + "-01", errors="coerce")

    if df["fecha_mes"].isna().any():
        raise ValueError("Hay valores inválidos en la columna mes.")

    # Variables calendario
    print("\n📅 Agregando variables calendario...")
    df["anio"] = df["fecha_mes"].dt.year
    df["mes_num"] = df["fecha_mes"].dt.month
    df["trimestre"] = df["fecha_mes"].dt.quarter
    df["semestre"] = df["mes_num"].apply(lambda x: 1 if x <= 6 else 2)

    # Features históricas por grid
    print("\n📈 Agregando features históricas por grid_id...")
    df = agregar_features_historicas(
        df=df,
        group_cols=["grid_id"],
        sufijo="grid"
    )

    # Features históricas por grid + franja
    print("\n📈 Agregando features históricas por grid_id + franja...")
    df = agregar_features_historicas(
        df=df,
        group_cols=["grid_id", "franja"],
        sufijo="franja"
    )

    # Hotspot anterior
    if HOTSPOT_COL in df.columns:
        print("\n🔥 Agregando hotspot anterior...")

        df = df.sort_values(["grid_id", "franja", "fecha_mes"]).copy()

        df["hotspot_mes_anterior_franja"] = (
            df.groupby(["grid_id", "franja"])[HOTSPOT_COL]
            .shift(1)
            .fillna(0)
            .astype(int)
        )

        df = df.sort_values(["grid_id", "fecha_mes", "franja"]).copy()

        df["hotspot_mes_anterior_grid"] = (
            df.groupby(["grid_id"])[HOTSPOT_COL]
            .shift(1)
            .fillna(0)
            .astype(int)
        )

    # Promedios históricos estacionales
    print("\n📊 Agregando promedios históricos estacionales...")
    df = agregar_promedios_estacionales(df)

    # Limpieza final
    print("\n🧹 Limpieza final...")
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    df = df.sort_values(["grid_id", "franja", "fecha_mes"]).reset_index(drop=True)

    if len(df) != filas_originales:
        raise ValueError(
            f"Error: cambió la cantidad de filas. "
            f"Antes={filas_originales:,}, después={len(df):,}"
        )

    # Exportar
    print(f"\n💾 Exportando dataset temporal a: {OUTPUT_CSV.relative_to(BASE_DIR)}")
    exportar_csv_seguro(df, OUTPUT_CSV)

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("===================================================")
    print(f"  Filas finales:      {len(df):,}")
    print(f"  Columnas finales:   {len(df.columns):,}")
    print(f"  Archivo generado:   outputs/{OUTPUT_CSV.name}")

    print("\n  Variables temporales agregadas:")
    nuevas = [
        c for c in df.columns
        if c.startswith("delitos_mes_anterior")
        or c.startswith("promedio_")
        or c.startswith("delitos_historicos")
        or c.startswith("hotspot_mes_anterior")
    ]

    for col in nuevas:
        print(f"    - {col}")

    print("===================================================")


if __name__ == "__main__":
    main()
"""
Script: ML_05_modelos_hotspots.py

Entrena modelos para predecir hotspots delictivos:
- Logistic Regression
- Random Forest
- XGBoost

Entrada:
    outputs/dataset_ml_features_temporales.csv

Salidas:
    outputs/resultados_modelos_hotspots.csv
    outputs/feature_importance_random_forest.csv
    outputs/feature_importance_xgboost.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_CSV = OUTPUT_DIR / "dataset_ml_features_temporales.csv"

OUTPUT_RESULTADOS = OUTPUT_DIR / "resultados_modelos_hotspots.csv"
OUTPUT_RF_IMPORTANCE = OUTPUT_DIR / "feature_importance_random_forest.csv"
OUTPUT_XGB_IMPORTANCE = OUTPUT_DIR / "feature_importance_xgboost.csv"

TARGET_COL = "hotspot_exploratorio"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# FUNCIONES
# ============================================================

def exportar_csv_seguro(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            raise PermissionError(
                f"No se puede sobrescribir {path}. Cerrá el archivo si está abierto."
            )
    df.to_csv(path, index=False)


def cargar_dataset() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No se encontró: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"No se encontró la columna target '{TARGET_COL}'. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    if "fecha_mes" not in df.columns:
        raise ValueError("No se encontró la columna 'fecha_mes'.")

    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"], errors="coerce")
    df = df.dropna(subset=["fecha_mes"]).copy()

    df = df.sort_values("fecha_mes").reset_index(drop=True)

    return df


def preparar_xy(df: pd.DataFrame):
    columnas_excluir = [
        TARGET_COL,
        "cantidad_delitos",
        "umbral_p90_mes_franja",
        "mes",
        "fecha_mes",
        "grid_id",
    ]

    columnas_excluir = [c for c in columnas_excluir if c in df.columns]

    X = df.drop(columns=columnas_excluir).copy()
    y = df[TARGET_COL].astype(int).copy()

    # Convertir variables categóricas a dummies
    X = pd.get_dummies(X, drop_first=True)

    # Quedarse solo con columnas numéricas
    X = X.select_dtypes(include=[np.number]).copy()

    # Reemplazar inf y nulos
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y


def obtener_modelos():
    modelos = {}

    modelos["Logistic Regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )

    modelos["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    try:
        from xgboost import XGBClassifier

        modelos["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    except ImportError:
        print("⚠️ XGBoost no está instalado. Para instalar:")
        print("   pip install xgboost")

    return modelos


def calcular_metricas(y_true, y_pred, y_proba):
    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba)
        if len(np.unique(y_true)) > 1 else np.nan,
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metricas.update({
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })

    return metricas


def evaluar_modelos(X, y):
    modelos = obtener_modelos()
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    resultados = []
    modelos_entrenados = {}

    for nombre_modelo, modelo in modelos.items():
        print("\n===================================================")
        print(f"ENTRENANDO MODELO: {nombre_modelo}")
        print("===================================================")

        fold = 1

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(
                f"Fold {fold}: "
                f"train={len(train_idx):,} | test={len(test_idx):,} | "
                f"hotspots train={y_train.mean()*100:.2f}% | "
                f"hotspots test={y_test.mean()*100:.2f}%"
            )

            modelo.fit(X_train, y_train)

            y_pred = modelo.predict(X_test)

            if hasattr(modelo, "predict_proba"):
                y_proba = modelo.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred

            metricas = calcular_metricas(y_test, y_pred, y_proba)

            metricas["modelo"] = nombre_modelo
            metricas["fold"] = fold
            resultados.append(metricas)

            fold += 1

        # Entrenar modelo final con todo el dataset
        modelo.fit(X, y)
        modelos_entrenados[nombre_modelo] = modelo

    return pd.DataFrame(resultados), modelos_entrenados


def exportar_feature_importance(modelos_entrenados, feature_names):
    if "Random Forest" in modelos_entrenados:
        rf = modelos_entrenados["Random Forest"]

        fi_rf = pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)

        exportar_csv_seguro(fi_rf, OUTPUT_RF_IMPORTANCE)

    if "XGBoost" in modelos_entrenados:
        xgb = modelos_entrenados["XGBoost"]

        fi_xgb = pd.DataFrame({
            "feature": feature_names,
            "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False)

        exportar_csv_seguro(fi_xgb, OUTPUT_XGB_IMPORTANCE)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("MODELADO PREDICTIVO DE HOTSPOTS")
    print("=" * 60)

    print("\n📂 Cargando dataset...")
    df = cargar_dataset()

    print(f"Filas:    {len(df):,}")
    print(f"Columnas: {len(df.columns):,}")
    print(f"Tasa de hotspots: {df[TARGET_COL].mean()*100:.2f}%")

    print("\n🧹 Preparando X e y...")
    X, y = preparar_xy(df)

    print(f"Features usadas: {X.shape[1]:,}")
    print(f"Observaciones:   {X.shape[0]:,}")

    print("\n🤖 Entrenando modelos con TimeSeriesSplit...")
    resultados_folds, modelos_entrenados = evaluar_modelos(X, y)

    print("\n📊 Resultados por fold:")
    print(resultados_folds.to_string(index=False))

    resumen = (
        resultados_folds
        .groupby("modelo")
        .agg({
            "accuracy": "mean",
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "roc_auc": "mean",
            "tn": "sum",
            "fp": "sum",
            "fn": "sum",
            "tp": "sum",
        })
        .reset_index()
        .sort_values("roc_auc", ascending=False)
    )

    print("\n📊 Resumen promedio por modelo:")
    print(resumen.to_string(index=False, float_format="%.4f"))

    exportar_csv_seguro(resumen, OUTPUT_RESULTADOS)

    print("\n📌 Exportando feature importance...")
    exportar_feature_importance(modelos_entrenados, X.columns.tolist())

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados:           outputs/{OUTPUT_RESULTADOS.name}")
    print(f"Feature Importance RF: outputs/{OUTPUT_RF_IMPORTANCE.name}")
    print(f"Feature Importance XGB: outputs/{OUTPUT_XGB_IMPORTANCE.name}")
    print("===================================================")


if __name__ == "__main__":
    main()
"""
Script: ML_05A_modelos_solo_infraestructura.py

Descripción:
Experimento A: entrena modelos usando únicamente variables de infraestructura urbana
y variables temporales básicas, excluyendo variables históricas delictivas.

Además, evalúa múltiples thresholds entre 0.05 y 0.95.

Entrada:
    outputs/dataset_ml_features_temporales.csv

Salidas:
    outputs/resultados_solo_infraestructura_thresholds.csv
    outputs/mejor_threshold_solo_infraestructura.csv
    outputs/feature_importance_rf_solo_infraestructura.csv
    outputs/feature_importance_xgb_solo_infraestructura.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_CSV = OUTPUT_DIR / "dataset_ml_features_temporales.csv"

OUTPUT_THRESHOLDS = OUTPUT_DIR / "resultados_solo_infraestructura_thresholds.csv"
OUTPUT_BEST = OUTPUT_DIR / "mejor_threshold_solo_infraestructura.csv"
OUTPUT_RF_IMPORTANCE = OUTPUT_DIR / "feature_importance_rf_solo_infraestructura.csv"
OUTPUT_XGB_IMPORTANCE = OUTPUT_DIR / "feature_importance_xgb_solo_infraestructura.csv"

TARGET_COL = "hotspot_exploratorio"

N_SPLITS = 5
RANDOM_STATE = 42

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


# ============================================================
# FUNCIONES
# ============================================================

def exportar_csv_seguro(df: pd.DataFrame, path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            raise PermissionError(
                f"No se puede sobrescribir {path}. Cerrá el archivo si está abierto."
            )

    df.to_csv(path, index=False)


def cargar_dataset() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No se encontró: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna target '{TARGET_COL}'.")

    if "fecha_mes" not in df.columns:
        raise ValueError("No se encontró la columna 'fecha_mes'.")

    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"], errors="coerce")
    df = df.dropna(subset=["fecha_mes"]).copy()
    df = df.sort_values("fecha_mes").reset_index(drop=True)

    return df


def seleccionar_features_infraestructura(df: pd.DataFrame) -> list[str]:
    candidatas = []

    for col in df.columns:
        if (
            col.startswith("cant_")
            or col.startswith("dist_min_")
            or col in [
                "area_celda_m2",
                "area_interseccion_caba_m2",
                "porcentaje_en_caba",
                "anio",
                "mes_num",
                "trimestre",
                "semestre",
                "franja",
            ]
        ):
            candidatas.append(col)

    patrones_excluir = [
        "delitos",
        "hotspot",
        "promedio",
        "historico",
        "lag",
        "rolling",
        "umbral",
        "cantidad_delitos",
        "fecha_mes",
        "grid_id",
        "mes",
    ]

    features = []
    for col in candidatas:
        if any(p in col.lower() for p in patrones_excluir):
            continue
        features.append(col)

    return features


def preparar_xy(df: pd.DataFrame):
    features = seleccionar_features_infraestructura(df)

    if not features:
        raise ValueError("No se seleccionó ninguna feature de infraestructura.")

    X = df[features].copy()
    y = df[TARGET_COL].astype(int).copy()

    X = pd.get_dummies(X, drop_first=True)
    X = X.select_dtypes(include=[np.number]).copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y, features


def obtener_modelos():
    modelos = {}

    modelos["Logistic Regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )

    modelos["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    try:
        from xgboost import XGBClassifier

        modelos["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    except ImportError:
        print("⚠️ XGBoost no está instalado. Para instalar:")
        print("   python -m pip install xgboost")

    return modelos


def calcular_metricas(y_true, y_pred, y_proba):
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_proba)
    else:
        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluar_modelos_thresholds(X, y):
    modelos = obtener_modelos()
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    resultados = []
    modelos_entrenados = {}

    for nombre_modelo, modelo in modelos.items():
        print("\n===================================================")
        print(f"ENTRENANDO MODELO: {nombre_modelo}")
        print("===================================================")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(
                f"Fold {fold}: "
                f"train={len(train_idx):,} | test={len(test_idx):,} | "
                f"hotspots train={y_train.mean()*100:.2f}% | "
                f"hotspots test={y_test.mean()*100:.2f}%"
            )

            modelo.fit(X_train, y_train)
            y_proba = modelo.predict_proba(X_test)[:, 1]

            for threshold in THRESHOLDS:
                y_pred = (y_proba >= threshold).astype(int)

                metricas = calcular_metricas(y_test, y_pred, y_proba)
                metricas["modelo"] = nombre_modelo
                metricas["fold"] = fold
                metricas["threshold"] = float(threshold)

                resultados.append(metricas)

        modelo.fit(X, y)
        modelos_entrenados[nombre_modelo] = modelo

    return pd.DataFrame(resultados), modelos_entrenados


def exportar_feature_importance(modelos_entrenados, feature_names):
    if "Random Forest" in modelos_entrenados:
        rf = modelos_entrenados["Random Forest"]

        fi_rf = pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)

        exportar_csv_seguro(fi_rf, OUTPUT_RF_IMPORTANCE)

    if "XGBoost" in modelos_entrenados:
        xgb = modelos_entrenados["XGBoost"]

        fi_xgb = pd.DataFrame({
            "feature": feature_names,
            "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False)

        exportar_csv_seguro(fi_xgb, OUTPUT_XGB_IMPORTANCE)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("EXPERIMENTO A - SOLO INFRAESTRUCTURA + OPTIMIZACIÓN THRESHOLD")
    print("=" * 75)

    print("\n📂 Cargando dataset...")
    df = cargar_dataset()

    print(f"Filas:              {len(df):,}")
    print(f"Columnas:           {len(df.columns):,}")
    print(f"Tasa de hotspots:   {df[TARGET_COL].mean()*100:.2f}%")

    print("\n🧹 Preparando X e y solo con infraestructura...")
    X, y, features_originales = preparar_xy(df)

    print(f"Features originales seleccionadas: {len(features_originales):,}")
    for f in features_originales:
        print(f" - {f}")

    print(f"\nFeatures finales después de dummies: {X.shape[1]:,}")
    print(f"Observaciones:                     {X.shape[0]:,}")
    print(f"Thresholds evaluados:              {len(THRESHOLDS)}")

    print("\n🤖 Entrenando modelos con TimeSeriesSplit...")
    resultados_folds, modelos_entrenados = evaluar_modelos_thresholds(X, y)

    resumen = (
        resultados_folds
        .groupby(["modelo", "threshold"])
        .agg({
            "accuracy": "mean",
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "roc_auc": "mean",
            "tn": "sum",
            "fp": "sum",
            "fn": "sum",
            "tp": "sum",
        })
        .reset_index()
        .sort_values(["modelo", "threshold"])
    )

    mejores = (
        resumen.sort_values("f1", ascending=False)
        .groupby("modelo", as_index=False)
        .head(1)
        .sort_values("f1", ascending=False)
    )

    print("\n📊 Mejores thresholds por modelo según F1:")
    print(mejores.to_string(index=False, float_format="%.4f"))

    exportar_csv_seguro(resumen, OUTPUT_THRESHOLDS)
    exportar_csv_seguro(mejores, OUTPUT_BEST)

    print("\n📌 Exportando feature importance...")
    exportar_feature_importance(modelos_entrenados, X.columns.tolist())

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados thresholds: outputs/{OUTPUT_THRESHOLDS.name}")
    print(f"Mejores thresholds:    outputs/{OUTPUT_BEST.name}")
    print(f"Feature Importance RF: outputs/{OUTPUT_RF_IMPORTANCE.name}")
    print(f"Feature Importance XGB: outputs/{OUTPUT_XGB_IMPORTANCE.name}")
    print("===================================================")


if __name__ == "__main__":
    main()
"""
Script: ML_05B_modelos_solo_historia.py

Descripción:
Experimento B: entrena modelos usando únicamente variables históricas delictivas.

Incluye:
    - delitos_mes_anterior_*
    - promedio_3_meses_*
    - promedio_6_meses_*
    - promedio_historico_*
    - delitos_historicos_*
    - hotspot_mes_anterior_*

Excluye:
    - variables de infraestructura urbana
    - variables de área
    - variables calendario simples
    - identificadores
    - target y cantidad_delitos actual

Entrada:
    outputs/dataset_ml_features_temporales.csv

Salidas:
    outputs/resultados_solo_historia_thresholds.csv
    outputs/mejor_threshold_solo_historia.csv
    outputs/feature_importance_rf_solo_historia.csv
    outputs/feature_importance_xgb_solo_historia.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_CSV = OUTPUT_DIR / "dataset_ml_features_temporales.csv"

OUTPUT_THRESHOLDS = OUTPUT_DIR / "resultados_solo_historia_thresholds.csv"
OUTPUT_BEST = OUTPUT_DIR / "mejor_threshold_solo_historia.csv"
OUTPUT_RF_IMPORTANCE = OUTPUT_DIR / "feature_importance_rf_solo_historia.csv"
OUTPUT_XGB_IMPORTANCE = OUTPUT_DIR / "feature_importance_xgb_solo_historia.csv"

TARGET_COL = "hotspot_exploratorio"

N_SPLITS = 5
RANDOM_STATE = 42

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


# ============================================================
# FUNCIONES
# ============================================================

def exportar_csv_seguro(df: pd.DataFrame, path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            raise PermissionError(
                f"No se puede sobrescribir {path}. Cerrá el archivo si está abierto."
            )

    df.to_csv(path, index=False)


def cargar_dataset() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No se encontró: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna target '{TARGET_COL}'.")

    if "fecha_mes" not in df.columns:
        raise ValueError("No se encontró la columna 'fecha_mes'.")

    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"], errors="coerce")
    df = df.dropna(subset=["fecha_mes"]).copy()
    df = df.sort_values("fecha_mes").reset_index(drop=True)

    return df


def seleccionar_features_historia(df: pd.DataFrame) -> list[str]:
    """
    Selecciona variables históricas delictivas.
    """

    prefijos_incluir = [
        "delitos_mes_anterior_",
        "promedio_3_meses_",
        "promedio_6_meses_",
        "promedio_historico_",
        "delitos_historicos_",
        "hotspot_mes_anterior_",
    ]

    columnas_excluir_exactas = {
        TARGET_COL,
        "cantidad_delitos",
        "umbral_p90_mes_franja",
        "grid_id",
        "mes",
        "fecha_mes",
        "franja",
        "anio",
        "mes_num",
        "trimestre",
        "semestre",
    }

    prefijos_excluir = [
        "cant_",
        "dist_min_",
        "area_",
    ]

    columnas_excluir_contienen = [
        "porcentaje_en_caba",
    ]

    features = []

    for col in df.columns:
        col_lower = col.lower()

        if col in columnas_excluir_exactas:
            continue

        if any(col_lower.startswith(p) for p in prefijos_excluir):
            continue

        if any(p in col_lower for p in columnas_excluir_contienen):
            continue

        if any(col_lower.startswith(p) for p in prefijos_incluir):
            features.append(col)

    return features


def preparar_xy(df: pd.DataFrame):
    features = seleccionar_features_historia(df)

    if not features:
        raise ValueError(
            "No se seleccionó ninguna feature histórica. "
            "Revisá los nombres de columnas del dataset."
        )

    X = df[features].copy()
    y = df[TARGET_COL].astype(int).copy()

    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y, features


def obtener_modelos():
    modelos = {}

    modelos["Logistic Regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )

    modelos["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    try:
        from xgboost import XGBClassifier

        modelos["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    except ImportError:
        print("⚠️ XGBoost no está instalado. Para instalar:")
        print("   python -m pip install xgboost")

    return modelos


def calcular_metricas(y_true, y_pred, y_proba):
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_proba)
    else:
        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluar_modelos_thresholds(X, y):
    modelos = obtener_modelos()
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    resultados = []
    modelos_entrenados = {}

    for nombre_modelo, modelo in modelos.items():
        print("\n===================================================")
        print(f"ENTRENANDO MODELO: {nombre_modelo}")
        print("===================================================")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(
                f"Fold {fold}: "
                f"train={len(train_idx):,} | test={len(test_idx):,} | "
                f"hotspots train={y_train.mean()*100:.2f}% | "
                f"hotspots test={y_test.mean()*100:.2f}%"
            )

            modelo.fit(X_train, y_train)

            y_proba = modelo.predict_proba(X_test)[:, 1]

            for threshold in THRESHOLDS:
                y_pred = (y_proba >= threshold).astype(int)

                metricas = calcular_metricas(y_test, y_pred, y_proba)
                metricas["modelo"] = nombre_modelo
                metricas["fold"] = fold
                metricas["threshold"] = float(threshold)

                resultados.append(metricas)

        modelo.fit(X, y)
        modelos_entrenados[nombre_modelo] = modelo

    return pd.DataFrame(resultados), modelos_entrenados


def exportar_feature_importance(modelos_entrenados, feature_names):
    if "Random Forest" in modelos_entrenados:
        rf = modelos_entrenados["Random Forest"]

        fi_rf = pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)

        exportar_csv_seguro(fi_rf, OUTPUT_RF_IMPORTANCE)

    if "XGBoost" in modelos_entrenados:
        xgb = modelos_entrenados["XGBoost"]

        fi_xgb = pd.DataFrame({
            "feature": feature_names,
            "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False)

        exportar_csv_seguro(fi_xgb, OUTPUT_XGB_IMPORTANCE)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("EXPERIMENTO B - SOLO HISTORIA DELICTIVA + OPTIMIZACIÓN THRESHOLD")
    print("=" * 75)

    print("\n📂 Cargando dataset...")
    df = cargar_dataset()

    print(f"Filas:              {len(df):,}")
    print(f"Columnas:           {len(df.columns):,}")
    print(f"Tasa de hotspots:   {df[TARGET_COL].mean()*100:.2f}%")

    print("\n🧹 Preparando X e y solo con historia delictiva...")
    X, y, features_originales = preparar_xy(df)

    print(f"Features históricas seleccionadas: {len(features_originales):,}")
    for f in features_originales:
        print(f" - {f}")

    print(f"\nFeatures finales:      {X.shape[1]:,}")
    print(f"Observaciones:         {X.shape[0]:,}")
    print(f"Thresholds evaluados:  {len(THRESHOLDS)}")

    print("\n🤖 Entrenando modelos con TimeSeriesSplit...")
    resultados_folds, modelos_entrenados = evaluar_modelos_thresholds(X, y)

    resumen = (
        resultados_folds
        .groupby(["modelo", "threshold"])
        .agg({
            "accuracy": "mean",
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "roc_auc": "mean",
            "tn": "sum",
            "fp": "sum",
            "fn": "sum",
            "tp": "sum",
        })
        .reset_index()
        .sort_values(["modelo", "threshold"])
    )

    mejores = (
        resumen.sort_values("f1", ascending=False)
        .groupby("modelo", as_index=False)
        .head(1)
        .sort_values("f1", ascending=False)
    )

    print("\n📊 Mejores thresholds por modelo según F1:")
    print(mejores.to_string(index=False, float_format="%.4f"))

    exportar_csv_seguro(resumen, OUTPUT_THRESHOLDS)
    exportar_csv_seguro(mejores, OUTPUT_BEST)

    print("\n📌 Exportando feature importance...")
    exportar_feature_importance(modelos_entrenados, X.columns.tolist())

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados thresholds: outputs/{OUTPUT_THRESHOLDS.name}")
    print(f"Mejores thresholds:    outputs/{OUTPUT_BEST.name}")
    print(f"Feature Importance RF: outputs/{OUTPUT_RF_IMPORTANCE.name}")
    print(f"Feature Importance XGB: outputs/{OUTPUT_XGB_IMPORTANCE.name}")
    print("===================================================")


if __name__ == "__main__":
    main()
"""
Script: ML_06_optimizacion_threshold.py

Descripción:
Optimiza el threshold de clasificación para el modelo XGBoost.

Objetivo:
Evaluar distintos umbrales de probabilidad para transformar las predicciones
probabilísticas del modelo en clases binarias:

    probabilidad >= threshold -> hotspot = 1
    probabilidad < threshold  -> hotspot = 0

Entrada:
    outputs/dataset_ml_features_temporales.csv

Salidas:
    outputs/resultados_threshold_xgboost.csv
    outputs/mejor_threshold_xgboost.csv

Métricas evaluadas:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - Matriz de confusión acumulada
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_CSV = OUTPUT_DIR / "dataset_ml_features_temporales.csv"

OUTPUT_THRESHOLDS = OUTPUT_DIR / "resultados_threshold_xgboost.csv"
OUTPUT_BEST = OUTPUT_DIR / "mejor_threshold_xgboost.csv"

TARGET_COL = "hotspot_exploratorio"

N_SPLITS = 5
RANDOM_STATE = 42

THRESHOLDS = np.arange(0.05, 0.81, 0.05)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def exportar_csv_seguro(df: pd.DataFrame, path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            raise PermissionError(
                f"No se puede sobrescribir {path}. Cerrá el archivo si está abierto."
            )

    df.to_csv(path, index=False)


def cargar_dataset() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No se encontró: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"No se encontró la columna target '{TARGET_COL}'. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    if "fecha_mes" not in df.columns:
        raise ValueError("No se encontró la columna 'fecha_mes'.")

    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"], errors="coerce")
    df = df.dropna(subset=["fecha_mes"]).copy()

    df = df.sort_values("fecha_mes").reset_index(drop=True)

    return df


def preparar_xy(df: pd.DataFrame):
    columnas_excluir = [
        TARGET_COL,
        "cantidad_delitos",
        "umbral_p90_mes_franja",
        "mes",
        "fecha_mes",
        "grid_id",
    ]

    columnas_excluir = [c for c in columnas_excluir if c in df.columns]

    X = df.drop(columns=columnas_excluir).copy()
    y = df[TARGET_COL].astype(int).copy()

    X = pd.get_dummies(X, drop_first=True)
    X = X.select_dtypes(include=[np.number]).copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y


def crear_modelo_xgboost(y_train: pd.Series):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError(
            "XGBoost no está instalado. Instalalo con:\n"
            "python -m pip install xgboost"
        )

    negativos = (y_train == 0).sum()
    positivos = (y_train == 1).sum()

    scale_pos_weight = negativos / positivos if positivos > 0 else 1

    modelo = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return modelo


def calcular_metricas(y_true, y_pred, y_proba):
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_proba)
    else:
        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 65)
    print("OPTIMIZACIÓN DE THRESHOLD - XGBOOST")
    print("=" * 65)

    print("\n📂 Cargando dataset...")
    df = cargar_dataset()

    print(f"  Filas:             {len(df):,}")
    print(f"  Columnas:          {len(df.columns):,}")
    print(f"  Tasa de hotspots:  {df[TARGET_COL].mean() * 100:.2f}%")

    print("\n🧹 Preparando X e y...")
    X, y = preparar_xy(df)

    print(f"  Features usadas:   {X.shape[1]:,}")
    print(f"  Observaciones:     {X.shape[0]:,}")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    resultados = []

    print("\n🤖 Entrenando XGBoost y evaluando thresholds...")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(
            f"\nFold {fold}: "
            f"train={len(train_idx):,} | test={len(test_idx):,} | "
            f"hotspots train={y_train.mean()*100:.2f}% | "
            f"hotspots test={y_test.mean()*100:.2f}%"
        )

        modelo = crear_modelo_xgboost(y_train)
        modelo.fit(X_train, y_train)

        y_proba = modelo.predict_proba(X_test)[:, 1]

        for threshold in THRESHOLDS:
            y_pred = (y_proba >= threshold).astype(int)

            metricas = calcular_metricas(y_test, y_pred, y_proba)

            metricas["fold"] = fold
            metricas["threshold"] = round(float(threshold), 2)

            resultados.append(metricas)

    resultados_df = pd.DataFrame(resultados)

    resumen = (
        resultados_df
        .groupby("threshold")
        .agg({
            "accuracy": "mean",
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "roc_auc": "mean",
            "tn": "sum",
            "fp": "sum",
            "fn": "sum",
            "tp": "sum",
        })
        .reset_index()
        .sort_values("threshold")
    )

    mejor_f1 = resumen.sort_values("f1", ascending=False).head(1).copy()
    mejor_recall_controlado = (
        resumen[resumen["precision"] >= 0.40]
        .sort_values("recall", ascending=False)
        .head(1)
        .copy()
    )

    print("\n📊 Resultados promedio por threshold:")
    print(resumen.to_string(index=False, float_format="%.4f"))

    print("\n🏆 Mejor threshold según F1:")
    print(mejor_f1.to_string(index=False, float_format="%.4f"))

    if not mejor_recall_controlado.empty:
        print("\n🎯 Mejor threshold maximizando recall con precision >= 0.40:")
        print(mejor_recall_controlado.to_string(index=False, float_format="%.4f"))

    exportar_csv_seguro(resumen, OUTPUT_THRESHOLDS)
    exportar_csv_seguro(mejor_f1, OUTPUT_BEST)

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados thresholds: outputs/{OUTPUT_THRESHOLDS.name}")
    print(f"Mejor threshold:       outputs/{OUTPUT_BEST.name}")
    print("===================================================")


if __name__ == "__main__":
    main()
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
