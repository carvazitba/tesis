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
