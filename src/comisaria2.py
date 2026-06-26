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