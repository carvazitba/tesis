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