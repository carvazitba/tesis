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