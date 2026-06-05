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