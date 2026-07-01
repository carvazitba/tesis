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