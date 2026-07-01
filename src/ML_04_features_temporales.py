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