"""
Script: ML_07_shap_interpretabilidad.py

Genera explicaciones SHAP para los modelos entrenados en ML_05.

Modelos analizados:
    - XGBoost (ML_05_xgboost.joblib)          -> TreeExplainer
    - Random Forest (ML_05_random_forest.joblib) -> TreeExplainer
    - Logistic Regression (ML_05_logistic_regression.pickle) -> LinearExplainer

Entrada:
    outputs/dataset_ml_features_temporales.csv
    outputs/feature_names_ml05.csv
    outputs/ML_05_xgboost.joblib
    outputs/ML_05_random_forest.joblib
    outputs/ML_05_logistic_regression.pickle

Salidas (en outputs/shap/):
    shap_summary_beeswarm_<modelo>.png   (dirección + magnitud por feature)
    shap_summary_bar_<modelo>.png        (importancia global |SHAP| medio)
    shap_dependence_<modelo>_<feature>.png (top 4 features)
    shap_importance_<modelo>.csv         (ranking numérico de importancia)

NOTA METODOLÓGICA:
    - Los modelos explicados son los reentrenados con el dataset completo
      (modelo final de ML_05), no los modelos de cada fold. Las métricas
      reportadas en la tesis provienen de la validación temporal; las
      explicaciones SHAP corresponden al modelo final.
    - Para XGBoost y Random Forest, TreeExplainer devuelve valores en
      espacio log-odds (margin). Esto no afecta el ranking de importancia,
      pero las magnitudes no son probabilidades.
    - Por el tamaño del dataset (~1.3M filas), los SHAP values se calculan
      sobre una muestra aleatoria estratificada. El ranking de importancia
      es estable con muestras de este tamaño.
"""

from pathlib import Path
import warnings
import pickle

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")  # sin ventana; solo guarda PNG
import matplotlib.pyplot as plt

import shap

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
SHAP_DIR = OUTPUT_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = OUTPUT_DIR / "dataset_ml_features_temporales.csv"
INPUT_FEATURE_NAMES = OUTPUT_DIR / "feature_names_ml05.csv"

MODELOS = {
    "xgboost": OUTPUT_DIR / "ML_05_xgboost.joblib",
    "random_forest": OUTPUT_DIR / "ML_05_random_forest.joblib",
    "logistic_regression": OUTPUT_DIR / "ML_05_logistic_regression.pickle",
}

TARGET_COL = "hotspot_exploratorio"

# Tamaño de muestra para calcular SHAP values.
# 20.000 filas es suficiente para importancia global estable
# y mantiene el tiempo de cómputo razonable (sobre todo para RF).
N_SAMPLE = 20_000
N_TOP_DEPENDENCE = 4
RANDOM_STATE = 42


# ============================================================
# PREPARACIÓN DE DATOS (idéntica a ML_05)
# ============================================================

def cargar_dataset() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No se encontró: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"], errors="coerce")
    df = df.dropna(subset=["fecha_mes"]).copy()

    # Mismo orden que ML_05 para reproducibilidad
    df = df.sort_values(["fecha_mes", "grid_id", "franja"]).reset_index(drop=True)

    return df


def preparar_xy(df: pd.DataFrame):
    """Misma preparación que ML_05 (mantener sincronizado)."""
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


def verificar_features(X: pd.DataFrame):
    """
    Verifica que las columnas de X coincidan exactamente (nombre y orden)
    con las guardadas por ML_05. Si no coinciden, los SHAP values
    quedarían asignados a features equivocadas.
    """
    if not INPUT_FEATURE_NAMES.exists():
        raise FileNotFoundError(
            f"No se encontró {INPUT_FEATURE_NAMES}. "
            f"Ejecutá primero ML_05_modelos_hotspots.py."
        )

    esperadas = pd.read_csv(INPUT_FEATURE_NAMES)["feature"].tolist()
    actuales = X.columns.tolist()

    if actuales != esperadas:
        solo_actuales = set(actuales) - set(esperadas)
        solo_esperadas = set(esperadas) - set(actuales)
        raise ValueError(
            "Las features no coinciden con feature_names_ml05.csv.\n"
            f"  Solo en dataset actual: {sorted(solo_actuales)}\n"
            f"  Solo en ML_05:          {sorted(solo_esperadas)}\n"
            "Verificá que el dataset de entrada sea el mismo que usó ML_05."
        )

    print(f"   ✓ Features verificadas: {len(actuales)} columnas, mismo orden que ML_05")


def muestrear(X: pd.DataFrame, y: pd.Series, n: int, seed: int):
    """
    Muestra aleatoria estratificada por target para calcular SHAP.
    La estratificación asegura presencia proporcional de hotspots.
    """
    if len(X) <= n:
        return X, y

    idx = (
        pd.DataFrame({"y": y})
        .groupby("y", group_keys=False)
        .apply(lambda g: g.sample(frac=n / len(X), random_state=seed))
        .index
    )

    return X.loc[idx], y.loc[idx]


# ============================================================
# CARGA DE MODELOS
# ============================================================

def cargar_modelo(ruta: Path):
    if ruta.suffix == ".joblib":
        return joblib.load(ruta)
    with open(ruta, "rb") as f:
        return pickle.load(f)


# ============================================================
# CÁLCULO SHAP POR MODELO
# ============================================================

def shap_values_arbol(modelo, X_sample: pd.DataFrame) -> np.ndarray:
    """
    TreeExplainer para XGBoost y Random Forest.
    Devuelve matriz (n_filas, n_features) para la clase positiva.
    """
    explainer = shap.TreeExplainer(modelo)
    sv = explainer.shap_values(X_sample)

    # RandomForestClassifier de sklearn devuelve lista [clase0, clase1]
    # o array 3D según versión de shap; normalizamos a clase 1.
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]

    return sv


def shap_values_lineal(pipeline, X_sample: pd.DataFrame) -> np.ndarray:
    """
    LinearExplainer para la Regresión Logística (Pipeline scaler + model).
    Se aplica el scaler manualmente y se explica el modelo lineal
    sobre los datos escalados. Mucho más rápido y exacto que KernelExplainer.
    """
    scaler = pipeline.named_steps["scaler"]
    modelo = pipeline.named_steps["model"]

    X_scaled = pd.DataFrame(
        scaler.transform(X_sample),
        columns=X_sample.columns,
        index=X_sample.index,
    )

    explainer = shap.LinearExplainer(modelo, X_scaled)
    sv = explainer.shap_values(X_scaled)

    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]

    return sv


# ============================================================
# VISUALIZACIONES Y EXPORTS
# ============================================================

def plot_beeswarm(sv: np.ndarray, X_sample: pd.DataFrame, nombre: str):
    plt.figure()
    shap.summary_plot(sv, X_sample, show=False, max_display=15)
    plt.title(f"SHAP summary — {nombre}", fontsize=13)
    plt.tight_layout()
    ruta = SHAP_DIR / f"shap_summary_beeswarm_{nombre}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {ruta.name}")


def plot_bar(sv: np.ndarray, X_sample: pd.DataFrame, nombre: str):
    plt.figure()
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False, max_display=15)
    plt.title(f"Importancia global |SHAP| — {nombre}", fontsize=13)
    plt.tight_layout()
    ruta = SHAP_DIR / f"shap_summary_bar_{nombre}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {ruta.name}")


def plot_dependence(sv: np.ndarray, X_sample: pd.DataFrame, nombre: str,
                    n_top: int = N_TOP_DEPENDENCE):
    importancia = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(importancia)[::-1][:n_top]

    for idx in top_idx:
        feature = X_sample.columns[idx]
        plt.figure()
        shap.dependence_plot(
            idx, sv, X_sample,
            interaction_index=None,
            show=False,
        )
        plt.title(f"SHAP dependence — {nombre} — {feature}", fontsize=12)
        plt.tight_layout()

        nombre_archivo = feature.replace("/", "_").replace(" ", "_")
        ruta = SHAP_DIR / f"shap_dependence_{nombre}_{nombre_archivo}.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✓ {ruta.name}")


def exportar_importancia(sv: np.ndarray, X_sample: pd.DataFrame, nombre: str):
    """
    Exporta ranking numérico de importancia SHAP:
      - shap_importance_mean_abs: |SHAP| medio (magnitud del efecto)
      - shap_mean: SHAP medio con signo (dirección predominante)
    """
    df_imp = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_importance_mean_abs": np.abs(sv).mean(axis=0),
        "shap_mean": sv.mean(axis=0),
    }).sort_values("shap_importance_mean_abs", ascending=False)

    ruta = SHAP_DIR / f"shap_importance_{nombre}.csv"
    df_imp.to_csv(ruta, index=False)
    print(f"   ✓ {ruta.name}")

    return df_imp


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("INTERPRETABILIDAD SHAP — MODELOS ML_05")
    print("=" * 60)

    print("\n📂 Cargando dataset...")
    df = cargar_dataset()
    print(f"Filas:    {len(df):,}")

    print("\n🧹 Preparando X e y (misma lógica que ML_05)...")
    X, y = preparar_xy(df)
    verificar_features(X)

    print(f"\n🎲 Muestreando {N_SAMPLE:,} filas (estratificado por target)...")
    X_sample, y_sample = muestrear(X, y, N_SAMPLE, RANDOM_STATE)
    print(f"   ✓ Muestra: {len(X_sample):,} filas | "
          f"hotspots: {y_sample.mean()*100:.2f}%")

    for nombre, ruta in MODELOS.items():
        print("\n" + "=" * 60)
        print(f"MODELO: {nombre}")
        print("=" * 60)

        if not ruta.exists():
            print(f"   ⚠️ No se encontró {ruta.name}, se omite.")
            continue

        modelo = cargar_modelo(ruta)

        print("📊 Calculando SHAP values...")
        if nombre == "logistic_regression":
            sv = shap_values_lineal(modelo, X_sample)
        else:
            sv = shap_values_arbol(modelo, X_sample)

        print("📈 Generando visualizaciones...")
        plot_beeswarm(sv, X_sample, nombre)
        plot_bar(sv, X_sample, nombre)
        plot_dependence(sv, X_sample, nombre)

        print("💾 Exportando ranking de importancia...")
        df_imp = exportar_importancia(sv, X_sample, nombre)

        print(f"\n   Top 10 features ({nombre}):")
        for _, row in df_imp.head(10).iterrows():
            signo = "+" if row["shap_mean"] >= 0 else "-"
            print(f"     {row['feature']:<40} "
                  f"|SHAP|={row['shap_importance_mean_abs']:.4f} ({signo})")

    print("\n" + "=" * 60)
    print("✅ ANÁLISIS SHAP COMPLETADO")
    print("=" * 60)
    print(f"Salidas en: {SHAP_DIR}")
    print("\n📌 Para la tesis:")
    print("  - beeswarm: dirección y magnitud del efecto de cada variable")
    print("  - bar: ranking de importancia global")
    print("  - dependence: forma de la relación variable → predicción")
    print("  - CSV: valores numéricos para tablas")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()