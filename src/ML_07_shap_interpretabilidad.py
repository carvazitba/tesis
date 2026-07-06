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

# Predicciones out-of-fold de ML_06 (para seleccionar casos individuales
# a explicar con waterfall plots)
INPUT_PREDICCIONES = OUTPUT_DIR / "predicciones_mejor_threshold_xgboost.csv"

# Grilla espacial con geometrías (para el mapa de SHAP). AJUSTAR el nombre
# al archivo real del repositorio. Si no existe, el mapa se omite con aviso.
INPUT_GRILLA = OUTPUT_DIR / "grilla_caba_250m.geojson"

# Agrupación conceptual de variables (para el gráfico de importancia
# agrupada). Toda variable no matcheada cae en "Geométricas".
GRUPOS_VARIABLES = {
    "Historial delictivo": ["promedio", "delitos", "hotspot_mes"],
    "Infraestructura urbana": ["cant_", "dist_min"],
    "Temporales": ["anio", "mes_num", "trimestre", "semestre"],
}

MODELOS = {
    "xgboost": OUTPUT_DIR / "ML_05_xgboost.joblib",
    "random_forest": OUTPUT_DIR / "ML_05_random_forest.joblib",
    "logistic_regression": OUTPUT_DIR / "ML_05_logistic_regression.pickle",
}

TARGET_COL = "hotspot_exploratorio"

# Tamaño de muestra para calcular SHAP values (XGBoost y Logistic
# Regression). El modelo explicado es el entrenado con TODO el dataset;
# la muestra solo define sobre cuántas observaciones se calculan las
# explicaciones. Con 20.000-50.000 el ranking global ya es estable.
N_SAMPLE = 50_000

# Random Forest con max_depth=None sobre 1.3M filas genera árboles
# enormes: el TreeExplainer exacto es inviable (días de cómputo).
# Se usa una submuestra chica + método aproximado (Saabas).
# NO subir este valor sin necesidad: fue la causa de la corrida de un día.
N_SAMPLE_RF = 3_000

# Modelos a saltear (si ya tenés sus outputs de una corrida previa,
# agregalos acá para no recalcularlos, ej.: ["xgboost"])
OMITIR_MODELOS = []

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

def shap_values_arbol(modelo, X_sample: pd.DataFrame,
                      aproximado: bool = False) -> np.ndarray:
    """
    TreeExplainer para XGBoost y Random Forest.
    Devuelve matriz (n_filas, n_features) para la clase positiva.

    aproximado=True usa el método de Saabas (rápido, adecuado para
    árboles muy profundos como el RF con max_depth=None). El ranking
    de importancia global es prácticamente idéntico al exacto.
    """
    explainer = shap.TreeExplainer(modelo)
    sv = explainer.shap_values(
        X_sample,
        approximate=aproximado,
        check_additivity=False,
    )

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
# ANÁLISIS EXTENDIDO (solo XGBoost, el modelo final)
# ============================================================

def clasificar_grupo(feature: str) -> str:
    for grupo, claves in GRUPOS_VARIABLES.items():
        if any(k in feature for k in claves):
            return grupo
    return "Geométricas"


def plot_importancia_agrupada(sv: np.ndarray, X_sample: pd.DataFrame,
                              nombre: str):
    """
    (3) Barras de importancia agregada por grupo conceptual de variables.
    Es la versión gráfica de la tabla de contribuciones de la tesis.
    """
    imp = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_abs": np.abs(sv).mean(axis=0),
    })
    imp["grupo"] = imp["feature"].map(clasificar_grupo)

    agg = imp.groupby("grupo")["shap_abs"].sum().sort_values()
    pct = agg / agg.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    barras = ax.barh(agg.index, agg.values, color="#1f77b4")
    for barra, p_ in zip(barras, pct.values):
        ax.text(barra.get_width() + agg.max() * 0.01,
                barra.get_y() + barra.get_height() / 2,
                f"{p_:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("|SHAP| medio agregado")
    ax.set_title(f"Contribución por grupo de variables — {nombre}")
    ax.set_xlim(0, agg.max() * 1.15)
    plt.tight_layout()

    ruta = SHAP_DIR / f"shap_importancia_agrupada_{nombre}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {ruta.name}")

    # CSV de respaldo para la tabla de la tesis
    salida = pd.DataFrame({
        "grupo": agg.index[::-1],
        "shap_abs_agregado": agg.values[::-1],
        "porcentaje": pct.values[::-1],
    })
    ruta_csv = SHAP_DIR / f"shap_importancia_agrupada_{nombre}.csv"
    salida.to_csv(ruta_csv, index=False)
    print(f"   ✓ {ruta_csv.name}")


def plot_dependence_infraestructura(sv: np.ndarray, X_sample: pd.DataFrame,
                                    nombre: str, n_top: int = 4):
    """
    (1) Panel 2x2 con dependence plots de las variables de infraestructura
    más importantes. Muestra la FORMA del efecto (ej.: radio crítico de
    distancia a partir del cual la variable deja de aportar riesgo).
    """
    imp = np.abs(sv).mean(axis=0)
    es_infra = [
        clasificar_grupo(f) == "Infraestructura urbana"
        for f in X_sample.columns
    ]
    idx_infra = [i for i, ok in enumerate(es_infra) if ok]
    idx_infra = sorted(idx_infra, key=lambda i: imp[i], reverse=True)[:n_top]

    if not idx_infra:
        print("   ⚠️ No se identificaron variables de infraestructura.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax_, idx in zip(axes, idx_infra):
        shap.dependence_plot(
            idx, sv, X_sample,
            interaction_index=None,
            show=False, ax=ax_,
        )
        ax_.set_title(X_sample.columns[idx], fontsize=11)

    # Ocultar ejes sobrantes si hay menos de 4 variables
    for ax_ in axes[len(idx_infra):]:
        ax_.set_visible(False)

    fig.suptitle(f"SHAP dependence — infraestructura urbana — {nombre}",
                 fontsize=13)
    plt.tight_layout()

    ruta = SHAP_DIR / f"shap_dependence_infraestructura_{nombre}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {ruta.name}")


def plot_dependence_anio(sv: np.ndarray, X_sample: pd.DataFrame, nombre: str):
    """
    (4) Dependence plot de 'anio' para verificar el quiebre temporal
    (hipótesis: disrupción por pandemia en 2020) antes de afirmarlo
    en la tesis.
    """
    if "anio" not in X_sample.columns:
        print("   ⚠️ No existe la variable 'anio'.")
        return

    idx = X_sample.columns.get_loc("anio")
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(idx, sv, X_sample,
                         interaction_index=None, show=False)
    plt.title(f"SHAP dependence — {nombre} — anio", fontsize=12)
    plt.tight_layout()

    ruta = SHAP_DIR / f"shap_dependence_{nombre}_anio.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {ruta.name}")


def plot_waterfalls(modelo, X: pd.DataFrame, df: pd.DataFrame, nombre: str):
    """
    (2) Waterfall plots de tres casos individuales, seleccionados con las
    predicciones out-of-fold de ML_06:
      - verdadero positivo típico (proba más alta entre los aciertos)
      - falso negativo "near miss" (hotspot real con proba justo bajo el umbral)
      - celda SIN historial delictivo con mayor proba (muestra a la
        infraestructura empujando la predicción sin ayuda del historial)

    Nota: las probabilidades provienen de los modelos por fold de ML_06,
    pero la explicación se calcula sobre el modelo final (consistente con
    el resto de la sección). Los casos se usan como ilustración cualitativa.
    """
    if not INPUT_PREDICCIONES.exists():
        print(f"   ⚠️ No se encontró {INPUT_PREDICCIONES.name}; "
              f"se omiten los waterfalls. Ejecutá ML_06 primero.")
        return

    pred = pd.read_csv(INPUT_PREDICCIONES)
    pred["fecha_mes"] = pd.to_datetime(pred["fecha_mes"], errors="coerce")

    # Mapear cada predicción a su fila del dataset via identificadores
    df_id = df.reset_index()[["index", "fecha_mes", "grid_id", "franja"]]
    pred = pred.merge(df_id, on=["fecha_mes", "grid_id", "franja"],
                      how="inner")

    if "promedio_historico_grid" in df.columns:
        pred["sin_historial"] = (
            df.loc[pred["index"], "promedio_historico_grid"].values == 0
        )
    else:
        pred["sin_historial"] = False

    casos = {}

    tp = pred[(pred["y_true"] == 1) & (pred["y_pred"] == 1)]
    if not tp.empty:
        casos["verdadero_positivo"] = tp.sort_values(
            "y_proba", ascending=False).iloc[0]

    fn = pred[(pred["y_true"] == 1) & (pred["y_pred"] == 0)]
    if not fn.empty:
        casos["falso_negativo"] = fn.sort_values(
            "y_proba", ascending=False).iloc[0]

    sh = pred[pred["sin_historial"]]
    if not sh.empty:
        casos["sin_historial"] = sh.sort_values(
            "y_proba", ascending=False).iloc[0]

    if not casos:
        print("   ⚠️ No se pudieron seleccionar casos para waterfall.")
        return

    explainer = shap.TreeExplainer(modelo)

    for etiqueta, caso in casos.items():
        idx = int(caso["index"])
        x_row = X.loc[[idx]]

        explicacion = explainer(x_row)

        plt.figure()
        shap.plots.waterfall(explicacion[0], max_display=12, show=False)
        fecha = pd.Timestamp(caso["fecha_mes"]).strftime("%Y-%m")
        plt.title(
            f"{etiqueta.replace('_', ' ')} — grid {int(caso['grid_id'])}, "
            f"{caso['franja']}, {fecha} "
            f"(proba OOF={caso['y_proba']:.2f}, real={int(caso['y_true'])})",
            fontsize=10,
        )
        plt.tight_layout()

        ruta = SHAP_DIR / f"shap_waterfall_{nombre}_{etiqueta}.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✓ {ruta.name}")


def mapa_shap_infraestructura(sv: np.ndarray, X_sample: pd.DataFrame,
                              df: pd.DataFrame, nombre: str):
    """
    (5) Mapa coroplético de CABA: contribución SHAP agregada de las
    variables de infraestructura por celda de la grilla. Muestra DÓNDE
    la infraestructura aporta más señal predictiva.

    Requiere geopandas y el archivo de grilla con geometrías
    (INPUT_GRILLA, columna 'grid_id'). Si falta, se omite con aviso.
    """
    if not INPUT_GRILLA.exists():
        print(f"   ⚠️ No se encontró {INPUT_GRILLA.name}; se omite el mapa.")
        print(f"      Ajustá INPUT_GRILLA al archivo de grilla real.")
        return

    try:
        import geopandas as gpd
    except ImportError:
        print("   ⚠️ geopandas no está instalado; se omite el mapa.")
        return

    es_infra = np.array([
        clasificar_grupo(f) == "Infraestructura urbana"
        for f in X_sample.columns
    ])

    shap_infra = pd.DataFrame({
        "grid_id": df.loc[X_sample.index, "grid_id"].values,
        "shap_infra_abs": np.abs(sv[:, es_infra]).sum(axis=1),
    })
    agg = shap_infra.groupby("grid_id")["shap_infra_abs"].mean().reset_index()

    grilla = gpd.read_file(INPUT_GRILLA)
    if "grid_id" not in grilla.columns:
        print(f"   ⚠️ La grilla no tiene columna 'grid_id'; se omite el mapa.")
        return

    grilla = grilla.merge(agg, on="grid_id", how="left")

    fig, ax = plt.subplots(figsize=(10, 10))
    grilla.plot(
        column="shap_infra_abs", ax=ax, cmap="viridis",
        legend=True, missing_kwds={"color": "#eeeeee"},
        legend_kwds={"label": "|SHAP| medio (infraestructura)",
                     "shrink": 0.6},
    )
    ax.set_title(f"Contribución SHAP de la infraestructura urbana "
                 f"por celda — {nombre}", fontsize=13)
    ax.set_axis_off()
    plt.tight_layout()

    ruta = SHAP_DIR / f"shap_mapa_infraestructura_{nombre}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {ruta.name}")




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

        if nombre in OMITIR_MODELOS:
            print(f"   ⏭️  Omitido (está en OMITIR_MODELOS).")
            continue

        if not ruta.exists():
            print(f"   ⚠️ No se encontró {ruta.name}, se omite.")
            continue

        modelo = cargar_modelo(ruta)

        print("📊 Calculando SHAP values...")
        if nombre == "logistic_regression":
            sv = shap_values_lineal(modelo, X_sample)
            X_plots = X_sample
        elif nombre == "random_forest":
            # Submuestra + método aproximado: el RF (max_depth=None,
            # 300 árboles sobre 1.3M filas) tiene árboles enormes y
            # el TreeExplainer exacto tardaría días.
            X_rf = X_sample.sample(
                n=min(N_SAMPLE_RF, len(X_sample)),
                random_state=RANDOM_STATE,
            )
            print(f"   (submuestra de {len(X_rf):,} filas, "
                  f"método aproximado)")
            sv = shap_values_arbol(modelo, X_rf, aproximado=True)
            X_plots = X_rf
        else:
            sv = shap_values_arbol(modelo, X_sample)
            X_plots = X_sample

        print("📈 Generando visualizaciones...")
        plot_beeswarm(sv, X_plots, nombre)
        plot_bar(sv, X_plots, nombre)
        plot_dependence(sv, X_plots, nombre)

        print("💾 Exportando ranking de importancia...")
        df_imp = exportar_importancia(sv, X_plots, nombre)

        # ---- Análisis extendido: solo para el modelo final (XGBoost) ----
        if nombre == "xgboost":
            print("\n📊 Análisis extendido (modelo final)...")
            plot_importancia_agrupada(sv, X_plots, nombre)
            plot_dependence_infraestructura(sv, X_plots, nombre)
            plot_dependence_anio(sv, X_plots, nombre)
            plot_waterfalls(modelo, X, df, nombre)
            mapa_shap_infraestructura(sv, X_plots, df, nombre)

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