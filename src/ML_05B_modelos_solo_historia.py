"""
Script: ML_05B_modelos_solo_historia.py

Descripción:
Experimento de ablación B: entrena modelos usando ÚNICAMENTE variables
históricas delictivas, excluyendo infraestructura, geometría y calendario.

Formulación que este experimento pone a prueba:
    ¿Cuánto discrimina el historial delictivo por sí solo?

Junto con ML_05A (solo infraestructura), permite descomponer el ROC-AUC del
modelo completo (ML_05) en el aporte de cada grupo de variables. La métrica
central de comparación es el ROC-AUC, independiente del threshold.

Incluye:
    - delitos_mes_anterior_*
    - promedio_3_meses_*
    - promedio_6_meses_*
    - promedio_historico_*
    - delitos_historicos_*
    - hotspot_mes_anterior_*

Excluye:
    - variables de infraestructura urbana (cant_*, dist_min_*)
    - variables geométricas (area_*, porcentaje_en_caba)
    - variables calendario (anio, mes_num, trimestre, semestre, franja)
    - identificadores, target y cantidad_delitos actual

Entrada:
    outputs/dataset_ml_features_temporales.csv

Salidas:
    outputs/resultados_solo_historia_thresholds.csv
    outputs/mejor_threshold_solo_historia.csv
    outputs/resumen_roc_auc_solo_historia.csv
    outputs/feature_importance_rf_solo_historia.csv
    outputs/feature_importance_xgb_solo_historia.csv
    outputs/shap_ML05B/ (análisis SHAP del XGBoost final)

NOTA METODOLÓGICA:
    - La validación usa splits temporales por MES COMPLETO (idéntica a
      ML_05, ML_06 y ML_05A): los ROC-AUC son directamente comparables.
    - XGBoost compensa el desbalance con scale_pos_weight por fold,
      igual que en los demás experimentos.
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

try:
    import matplotlib
    matplotlib.use("Agg")  # sin ventana; solo guarda PNG
    import matplotlib.pyplot as plt
    import shap
    SHAP_DISPONIBLE = True
except ImportError:
    SHAP_DISPONIBLE = False


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
SHAP_DIR = OUTPUT_DIR / "shap_ML05B"

INPUT_CSV = OUTPUT_DIR / "dataset_ml_features_temporales.csv"

OUTPUT_THRESHOLDS = OUTPUT_DIR / "resultados_solo_historia_thresholds.csv"
OUTPUT_BEST = OUTPUT_DIR / "mejor_threshold_solo_historia.csv"
OUTPUT_RESUMEN_AUC = OUTPUT_DIR / "resumen_roc_auc_solo_historia.csv"
OUTPUT_RF_IMPORTANCE = OUTPUT_DIR / "feature_importance_rf_solo_historia.csv"
OUTPUT_XGB_IMPORTANCE = OUTPUT_DIR / "feature_importance_xgb_solo_historia.csv"

TARGET_COL = "hotspot_exploratorio"

N_SPLITS = 5
RANDOM_STATE = 42

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

# Muestra para el análisis SHAP (solo XGBoost, el modelo de comparación)
N_SAMPLE_SHAP = 50_000
N_TOP_DEPENDENCE = 4

# Prefijos de variables históricas delictivas a INCLUIR
PREFIJOS_INCLUIR = [
    "delitos_mes_anterior_",
    "promedio_3_meses_",
    "promedio_6_meses_",
    "promedio_historico_",
    "delitos_historicos_",
    "hotspot_mes_anterior_",
]

# Cualquier feature que contenga alguno de estos patrones queda EXCLUIDA
# (defensa en profundidad contra fugas de infraestructura, geometría o
# identificadores). NO incluir términos de calendario como "trimestre" o
# "mes": aparecen legítimamente dentro de nombres de variables históricas
# (promedio_historico_trimestre_grid, delitos_mes_anterior_grid) y las
# variables de calendario puras no pueden colarse porque no matchean
# ningún prefijo de PREFIJOS_INCLUIR.
PATRONES_PROHIBIDOS = [
    "cant_", "dist_min_", "area_", "porcentaje",
    "fecha", "grid_id", "umbral", "cantidad_delitos",
]


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

    # Ordenar por temporal + espacial para reproducibilidad (igual que ML_05)
    df = df.sort_values(["fecha_mes", "grid_id", "franja"]).reset_index(drop=True)

    return df


def seleccionar_features_historia(df: pd.DataFrame) -> list:
    """
    Selecciona únicamente variables históricas delictivas (por prefijo).
    El target y cantidad_delitos actual nunca matchean los prefijos, pero
    los patrones prohibidos actúan como verificación adicional contra
    cualquier fuga de infraestructura, geometría o calendario.
    """
    candidatas = [
        col for col in df.columns
        if any(col.lower().startswith(p) for p in PREFIJOS_INCLUIR)
    ]

    features = [
        col for col in candidatas
        if not any(p in col.lower() for p in PATRONES_PROHIBIDOS)
    ]

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

    # labels=[0, 1] garantiza matriz 2x2 aunque un fold tenga una sola clase
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

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


def generar_splits_temporales_por_mes(df: pd.DataFrame, n_splits: int = 5):
    """
    Genera splits temporales cortando por meses COMPLETOS (fecha_mes),
    no por filas. Idéntica a ML_05, ML_06 y ML_05A: los resultados son
    directamente comparables entre experimentos.
    """
    meses = np.array(sorted(df["fecha_mes"].unique()))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_mes_idx, test_mes_idx in tscv.split(meses):
        meses_train = meses[train_mes_idx]
        meses_test = meses[test_mes_idx]

        train_idx = df.index[df["fecha_mes"].isin(meses_train)].to_numpy()
        test_idx = df.index[df["fecha_mes"].isin(meses_test)].to_numpy()

        yield train_idx, test_idx


def evaluar_modelos_thresholds(X, y, df):
    modelos = obtener_modelos()

    resultados = []
    modelos_entrenados = {}

    for nombre_modelo, modelo in modelos.items():
        print("\n===================================================")
        print(f"ENTRENANDO MODELO: {nombre_modelo}")
        print("===================================================")

        for fold, (train_idx, test_idx) in enumerate(
            generar_splits_temporales_por_mes(df, N_SPLITS), start=1
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(
                f"Fold {fold}: "
                f"train={len(train_idx):,} | test={len(test_idx):,} | "
                f"hotspots train={y_train.mean()*100:.2f}% | "
                f"hotspots test={y_test.mean()*100:.2f}%"
            )

            # Compensar desbalance en XGBoost igual que en ML_05
            if nombre_modelo == "XGBoost":
                positivos = y_train.sum()
                negativos = len(y_train) - positivos

                if positivos > 0:
                    scale_weight = negativos / positivos
                    modelo.set_params(scale_pos_weight=scale_weight)
                    print(f"         scale_pos_weight={scale_weight:.2f}")

            modelo.fit(X_train, y_train)
            y_proba = modelo.predict_proba(X_test)[:, 1]

            for threshold in THRESHOLDS:
                y_pred = (y_proba >= threshold).astype(int)

                metricas = calcular_metricas(y_test, y_pred, y_proba)
                metricas["modelo"] = nombre_modelo
                metricas["fold"] = fold
                metricas["threshold"] = float(threshold)

                resultados.append(metricas)

        # Entrenar modelo final con todo el dataset
        # (scale_pos_weight también en el modelo final)
        if nombre_modelo == "XGBoost":
            positivos = y.sum()
            negativos = len(y) - positivos

            if positivos > 0:
                modelo.set_params(scale_pos_weight=negativos / positivos)

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
# ANÁLISIS SHAP (solo XGBoost, el modelo de comparación)
# ============================================================

def muestrear_estratificado(X: pd.DataFrame, y: pd.Series, n: int, seed: int):
    """Muestra aleatoria estratificada por target."""
    if len(X) <= n:
        return X

    idx = (
        pd.DataFrame({"y": y})
        .groupby("y", group_keys=False)
        .apply(lambda g: g.sample(frac=n / len(X), random_state=seed))
        .index
    )
    return X.loc[idx]


def analisis_shap_xgboost(modelo, X: pd.DataFrame, y: pd.Series):
    """
    Genera el análisis SHAP del modelo XGBoost solo-historia.

    Salidas (en outputs/shap_ML05B/):
        shap_summary_bar_solo_historia.png
        shap_summary_beeswarm_solo_historia.png
        shap_dependence_solo_historia_<feature>.png (top 4)
        shap_importance_solo_historia.csv
    """
    if not SHAP_DISPONIBLE:
        print("\n⚠️ shap y/o matplotlib no están instalados; se omite el "
              "análisis SHAP.")
        print("   Instalá con: python -m pip install shap matplotlib")
        return

    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    print("\n📊 Análisis SHAP del modelo solo-historia (XGBoost)...")

    X_sample = muestrear_estratificado(X, y, N_SAMPLE_SHAP, RANDOM_STATE)
    print(f"   Muestra: {len(X_sample):,} filas (estratificada por target)")

    explainer = shap.TreeExplainer(modelo)
    sv = explainer.shap_values(X_sample)

    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]

    # --- Summary bar ---
    plt.figure()
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False, max_display=15)
    plt.title("Importancia global |SHAP| — solo historia", fontsize=13)
    plt.tight_layout()
    ruta = SHAP_DIR / "shap_summary_bar_solo_historia.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {ruta.name}")

    # --- Beeswarm ---
    plt.figure()
    shap.summary_plot(sv, X_sample, show=False, max_display=15)
    plt.title("SHAP summary — solo historia", fontsize=13)
    plt.tight_layout()
    ruta = SHAP_DIR / "shap_summary_beeswarm_solo_historia.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {ruta.name}")

    # --- Dependence plots (top N) ---
    importancia = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(importancia)[::-1][:N_TOP_DEPENDENCE]

    for idx in top_idx:
        feature = X_sample.columns[idx]
        plt.figure()
        shap.dependence_plot(idx, sv, X_sample,
                             interaction_index=None, show=False)
        plt.title(f"SHAP dependence — solo historia — {feature}",
                  fontsize=12)
        plt.tight_layout()

        nombre_archivo = feature.replace("/", "_").replace(" ", "_")
        ruta = SHAP_DIR / f"shap_dependence_solo_historia_{nombre_archivo}.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✓ {ruta.name}")

    # --- Ranking numérico ---
    df_imp = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_importance_mean_abs": np.abs(sv).mean(axis=0),
        "shap_mean": sv.mean(axis=0),
    }).sort_values("shap_importance_mean_abs", ascending=False)

    ruta = SHAP_DIR / "shap_importance_solo_historia.csv"
    df_imp.to_csv(ruta, index=False)
    print(f"   ✓ {ruta.name}")

    print("\n   Top 10 features (solo historia):")
    for _, row in df_imp.head(10).iterrows():
        print(f"     {row['feature']:<42} "
              f"|SHAP|={row['shap_importance_mean_abs']:.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("EXPERIMENTO DE ABLACIÓN B — SOLO HISTORIA DELICTIVA")
    print("=" * 75)

    print("\n📂 Cargando dataset...")
    df = cargar_dataset()

    print(f"Filas:              {len(df):,}")
    print(f"Columnas:           {len(df.columns):,}")
    print(f"Tasa de hotspots:   {df[TARGET_COL].mean()*100:.2f}%")

    print("\n🧹 Preparando X e y solo con historia delictiva...")
    X, y, features = preparar_xy(df)

    print(f"Features seleccionadas: {len(features):,}")
    for f in features:
        print(f" - {f}")

    print(f"\nObservaciones:          {X.shape[0]:,}")
    print(f"Thresholds evaluados:   {len(THRESHOLDS)}")

    print("\n🤖 Entrenando modelos con validación temporal por mes completo...")
    resultados_folds, modelos_entrenados = evaluar_modelos_thresholds(X, y, df)

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

    # ROC-AUC por modelo (independiente del threshold): la métrica central
    # de comparación contra ML_05 (completo) y ML_05A (solo infraestructura).
    resumen_auc = (
        resultados_folds
        .groupby("modelo")["roc_auc"]
        .mean()
        .reset_index()
        .sort_values("roc_auc", ascending=False)
    )

    mejores = (
        resumen.sort_values("f1", ascending=False)
        .groupby("modelo", as_index=False)
        .head(1)
        .sort_values("f1", ascending=False)
    )

    print("\n📊 ROC-AUC promedio por modelo (comparar contra ML_05 y ML_05A):")
    print(resumen_auc.to_string(index=False, float_format="%.4f"))

    print("\n📊 Mejores thresholds por modelo según F1:")
    print(mejores.to_string(index=False, float_format="%.4f"))

    exportar_csv_seguro(resumen, OUTPUT_THRESHOLDS)
    exportar_csv_seguro(mejores, OUTPUT_BEST)
    exportar_csv_seguro(resumen_auc, OUTPUT_RESUMEN_AUC)

    print("\n📌 Exportando feature importance...")
    exportar_feature_importance(modelos_entrenados, X.columns.tolist())

    if "XGBoost" in modelos_entrenados:
        analisis_shap_xgboost(modelos_entrenados["XGBoost"], X, y)

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados thresholds:  outputs/{OUTPUT_THRESHOLDS.name}")
    print(f"Mejores thresholds:     outputs/{OUTPUT_BEST.name}")
    print(f"Resumen ROC-AUC:        outputs/{OUTPUT_RESUMEN_AUC.name}")
    print(f"Feature Importance RF:  outputs/{OUTPUT_RF_IMPORTANCE.name}")
    print(f"Feature Importance XGB: outputs/{OUTPUT_XGB_IMPORTANCE.name}")
    if SHAP_DISPONIBLE and "XGBoost" in modelos_entrenados:
        print(f"Análisis SHAP:          outputs/{SHAP_DIR.name}/")
    print(f"\n📌 NOTA (para la tesis):")
    print(f"  La métrica de comparación es ROC-AUC.")
    print(f"  Referencia ML_05 (todas las variables, XGBoost): 0.893")
    print(f"  Comparar también contra ML_05A (solo infraestructura).")
    print("===================================================")


if __name__ == "__main__":
    main()