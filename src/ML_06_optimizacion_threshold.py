"""
Script: ML_06_optimizacion_threshold.py

Descripción:
Optimiza el threshold de clasificación para el modelo XGBoost.

Objetivo:
Evaluar distintos umbrales de probabilidad para transformar las predicciones
probabilísticas del modelo en clases binarias:

    probabilidad >= threshold -> hotspot = 1
    probabilidad < threshold  -> hotspot = 0

Entrada:
    outputs/dataset_ml_features_temporales.csv

Salidas:
    outputs/resultados_threshold_xgboost.csv
    outputs/mejor_threshold_xgboost.csv
    outputs/predicciones_mejor_threshold_xgboost.csv  (para SHAP / análisis posterior)

Métricas evaluadas:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - Matriz de confusión acumulada

NOTA METODOLÓGICA:
    - La validación usa splits temporales por MES COMPLETO (misma lógica
      que ML_05): se entrena con meses anteriores y se evalúa contra meses
      posteriores completos, sin partir ningún mes entre train y test.
    - Las predicciones exportadas son OUT-OF-FOLD: cada fila fue predicha
      por un modelo que NO la vio en entrenamiento.
    - Solo se exportan las predicciones binarizadas con el MEJOR threshold
      (según F1), junto con la probabilidad continua.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_CSV = OUTPUT_DIR / "dataset_ml_features_temporales.csv"

OUTPUT_THRESHOLDS = OUTPUT_DIR / "resultados_threshold_xgboost.csv"
OUTPUT_BEST = OUTPUT_DIR / "mejor_threshold_xgboost.csv"
OUTPUT_BEST_RECALL_CONTROLADO = OUTPUT_DIR / "mejor_threshold_recall_controlado_xgboost.csv"
OUTPUT_PREDICCIONES = OUTPUT_DIR / "predicciones_mejor_threshold_xgboost.csv"

TARGET_COL = "hotspot_exploratorio"

N_SPLITS = 5
RANDOM_STATE = 42

# Rango fino: 0.05 a 0.95 cada 0.01. El costo es despreciable porque
# las probabilidades se calculan una sola vez por fold; solo se
# re-binariza para cada threshold.
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

# Columnas identificadoras a incluir en el export de predicciones
COLUMNAS_ID = ["fecha_mes", "grid_id", "franja"]


# ============================================================
# FUNCIONES AUXILIARES
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
        raise ValueError(
            f"No se encontró la columna target '{TARGET_COL}'. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    if "fecha_mes" not in df.columns:
        raise ValueError("No se encontró la columna 'fecha_mes'.")

    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"], errors="coerce")
    df = df.dropna(subset=["fecha_mes"]).copy()

    # Ordenar por temporal + espacial para reproducibilidad (igual que ML_05)
    df = df.sort_values(["fecha_mes", "grid_id", "franja"]).reset_index(drop=True)

    return df


def preparar_xy(df: pd.DataFrame):
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


def generar_splits_temporales_por_mes(df: pd.DataFrame, n_splits: int = 5):
    """
    Genera splits temporales cortando por meses COMPLETOS (fecha_mes),
    no por filas. Garantiza que ningún mes quede repartido entre
    train y test: se entrena con meses anteriores y se evalúa contra
    meses posteriores completos.

    Misma lógica que ML_05 para que los resultados sean comparables.
    """
    meses = np.array(sorted(df["fecha_mes"].unique()))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_mes_idx, test_mes_idx in tscv.split(meses):
        meses_train = meses[train_mes_idx]
        meses_test = meses[test_mes_idx]

        train_idx = df.index[df["fecha_mes"].isin(meses_train)].to_numpy()
        test_idx = df.index[df["fecha_mes"].isin(meses_test)].to_numpy()

        yield train_idx, test_idx


def crear_modelo_xgboost(y_train: pd.Series):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError(
            "XGBoost no está instalado. Instalalo con:\n"
            "python -m pip install xgboost"
        )

    negativos = (y_train == 0).sum()
    positivos = (y_train == 1).sum()

    scale_pos_weight = negativos / positivos if positivos > 0 else 1

    modelo = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return modelo


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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 65)
    print("OPTIMIZACIÓN DE THRESHOLD - XGBOOST")
    print("=" * 65)

    print("\n📂 Cargando dataset...")
    df = cargar_dataset()

    print(f"  Filas:             {len(df):,}")
    print(f"  Columnas:          {len(df.columns):,}")
    print(f"  Tasa de hotspots:  {df[TARGET_COL].mean() * 100:.2f}%")

    print("\n🧹 Preparando X e y...")
    X, y = preparar_xy(df)

    print(f"  Features usadas:   {X.shape[1]:,}")
    print(f"  Observaciones:     {X.shape[0]:,}")

    resultados = []
    predicciones_oof = []  # acumula predicciones out-of-fold

    print("\n🤖 Entrenando XGBoost y evaluando thresholds...")
    print("   (splits temporales por mes completo, igual que ML_05)")

    for fold, (train_idx, test_idx) in enumerate(
        generar_splits_temporales_por_mes(df, N_SPLITS), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(
            f"\nFold {fold}: "
            f"train={len(train_idx):,} | test={len(test_idx):,} | "
            f"hotspots train={y_train.mean()*100:.2f}% | "
            f"hotspots test={y_test.mean()*100:.2f}%"
        )

        modelo = crear_modelo_xgboost(y_train)
        modelo.fit(X_train, y_train)

        y_proba = modelo.predict_proba(X_test)[:, 1]

        # Acumular predicciones out-of-fold con identificadores
        pred_fold = df.loc[test_idx, COLUMNAS_ID].copy()
        pred_fold["fold"] = fold
        pred_fold["y_true"] = y_test.values
        pred_fold["y_proba"] = y_proba
        predicciones_oof.append(pred_fold)

        for threshold in THRESHOLDS:
            y_pred = (y_proba >= threshold).astype(int)

            metricas = calcular_metricas(y_test, y_pred, y_proba)

            metricas["fold"] = fold
            metricas["threshold"] = round(float(threshold), 2)

            resultados.append(metricas)

    resultados_df = pd.DataFrame(resultados)

    resumen = (
        resultados_df
        .groupby("threshold")
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
        .sort_values("threshold")
    )

    mejor_f1 = resumen.sort_values("f1", ascending=False).head(1).copy()
    mejor_recall_controlado = (
        resumen[resumen["precision"] >= 0.40]
        .sort_values("recall", ascending=False)
        .head(1)
        .copy()
    )

    print("\n📊 Resultados promedio por threshold:")
    print(resumen.to_string(index=False, float_format="%.4f"))

    print("\n🏆 Mejor threshold según F1:")
    print(mejor_f1.to_string(index=False, float_format="%.4f"))

    if not mejor_recall_controlado.empty:
        print("\n🎯 Mejor threshold maximizando recall con precision >= 0.40:")
        print(mejor_recall_controlado.to_string(index=False, float_format="%.4f"))

    # ========================================================
    # EXPORT DE PREDICCIONES AL MEJOR THRESHOLD (para SHAP /
    # análisis posterior). Solo se binariza con el mejor
    # threshold según F1; se conserva también la probabilidad.
    # ========================================================
    mejor_threshold = float(mejor_f1["threshold"].iloc[0])

    predicciones_df = pd.concat(predicciones_oof, ignore_index=True)
    predicciones_df["threshold_usado"] = mejor_threshold
    predicciones_df["y_pred"] = (
        predicciones_df["y_proba"] >= mejor_threshold
    ).astype(int)

    print(f"\n💾 Exportando predicciones out-of-fold "
          f"binarizadas con threshold = {mejor_threshold:.2f}...")
    print(f"   Filas: {len(predicciones_df):,} "
          f"(cada una predicha por un modelo que no la vio en train)")

    exportar_csv_seguro(predicciones_df, OUTPUT_PREDICCIONES)
    exportar_csv_seguro(resumen, OUTPUT_THRESHOLDS)
    exportar_csv_seguro(mejor_f1, OUTPUT_BEST)

    if not mejor_recall_controlado.empty:
        exportar_csv_seguro(
            mejor_recall_controlado,
            OUTPUT_BEST_RECALL_CONTROLADO
        )

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados thresholds: outputs/{OUTPUT_THRESHOLDS.name}")
    print(f"Mejor threshold (F1):  outputs/{OUTPUT_BEST.name}")
    if not mejor_recall_controlado.empty:
        print(f"Mejor threshold (recall, precision>=0.40): "
              f"outputs/{OUTPUT_BEST_RECALL_CONTROLADO.name}")
    print(f"Predicciones (mejor threshold F1): outputs/{OUTPUT_PREDICCIONES.name}")
    print(f"\n📌 NOTA (para la tesis):")
    print(f"  El mejor threshold ({mejor_threshold:.2f}) fue seleccionado por F1")
    print(f"  sobre predicciones out-of-fold con validación temporal por mes.")
    print("===================================================")


if __name__ == "__main__":
    main()