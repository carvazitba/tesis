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

Métricas evaluadas:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - Matriz de confusión acumulada
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

TARGET_COL = "hotspot_exploratorio"

N_SPLITS = 5
RANDOM_STATE = 42

THRESHOLDS = np.arange(0.05, 0.81, 0.05)


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

    df = df.sort_values("fecha_mes").reset_index(drop=True)

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

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

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

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    resultados = []

    print("\n🤖 Entrenando XGBoost y evaluando thresholds...")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
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

    exportar_csv_seguro(resumen, OUTPUT_THRESHOLDS)
    exportar_csv_seguro(mejor_f1, OUTPUT_BEST)

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados thresholds: outputs/{OUTPUT_THRESHOLDS.name}")
    print(f"Mejor threshold:       outputs/{OUTPUT_BEST.name}")
    print("===================================================")


if __name__ == "__main__":
    main()