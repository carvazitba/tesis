"""
Script: ML_05_modelos_hotspots.py

Entrena modelos para predecir hotspots delictivos:
- Logistic Regression
- Random Forest
- XGBoost

Entrada:
    outputs/dataset_ml_features_temporales.csv

Salidas:
    outputs/resultados_modelos_hotspots.csv
    outputs/feature_importance_random_forest.csv
    outputs/feature_importance_xgboost.csv
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


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_CSV = OUTPUT_DIR / "dataset_ml_features_temporales.csv"

OUTPUT_RESULTADOS = OUTPUT_DIR / "resultados_modelos_hotspots.csv"
OUTPUT_RF_IMPORTANCE = OUTPUT_DIR / "feature_importance_random_forest.csv"
OUTPUT_XGB_IMPORTANCE = OUTPUT_DIR / "feature_importance_xgboost.csv"

TARGET_COL = "hotspot_exploratorio"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# FUNCIONES
# ============================================================

def exportar_csv_seguro(df: pd.DataFrame, path: Path) -> None:
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

    # Convertir variables categóricas a dummies
    X = pd.get_dummies(X, drop_first=True)

    # Quedarse solo con columnas numéricas
    X = X.select_dtypes(include=[np.number]).copy()

    # Reemplazar inf y nulos
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y


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
        print("   pip install xgboost")

    return modelos


def calcular_metricas(y_true, y_pred, y_proba):
    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba)
        if len(np.unique(y_true)) > 1 else np.nan,
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metricas.update({
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })

    return metricas


def evaluar_modelos(X, y):
    modelos = obtener_modelos()
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    resultados = []
    modelos_entrenados = {}

    for nombre_modelo, modelo in modelos.items():
        print("\n===================================================")
        print(f"ENTRENANDO MODELO: {nombre_modelo}")
        print("===================================================")

        fold = 1

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(
                f"Fold {fold}: "
                f"train={len(train_idx):,} | test={len(test_idx):,} | "
                f"hotspots train={y_train.mean()*100:.2f}% | "
                f"hotspots test={y_test.mean()*100:.2f}%"
            )

            modelo.fit(X_train, y_train)

            y_pred = modelo.predict(X_test)

            if hasattr(modelo, "predict_proba"):
                y_proba = modelo.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred

            metricas = calcular_metricas(y_test, y_pred, y_proba)

            metricas["modelo"] = nombre_modelo
            metricas["fold"] = fold
            resultados.append(metricas)

            fold += 1

        # Entrenar modelo final con todo el dataset
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
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("MODELADO PREDICTIVO DE HOTSPOTS")
    print("=" * 60)

    print("\n📂 Cargando dataset...")
    df = cargar_dataset()

    print(f"Filas:    {len(df):,}")
    print(f"Columnas: {len(df.columns):,}")
    print(f"Tasa de hotspots: {df[TARGET_COL].mean()*100:.2f}%")

    print("\n🧹 Preparando X e y...")
    X, y = preparar_xy(df)

    print(f"Features usadas: {X.shape[1]:,}")
    print(f"Observaciones:   {X.shape[0]:,}")

    print("\n🤖 Entrenando modelos con TimeSeriesSplit...")
    resultados_folds, modelos_entrenados = evaluar_modelos(X, y)

    print("\n📊 Resultados por fold:")
    print(resultados_folds.to_string(index=False))

    resumen = (
        resultados_folds
        .groupby("modelo")
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
        .sort_values("roc_auc", ascending=False)
    )

    print("\n📊 Resumen promedio por modelo:")
    print(resumen.to_string(index=False, float_format="%.4f"))

    exportar_csv_seguro(resumen, OUTPUT_RESULTADOS)

    print("\n📌 Exportando feature importance...")
    exportar_feature_importance(modelos_entrenados, X.columns.tolist())

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados:           outputs/{OUTPUT_RESULTADOS.name}")
    print(f"Feature Importance RF: outputs/{OUTPUT_RF_IMPORTANCE.name}")
    print(f"Feature Importance XGB: outputs/{OUTPUT_XGB_IMPORTANCE.name}")
    print("===================================================")


if __name__ == "__main__":
    main()