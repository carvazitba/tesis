"""
Script: ML_05A_modelos_solo_infraestructura.py

Descripción:
Experimento A: entrena modelos usando únicamente variables de infraestructura urbana
y variables temporales básicas, excluyendo variables históricas delictivas.

Además, evalúa múltiples thresholds entre 0.05 y 0.95.

Entrada:
    outputs/dataset_ml_features_temporales.csv

Salidas:
    outputs/resultados_solo_infraestructura_thresholds.csv
    outputs/mejor_threshold_solo_infraestructura.csv
    outputs/feature_importance_rf_solo_infraestructura.csv
    outputs/feature_importance_xgb_solo_infraestructura.csv
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

OUTPUT_THRESHOLDS = OUTPUT_DIR / "resultados_solo_infraestructura_thresholds.csv"
OUTPUT_BEST = OUTPUT_DIR / "mejor_threshold_solo_infraestructura.csv"
OUTPUT_RF_IMPORTANCE = OUTPUT_DIR / "feature_importance_rf_solo_infraestructura.csv"
OUTPUT_XGB_IMPORTANCE = OUTPUT_DIR / "feature_importance_xgb_solo_infraestructura.csv"

TARGET_COL = "hotspot_exploratorio"

N_SPLITS = 5
RANDOM_STATE = 42

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


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
    df = df.sort_values("fecha_mes").reset_index(drop=True)

    return df


def seleccionar_features_infraestructura(df: pd.DataFrame) -> list[str]:
    candidatas = []

    for col in df.columns:
        if (
            col.startswith("cant_")
            or col.startswith("dist_min_")
            or col in [
                "area_celda_m2",
                "area_interseccion_caba_m2",
                "porcentaje_en_caba",
                "anio",
                "mes_num",
                "trimestre",
                "semestre",
                "franja",
            ]
        ):
            candidatas.append(col)

    patrones_excluir = [
        "delitos",
        "hotspot",
        "promedio",
        "historico",
        "lag",
        "rolling",
        "umbral",
        "cantidad_delitos",
        "fecha_mes",
        "grid_id",
        "mes",
    ]

    features = []
    for col in candidatas:
        if any(p in col.lower() for p in patrones_excluir):
            continue
        features.append(col)

    return features


def preparar_xy(df: pd.DataFrame):
    features = seleccionar_features_infraestructura(df)

    if not features:
        raise ValueError("No se seleccionó ninguna feature de infraestructura.")

    X = df[features].copy()
    y = df[TARGET_COL].astype(int).copy()

    X = pd.get_dummies(X, drop_first=True)
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


def evaluar_modelos_thresholds(X, y):
    modelos = obtener_modelos()
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    resultados = []
    modelos_entrenados = {}

    for nombre_modelo, modelo in modelos.items():
        print("\n===================================================")
        print(f"ENTRENANDO MODELO: {nombre_modelo}")
        print("===================================================")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(
                f"Fold {fold}: "
                f"train={len(train_idx):,} | test={len(test_idx):,} | "
                f"hotspots train={y_train.mean()*100:.2f}% | "
                f"hotspots test={y_test.mean()*100:.2f}%"
            )

            modelo.fit(X_train, y_train)
            y_proba = modelo.predict_proba(X_test)[:, 1]

            for threshold in THRESHOLDS:
                y_pred = (y_proba >= threshold).astype(int)

                metricas = calcular_metricas(y_test, y_pred, y_proba)
                metricas["modelo"] = nombre_modelo
                metricas["fold"] = fold
                metricas["threshold"] = float(threshold)

                resultados.append(metricas)

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
    print("=" * 75)
    print("EXPERIMENTO A - SOLO INFRAESTRUCTURA + OPTIMIZACIÓN THRESHOLD")
    print("=" * 75)

    print("\n📂 Cargando dataset...")
    df = cargar_dataset()

    print(f"Filas:              {len(df):,}")
    print(f"Columnas:           {len(df.columns):,}")
    print(f"Tasa de hotspots:   {df[TARGET_COL].mean()*100:.2f}%")

    print("\n🧹 Preparando X e y solo con infraestructura...")
    X, y, features_originales = preparar_xy(df)

    print(f"Features originales seleccionadas: {len(features_originales):,}")
    for f in features_originales:
        print(f" - {f}")

    print(f"\nFeatures finales después de dummies: {X.shape[1]:,}")
    print(f"Observaciones:                     {X.shape[0]:,}")
    print(f"Thresholds evaluados:              {len(THRESHOLDS)}")

    print("\n🤖 Entrenando modelos con TimeSeriesSplit...")
    resultados_folds, modelos_entrenados = evaluar_modelos_thresholds(X, y)

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

    mejores = (
        resumen.sort_values("f1", ascending=False)
        .groupby("modelo", as_index=False)
        .head(1)
        .sort_values("f1", ascending=False)
    )

    print("\n📊 Mejores thresholds por modelo según F1:")
    print(mejores.to_string(index=False, float_format="%.4f"))

    exportar_csv_seguro(resumen, OUTPUT_THRESHOLDS)
    exportar_csv_seguro(mejores, OUTPUT_BEST)

    print("\n📌 Exportando feature importance...")
    exportar_feature_importance(modelos_entrenados, X.columns.tolist())

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados thresholds: outputs/{OUTPUT_THRESHOLDS.name}")
    print(f"Mejores thresholds:    outputs/{OUTPUT_BEST.name}")
    print(f"Feature Importance RF: outputs/{OUTPUT_RF_IMPORTANCE.name}")
    print(f"Feature Importance XGB: outputs/{OUTPUT_XGB_IMPORTANCE.name}")
    print("===================================================")


if __name__ == "__main__":
    main()