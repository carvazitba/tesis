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
import pickle

import numpy as np
import pandas as pd
import joblib

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
OUTPUT_FEATURE_NAMES = OUTPUT_DIR / "feature_names_ml05.csv"

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

    # Ordenar por temporal + espacial para reproducibilidad
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

    # labels=[0, 1] garantiza matriz 2x2 aunque un fold tenga una sola clase
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metricas.update({
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })

    return metricas


def generar_splits_temporales_por_mes(df: pd.DataFrame, n_splits: int = 5):
    """
    Genera splits temporales cortando por meses COMPLETOS (fecha_mes),
    no por filas. Garantiza que ningún mes quede repartido entre
    train y test: se entrena con meses anteriores y se evalúa contra
    meses posteriores completos.
    """
    meses = np.array(sorted(df["fecha_mes"].unique()))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_mes_idx, test_mes_idx in tscv.split(meses):
        meses_train = meses[train_mes_idx]
        meses_test = meses[test_mes_idx]

        train_idx = df.index[df["fecha_mes"].isin(meses_train)].to_numpy()
        test_idx = df.index[df["fecha_mes"].isin(meses_test)].to_numpy()

        yield train_idx, test_idx


def evaluar_modelos(X, y, df):
    modelos = obtener_modelos()

    resultados = []
    modelos_entrenados = {}

    for nombre_modelo, modelo in modelos.items():
        print("\n===================================================")
        print(f"ENTRENANDO MODELO: {nombre_modelo}")
        print("===================================================")

        fold = 1

        # ✅ AJUSTE: splits por fecha_mes completo (no por filas).
        # Cada fold entrena con meses anteriores y evalúa meses
        # posteriores completos, sin partir ningún mes.
        for train_idx, test_idx in generar_splits_temporales_por_mes(df, N_SPLITS):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(
                f"Fold {fold}: "
                f"train={len(train_idx):,} | test={len(test_idx):,} | "
                f"hotspots train={y_train.mean()*100:.2f}% | "
                f"hotspots test={y_test.mean()*100:.2f}%"
            )

            # ✅ AJUSTE: Compensar desbalance en XGBoost igual que en otros modelos
            if nombre_modelo == "XGBoost":
                positivos = y_train.sum()
                negativos = len(y_train) - positivos
                
                if positivos > 0:
                    scale_weight = negativos / positivos
                    modelo.set_params(scale_pos_weight=scale_weight)
                    print(f"         scale_pos_weight={scale_weight:.2f}")

            modelo.fit(X_train, y_train)

            # NOTA METODOLÓGICA (IMPORTANTE PARA TESIS):
            # El threshold usado aquí es 0.50 (umbral por defecto).
            # Esto permite comparación inicial entre modelos.
            # Para el modelo final, ver ML_06 donde se optimiza threshold.
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
        # ✅ AJUSTE: Aplicar scale_pos_weight al modelo final también
        if nombre_modelo == "XGBoost":
            positivos = y.sum()
            negativos = len(y) - positivos
            
            if positivos > 0:
                scale_weight = negativos / positivos
                modelo.set_params(scale_pos_weight=scale_weight)
        
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


def guardar_modelos(modelos_entrenados):
    """
    Guarda los modelos entrenados en disco.
    
    Utiliza:
    - joblib para RandomForest y XGBoost (formatos binarios compactos)
    - pickle para Logistic Regression (Pipeline)
    
    NOTA: El threshold = 0.50 fue usado solo para evaluar métricas en ML_05.
    Los modelos aprenden probabilidades; el threshold es una decisión de
    evaluación posterior. Para threshold optimizado, ver ML_06.
    
    Returns:
        Lista de nombres de archivo guardados.
    """
    print("\n💾 Guardando modelos entrenados...")
    print("   Nota: el threshold = 0.50 fue usado solo para evaluar métricas en ML_05.")
    
    guardados = []
    
    for nombre_modelo, modelo in modelos_entrenados.items():
        
        if "Logistic Regression" in nombre_modelo:
            ruta = OUTPUT_DIR / "ML_05_logistic_regression.pickle"
            with open(ruta, "wb") as f:
                pickle.dump(modelo, f)
            print(f"   ✓ {nombre_modelo}: {ruta.name}")
            guardados.append(ruta.name)
        
        elif "Random Forest" in nombre_modelo:
            ruta = OUTPUT_DIR / "ML_05_random_forest.joblib"
            joblib.dump(modelo, ruta)
            print(f"   ✓ {nombre_modelo}: {ruta.name}")
            guardados.append(ruta.name)
        
        elif "XGBoost" in nombre_modelo:
            ruta = OUTPUT_DIR / "ML_05_xgboost.joblib"
            joblib.dump(modelo, ruta)
            print(f"   ✓ {nombre_modelo}: {ruta.name}")
            guardados.append(ruta.name)
    
    print("   ✅ Todos los modelos guardados exitosamente")
    return guardados


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

    print("\n🤖 Entrenando modelos con validación temporal por mes completo...")
    resultados_folds, modelos_entrenados = evaluar_modelos(X, y, df)

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

    print("\n📌 Guardando feature names (para SHAP)...")
    pd.DataFrame({"feature": X.columns.tolist()}).to_csv(
        OUTPUT_FEATURE_NAMES,
        index=False
    )
    print(f"   ✓ {OUTPUT_FEATURE_NAMES.name}")

    print("\n📌 Guardando modelos...")
    modelos_guardados = guardar_modelos(modelos_entrenados)

    print("\n===================================================")
    print("✅ PROCESO COMPLETADO")
    print("===================================================")
    print(f"Resultados:              outputs/{OUTPUT_RESULTADOS.name}")
    print(f"Feature Importance RF:   outputs/{OUTPUT_RF_IMPORTANCE.name}")
    print(f"Feature Importance XGB:  outputs/{OUTPUT_XGB_IMPORTANCE.name}")
    print(f"Feature Names:           outputs/{OUTPUT_FEATURE_NAMES.name}")
    print(f"\n🤖 Modelos guardados:")
    for nombre in modelos_guardados:
        print(f"  - {nombre}")
    
    if "XGBoost" not in modelos_entrenados:
        print(f"\n⚠️  ADVERTENCIA: XGBoost NO fue entrenado (falta instalarlo).")
        print(f"   Instalá con: python -m pip install xgboost")
        print(f"   y volvé a correr este script.")
    
    print(f"\n📌 NOTA IMPORTANTE (para la tesis):")
    print(f"  Los resultados de ML_05 usan threshold = 0.50 (comparación inicial).")
    print(f"  Para threshold optimizado, ver outputs de ML_06.")
    print(f"\n📌 Próximos pasos:")
    print(f"  1. Ejecutar shap_analysis.py para generar explicaciones SHAP")
    print(f"  2. Integrar visualizaciones en sección 5.3 de la tesis")
    print("===================================================\n")


if __name__ == "__main__":
    main()