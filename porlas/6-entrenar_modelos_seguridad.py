import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# === Rutas ===
data_path = 'C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/alojamientos-geocodificados-2.csv'
modelo_guardado = 'mejor_modelo.pkl'
reporte_txt = 'resultados_modelos.txt'

# === Cargar datos ===
df = pd.read_csv(data_path)
print(f"Total de registros cargados: {len(df)}")

# === Features y target ===
X = df[['latitud', 'longitud', 'densidad', 'cluster']]
y = df['seguridad']

# === Codificar clases de seguridad ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
clases = label_encoder.classes_
clase_ids = np.arange(len(clases))

# === Escalar features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Separar datos ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# === Inicializar modelos ===
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Red Neuronal": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
}

# === Evaluación ===
mejor_modelo = None
mejor_score = 0
modelo_nombre = ''
reportes = []

for nombre, modelo in modelos.items():
    print(f"\n🔍 Entrenando modelo: {nombre}")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # === Accuracy train vs test
    acc_train = modelo.score(X_train, y_train)
    acc_test = modelo.score(X_test, y_test)

    # === Cross-validation
    scores_cv = cross_val_score(modelo, X_scaled, y_encoded, cv=5)
    acc_cv_mean = scores_cv.mean()
    acc_cv_std = scores_cv.std()

    # === Classification Report
    reporte = classification_report(
        y_test, y_pred,
        labels=clase_ids,
        target_names=clases,
        digits=4,
        zero_division=0
    )

    # === Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=clase_ids)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    disp.plot()
    plt.title(f"Matriz de confusión - {nombre}")
    plt.savefig(f"confusion_{nombre.replace(' ', '_')}.png")
    plt.close()

    # === Consolidar resultados
    resumen = f"""
Modelo: {nombre}
Train Accuracy: {acc_train:.4f}
Test Accuracy : {acc_test:.4f}
Cross-val mean: {acc_cv_mean:.4f} (± {acc_cv_std:.4f})

Clasificación:
{reporte}
"""
    print(resumen)
    reportes.append(resumen)

    if acc_test > mejor_score:
        mejor_score = acc_test
        mejor_modelo = modelo
        modelo_nombre = nombre

# === Guardar el mejor modelo con scaler y label encoder ===
joblib.dump({
    "modelo": mejor_modelo,
    "scaler": scaler,
    "label_encoder": label_encoder
}, modelo_guardado)

print(f"\n💾 Mejor modelo guardado: {modelo_nombre} con accuracy {mejor_score:.4f}")
print(f"📁 Archivo: {modelo_guardado}")

# === Guardar los reportes de todos los modelos ===
with open(reporte_txt, 'w', encoding='utf-8') as f:
    f.write("Comparación de Modelos de Clasificación de Seguridad\n")
    f.write("="*60 + "\n")
    for r in reportes:
        f.write(r + "\n" + "-"*60 + "\n")

print(f"📝 Reporte completo guardado en: {reporte_txt}")
    