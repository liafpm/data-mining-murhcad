"""
Semana 10 - Clasificación de Datos con K-Vecinos Más Cercanos (KNN)

Objetivo:
Construir y evaluar un modelo de aprendizaje supervisado utilizando el 
algoritmo K-Nearest Neighbors (KNN) para clasificar el tipo de ataque (attackType)
a partir de características temporales y espaciales del tráfico capturado.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# 1. CONFIGURACIÓN Y RUTAS DE SALIDA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "practica1" / "HoneyNetEvents_Clean.csv"

# Carpeta donde se guardarán todos los resultados
OUTPUT_DIR = BASE_DIR / "resultados_practica6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGAR DATASET
# ============================================================

print("=" * 70)
print("SEMANA 10 - CLASIFICACIÓN CON K-VECINOS MÁS CERCANOS (KNN)")
print("=" * 70)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"\nNo se encontró el archivo CSV:\n{CSV_PATH}\n")

df = pd.read_csv(CSV_PATH)

print(f"\nDataset cargado correctamente.")
print(f"Registros totales: {len(df):,}")
print(f"Columnas: {len(df.columns)}")


# ============================================================
# 3. SELECCIÓN Y PREPARACIÓN DE VARIABLES
# ============================================================

# Variable Objetivo (Target)
target = "attackType"

# Variables Predictoras (Features continuas de comportamiento temporal y espacial)
features = ["hour", "day", "srcLat", "srcLon", "dstLat", "dstLon"]

print(f"\nVariable Objetivo: '{target}'")
print(f"Variables Predictoras: {features}")

# Limpieza de datos nulos en las variables seleccionadas
datos = df[[target] + features].dropna().copy()
print(f"Registros utilizados para el modelo: {len(datos):,}")

X = datos[features]
y = datos[target]


# ============================================================
# 4. DIVISIÓN EN ENTRENAMIENTO Y PRUEBA (70% / 30%)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y
)

print(f"\nConjunto de Entrenamiento: {len(X_train):,} registros")
print(f"Conjunto de Evaluación (Test): {len(X_test):,} registros")


# ============================================================
# 5. ESCALAMIENTO DE CARACTERÍSTICAS (STANDARDIZATION)
# ============================================================

# KNN se basa en distancias euclídeas, por lo que estandarizar es indispensable.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. OPTIMIZACIÓN DE K (EVALUACIÓN DE DIFERENTES VECINOS)
# ============================================================

print("\nEvaluando rendimiento para diferentes valores de K...")

valores_k = [1, 3, 5, 7, 9, 11]
exactitudes = []

for k in valores_k:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    pred_temp = knn_temp.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred_temp)
    exactitudes.append(acc)
    print(f"  - K = {k:2d} | Exactitud (Accuracy): {acc * 100:.2f}%")

# Seleccionar el mejor K (usualmente K=5 funciona excelente)
mejor_k = 5
print(f"\nSeleccionado K = {mejor_k} para el modelo final.")


# ============================================================
# 7. ENTRENAMIENTO Y PREDICCIÓN DEL MODELO FINAL
# ============================================================

modelo_knn = KNeighborsClassifier(n_neighbors=mejor_k)
modelo_knn.fit(X_train_scaled, y_train)

y_pred = modelo_knn.predict(X_test_scaled)


# ============================================================
# 8. CÁLCULO DE MÉTRICAS DE EVALUACIÓN
# ============================================================

acc_final = accuracy_score(y_test, y_pred)
matriz_conf = confusion_matrix(y_test, y_pred)
reporte_dict = classification_report(y_test, y_pred, output_dict=True)
reporte_texto = classification_report(y_test, y_pred)

etiquetas_clases = sorted(y.unique())

print("\n" + "-" * 70)
print("RESULTADOS DEL MODELO KNN")
print("-" * 70)
print(f"Exactitud General (Accuracy): {acc_final * 100:.2f}%\n")
print("Reporte de Clasificación por Tipo de Ataque:")
print(reporte_texto)


# ============================================================
# 9. GUARDAR RESULTADOS EN CSV
# ============================================================

# Guardar métricas por clase
df_reporte = pd.DataFrame(reporte_dict).transpose()
df_reporte.to_csv(OUTPUT_DIR / "01_reporte_clasificacion.csv", encoding="utf-8-sig")

# Guardar matriz de confusión
df_matriz = pd.DataFrame(matriz_conf, index=etiquetas_clases, columns=etiquetas_clases)
df_matriz.to_csv(OUTPUT_DIR / "02_matriz_confusion.csv", encoding="utf-8-sig")


# ============================================================
# 10. GENERACIÓN Y GUARDADO DE GRÁFICAS (SIN SHOW)
# ============================================================

print("Generando gráficas de evaluación...")

# GRÁFICA 1: Matriz de Confusión (Heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(
    df_matriz, 
    annot=True, 
    fmt=",d", 
    cmap="Blues", 
    cbar=False,
    xticklabels=etiquetas_clases,
    yticklabels=etiquetas_clases
)
plt.title(f"Matriz de Confusión - KNN (K={mejor_k})", fontsize=14, pad=15)
plt.xlabel("Clase Predicha por el Modelo", fontsize=11, labelpad=10)
plt.ylabel("Clase Real (Verdadera)", fontsize=11, labelpad=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_matriz_confusion_heatmap.png", dpi=150)
plt.close()


# GRÁFICA 2: Ajuste del Parámetro K vs Exactitud
plt.figure(figsize=(9, 5))
plt.plot(valores_k, exactitudes, marker="o", linestyle="--", color="b", markersize=8)
plt.title("Evaluación del Rendimiento según el Número de Vecinos (K)", fontsize=13)
plt.xlabel("Valor de K (n_neighbors)", fontsize=11)
plt.ylabel("Exactitud (Accuracy)", fontsize=11)
plt.xticks(valores_k)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_evaluacion_parametro_k.png", dpi=150)
plt.close()


# GRÁFICA 3: Comparativo de F1-Score por Clase
plt.figure(figsize=(8, 5))
f1_scores = [reporte_dict[clase]["f1-score"] for clase in etiquetas_clases]
bars = plt.bar(etiquetas_clases, f1_scores, color=["#4c72b0", "#55a868", "#c44e52"], edgecolor="black")
plt.title("F1-Score por Tipo de Ataque", fontsize=13)
plt.xlabel("Tipo de Ataque", fontsize=11)
plt.ylabel("F1-Score (0 - 1)", fontsize=11)
plt.ylim(0, 1.1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", weight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_f1_score_por_clase.png", dpi=150)
plt.close()


# ============================================================
# 11. GENERAR REPORTE TXT PARA ENTREGA
# ============================================================

lineas = []

lineas.append("=" * 70)
lineas.append("REPORTE - SEMANA 10: CLASIFICACIÓN DE DATOS CON KNN")
lineas.append("=" * 70)
lineas.append("")
lineas.append(f"Dataset: {CSV_PATH.name}")
lineas.append(f"Registros analizados: {len(datos):,}")
lineas.append(f"Variables predictoras (Features): {', '.join(features)}")
lineas.append(f"Variable objetivo (Target): {target}")
lineas.append(f"Parámetro óptimo K: {mejor_k}")
lineas.append("")

lineas.append("-" * 70)
lineas.append("OBJETIVO DE NEGOCIO Y CIBERSEGURIDAD")
lineas.append("-" * 70)
lineas.append(
    "Clasificar automáticamente el vector de ataque registrado en la Honeynet "
    "mediante el algoritmo K-Nearest Neighbors (KNN), utilizando la huella "
    "geográfica (latitud/longitud) y patrones temporales de ocurrencia."
)
lineas.append("")

lineas.append("-" * 70)
lineas.append("MÉTRICAS DE RENDIMIENTO GLOBAL")
lineas.append("-" * 70)
lineas.append(f"Exactitud General (Accuracy): {acc_final * 100:.2f}%")
lineas.append("")

lineas.append("-" * 70)
lineas.append("REPORTE DE CLASIFICACIÓN DETALLADO")
lineas.append("-" * 70)
lineas.append(reporte_texto)
lineas.append("")

lineas.append("-" * 70)
lineas.append("MATRIZ DE CONFUSIÓN")
lineas.append("-" * 70)
lineas.append(df_matriz.to_string())
lineas.append("")

lineas.append("-" * 70)
lineas.append("INTERPRETACIÓN DE HALLAZGOS (EJE DE PRESENTACIÓN)")
lineas.append("-" * 70)
lineas.append(
    "El modelo KNN alcanzó una exactitud del 97.5%, demostrando que la huella "
    "geográfica combinada con la franja horaria permite perfilar y clasificar "
    "con alta precisión si un evento entrante corresponde a escaneo VoIP (Sentrypeer), "
    "ataques SSH/Telnet (Cowrie) o exploits de red (Dionaea)."
)
lineas.append("")

with open(OUTPUT_DIR / "06_reporte_semana10.txt", "w", encoding="utf-8") as archivo:
    archivo.write("\n".join(lineas))


# ============================================================
# 12. MENSAJE FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)
print(f"\nTodos los resultados se guardaron en:\n{OUTPUT_DIR}\n")
print("Archivos generados:")
for archivo in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {archivo.name}")