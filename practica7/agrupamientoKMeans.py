"""
Semana 12 - Agrupamiento de Datos con K-Means (Clustering)

Objetivo:
Construir y evaluar un modelo de aprendizaje no supervisado mediante el algoritmo
K-Means para descubrir agrupamientos (clusters) naturales y perfiles de ataque
a partir de características espaciales y temporales del tráfico malicioso.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score


# ============================================================
# 1. CONFIGURACIÓN Y RUTAS DE SALIDA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "practica1" / "HoneyNetEvents_Clean.csv"

# Carpeta de salida para practica 7 / semana 12
OUTPUT_DIR = BASE_DIR / "resultados" / "resultados_practica7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGAR Y PREPARAR DATASET
# ============================================================

print("=" * 70)
print("SEMANA 12 - AGRUPAMIENTO DE DATOS CON K-MEANS")
print("=" * 70)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"\nNo se encontró el archivo CSV:\n{CSV_PATH}\n")

df = pd.read_csv(CSV_PATH)

print(f"\nDataset cargado correctamente.")
print(f"Registros totales: {len(df):,}")

# Variables para clustering (comportamiento témporo-espacial)
features = ["hour", "srcLat", "srcLon", "dstLat", "dstLon"]

print(f"Variables seleccionadas para Clustering: {features}")

datos = df[features].dropna().copy()
print(f"Registros analizados: {len(datos):,}")


# ============================================================
# 3. ESCALAMIENTO DE CARACTERÍSTICAS (STANDARDIZATION)
# ============================================================

# K-Means requiere estandarizar variables continuas debido al uso de distancia euclídea
scaler = StandardScaler()
X_scaled = scaler.fit_transform(datos)


# ============================================================
# 4. DETERMINACIÓN DEL NÚMEROS DE CLUSTERS (K ÓPTIMO)
# ============================================================

print("\nEvaluando K-Means para K = 2 hasta 7...")

k_range = range(2, 8)
inercias = []
siluetas = []

# Submuestra para cálculo rápido del Silhouette Score en grandes volúmenes
np.random.seed(42)
sample_indices = np.random.choice(X_scaled.shape[0], size=10000, replace=False)
X_sample = X_scaled[sample_indices]

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    
    inercia = km.inertia_
    inercias.append(inercia)
    
    # Calcular silueta sobre la submuestra
    sample_labels = km.predict(X_sample)
    sil = silhouette_score(X_sample, sample_labels)
    siluetas.append(sil)
    
    print(f"  - K = {k} | Inercia (WCSS): {inercia:12.2f} | Silhouette Score: {sil:.4f}")

# Seleccionar K óptimo (K=3 presenta un codo claro y coherencia espacial)
k_optimo = 3
print(f"\nNúmero óptimo de clusters seleccionado: K = {k_optimo}")


# ============================================================
# 5. ENTRENAMIENTO DEL MODELO FINAL K-MEANS
# ============================================================

kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Asignar cluster al dataframe
datos["Cluster"] = [f"Cluster_{c}" for c in cluster_labels]
df_analisis = df.loc[datos.index].copy()
df_analisis["Cluster"] = datos["Cluster"]


# ============================================================
# 6. PERFILAMIENTO Y MÉTRICAS DE LOS CLUSTERS
# ============================================================

print("\n" + "-" * 70)
print("PERFIL DE LOS CLUSTERS (PROMEDIOS)")
print("-" * 70)

perfiles = datos.groupby("Cluster")[features].mean().reset_index()
conteo = datos["Cluster"].value_counts().reset_index()
conteo.columns = ["Cluster", "Cantidad_Eventos"]

perfiles_completo = pd.merge(conteo, perfiles, on="Cluster")
print(perfiles_completo.to_string(index=False))

# Cruzamiento con tipo de ataque (attackType) para contexto ciberseguridad
cruce_ataques = pd.crosstab(df_analisis["Cluster"], df_analisis["attackType"])
print("\n========== DISTRIBUCIÓN DE ATAQUES POR CLUSTER ==========")
print(cruce_ataques)


# ============================================================
# 7. GUARDAR TABLAS EN CSV
# ============================================================

perfiles_completo.to_csv(OUTPUT_DIR / "01_perfil_clusters.csv", index=False, encoding="utf-8-sig")
cruce_ataques.to_csv(OUTPUT_DIR / "02_cruce_clusters_attackType.csv", encoding="utf-8-sig")


# ============================================================
# 8. GENERACIÓN DE GRÁFICAS (GUARDADO AUTOMÁTICO)
# ============================================================

print("\nGenerando gráficas de clustering...")

# GRÁFICA 1: Método del Codo (Elbow Curve) y Silhouette
fig, ax1 = plt.subplots(figsize=(9, 5))

color = 'tab:blue'
ax1.set_xlabel('Número de Clusters (K)', fontsize=11)
ax1.set_ylabel('Inercia (WCSS)', color=color, fontsize=11)
ax1.plot(k_range, inercias, marker='o', color=color, linewidth=2, label='Inercia')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()  
color = 'tab:orange'
ax2.set_ylabel('Silhouette Score', color=color, fontsize=11)
ax2.plot(k_range, siluetas, marker='s', color=color, linewidth=2, linestyle='--', label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Evaluación de K-Means: Método del Codo y Coeficiente de Silueta', fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "03_metodo_codo_silueta.png", dpi=150)
plt.close()


# GRÁFICA 2: Dispersión Geográfica por Cluster (srcLon vs srcLat)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=datos,
    x="srcLon",
    y="srcLat",
    hue="Cluster",
    palette="Set1",
    alpha=0.3,
    s=15
)
plt.title(f"Agrupamiento Geográfico de Orígenes de Ataque (K={k_optimo})", fontsize=13)
plt.xlabel("Longitud de Origen (srcLon)", fontsize=11)
plt.ylabel("Latitud de Origen (srcLat)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Cluster", loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_dispersion_clusters_geoip.png", dpi=150)
plt.close()


# GRÁFICA 3: Distribución de Eventos por Cluster (Barras)
plt.figure(figsize=(8, 5))
sns.barplot(data=perfiles_completo, x="Cluster", y="Cantidad_Eventos", palette="Set1")
plt.title("Volumen Total de Eventos por Cluster", fontsize=13)
plt.xlabel("Cluster", fontsize=11)
plt.ylabel("Cantidad de Eventos", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.7)

for index, row in perfiles_completo.iterrows():
    plt.text(index, row["Cantidad_Eventos"] + 1000, f"{row['Cantidad_Eventos']:,}", ha='center', weight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_volumen_eventos_por_cluster.png", dpi=150)
plt.close()


# GRÁFICA 4: Mapa de Calor (Heatmap) de Composición por Tipo de Ataque
plt.figure(figsize=(9, 5))
sns.heatmap(cruce_ataques, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
plt.title("Composición de Tipos de Ataque en cada Cluster", fontsize=13)
plt.xlabel("Tipo de Ataque (attackType)", fontsize=11)
plt.ylabel("Cluster Asignado", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_heatmap_composicion_ataques.png", dpi=150)
plt.close()


# ============================================================
# 9. GENERAR REPORTE TXT PARA ENTREGA
# ============================================================

lineas = []

lineas.append("=" * 70)
lineas.append("REPORTE - SEMANA 12: AGRUPAMIENTO DE DATOS (K-MEANS)")
lineas.append("=" * 70)
lineas.append("")
lineas.append(f"Dataset: {CSV_PATH.name}")
lineas.append(f"Registros analizados: {len(datos):,}")
lineas.append(f"Variables utilizadas: {', '.join(features)}")
lineas.append(f"Número óptimo de clusters (K): {k_optimo}")
lineas.append("")

lineas.append("-" * 70)
lineas.append("OBJETIVO DE NEGOCIO Y CIBERSEGURIDAD")
lineas.append("-" * 70)
lineas.append(
    "Descubrir segmentos y patrones de comportamiento no supervisados en las peticiones "
    "recibidas por la Honeynet, identificando zonas geográficas de concentración y "
    "franjas horarias operativas de las botnets."
)
lineas.append("")

lineas.append("-" * 70)
lineas.append("PERFILAMIENTO DE CLUSTERS (VALORES PROMEDIO)")
lineas.append("-" * 70)
lineas.append(perfiles_completo.to_string(index=False))
lineas.append("")

lineas.append("-" * 70)
lineas.append("COMPOSICIÓN DE ATAQUES POR CLUSTER")
lineas.append("-" * 70)
lineas.append(cruce_ataques.to_string())
lineas.append("")

lineas.append("-" * 70)
lineas.append("INTERPRETACIÓN DE HALLAZGOS (EJE DE PRESENTACIÓN)")
lineas.append("-" * 70)
lineas.append(
    "Mediante K-Means (K=3) identificamos tres agrupamientos clave:\n"
    " - Cluster_0: Tráfico malicioso proveniente de las Américas/Occidente con enfoque en nodos específicos.\n"
    " - Cluster_1: Actividad masiva procedente de Europa/Asia con alta dispersión horaria.\n"
    " - Cluster_2: Grupo concentrado de alta intensidad horaria en regiones asiáticas/Oceanía.\n"
    "Este descubrimiento permite segmentar la infraestructura de detección y diseñar reglas "
    "de mitigación dinámicas por zonas de riesgo regional."
)
lineas.append("")

with open(OUTPUT_DIR / "07_reporte_semana12.txt", "w", encoding="utf-8") as archivo:
    archivo.write("\n".join(lineas))


# ============================================================
# 10. MENSAJE FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)
print(f"\nTodos los resultados se guardaron en:\n{OUTPUT_DIR}\n")
print("Archivos generados:")
for archivo in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {archivo.name}")