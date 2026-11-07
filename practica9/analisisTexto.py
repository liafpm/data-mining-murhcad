"""
Semana 14 - Análisis de Texto y Nube de Palabras (Text Mining)

Objetivo:
Realizar un análisis exploratorio de datos textuales a partir de los campos de 
inteligencia de red (srcOrg, srcCountryName, protocol) del dataset Honeynet, 
generando métricas de frecuencia de términos y una Nube de Palabras (Word Cloud).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# ============================================================
# 1. CONFIGURACIÓN Y RUTAS DE SALIDA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "practica1" / "HoneyNetEvents_Clean.csv"

# Carpeta de salida para practica 9 / semana 14
OUTPUT_DIR = BASE_DIR / "resultados_practica9"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGAR DATASET Y SELECCIÓN DE CAMPOS TEXTUALES
# ============================================================

print("=" * 70)
print("SEMANA 14 - ANÁLISIS DE TEXTO Y NUBE DE PALABRAS")
print("=" * 70)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"\nNo se encontró el archivo CSV:\n{CSV_PATH}\n")

df = pd.read_csv(CSV_PATH)

print(f"\nDataset cargado correctamente.")
print(f"Registros totales: {len(df):,}")

# Campos de texto para minería de texto
columnas_texto = ["srcOrg", "srcCountryName", "protocol", "dstHostname"]
print(f"Campos de texto analizados: {columnas_texto}")


# ============================================================
# 3. FRECUENCIA DE TÉRMINOS (TEXT MINING / TF)
# ============================================================

print("\n" + "-" * 70)
print("TOP 10 ORGANIZACIONES / ISPs ATACANTES (srcOrg)")
print("-" * 70)

frecuencia_org = df["srcOrg"].value_counts().reset_index()
frecuencia_org.columns = ["Organizacion_ISP", "Frecuencia"]

print(frecuencia_org.head(10).to_string(index=False))

print("\n" + "-" * 70)
print("TOP 10 PAÍSES DE ORIGEN (srcCountryName)")
print("-" * 70)

frecuencia_pais = df["srcCountryName"].value_counts().reset_index()
frecuencia_pais.columns = ["Pais_Origen", "Frecuencia"]

print(frecuencia_pais.head(10).to_string(index=False))


# ============================================================
# 4. GENERACIÓN DE LA NUBE DE PALABRAS (WORD CLOUD)
# ============================================================

print("\nGenerando Nube de Palabras a partir de Organizaciones e ISPs...")

# Concatenar todos los términos de texto de srcOrg
texto_corpus = " ".join(df["srcOrg"].dropna().astype(str))

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="Dark2",
    max_words=100,
    collocations=False
).generate(texto_corpus)


# ============================================================
# 5. GUARDAR DATOS EN CSV
# ============================================================

frecuencia_org.to_csv(OUTPUT_DIR / "01_frecuencia_organizaciones.csv", index=False, encoding="utf-8-sig")
frecuencia_pais.to_csv(OUTPUT_DIR / "02_frecuencia_paises.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 6. GENERACIÓN DE GRÁFICAS (GUARDADO AUTOMÁTICO SIN SHOW)
# ============================================================

print("Guardando visualizaciones del análisis de texto...")

# GRÁFICA 1: Nube de Palabras de Organizaciones
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Nube de Palabras: Organizaciones e ISPs de Origen Malicioso (srcOrg)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_nube_de_palabras_org.png", dpi=150)
plt.close()


# GRÁFICA 2: Barplot Top 10 Organizaciones Atacantes
plt.figure(figsize=(10, 6))
top_orgs = frecuencia_org.head(10)
sns.barplot(data=top_orgs, y="Organizacion_ISP", x="Frecuencia", palette="viridis")
plt.title("Top 10 Organizaciones / ISPs de Origen de los Ataques", fontsize=13)
plt.xlabel("Cantidad de Eventos", fontsize=11)
plt.ylabel("Organización / ISP", fontsize=11)
plt.grid(axis="x", linestyle="--", alpha=0.7)

for index, value in enumerate(top_orgs["Frecuencia"]):
    plt.text(value + 100, index, f"{value:,}", va="center", weight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_top10_organizaciones_barras.png", dpi=150)
plt.close()


# GRÁFICA 3: Barplot Top 10 Países de Origen
plt.figure(figsize=(10, 6))
top_paises = frecuencia_pais.head(10)
sns.barplot(data=top_paises, y="Pais_Origen", x="Frecuencia", palette="rocket")
plt.title("Top 10 Países con mayor Volumen de Eventos Maliciosos", fontsize=13)
plt.xlabel("Cantidad de Eventos", fontsize=11)
plt.ylabel("País de Origen", fontsize=11)
plt.grid(axis="x", linestyle="--", alpha=0.7)

for index, value in enumerate(top_paises["Frecuencia"]):
    plt.text(value + 100, index, f"{value:,}", va="center", weight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_top10_paises_barras.png", dpi=150)
plt.close()


# GRÁFICA 4: Heatmap Relación Top 5 Organizaciones vs Tipo de Ataque
top5_orgs_nombres = frecuencia_org.head(5)["Organizacion_ISP"].tolist()
df_top_orgs = df[df["srcOrg"].isin(top5_orgs_nombres)]
cruce_org_ataque = pd.crosstab(df_top_orgs["srcOrg"], df_top_orgs["attackType"])

plt.figure(figsize=(10, 5))
sns.heatmap(cruce_org_ataque, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title("Relación entre las Top 5 Organizaciones Atacantes y el Tipo de Ataque", fontsize=13)
plt.xlabel("Tipo de Ataque (attackType)", fontsize=11)
plt.ylabel("Organización (srcOrg)", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_heatmap_org_vs_ataque.png", dpi=150)
plt.close()


# ============================================================
# 7. GENERAR REPORTE TXT PARA ENTREGA
# ============================================================

lineas = []

lineas.append("=" * 70)
lineas.append("REPORTE - SEMANA 14: ANÁLISIS DE TEXTO Y NUBE DE PALABRAS")
lineas.append("=" * 70)
lineas.append("")
lineas.append(f"Dataset: {CSV_PATH.name}")
lineas.append(f"Registros analizados: {len(df):,}")
lineas.append(f"Campos textuales procesados: {', '.join(columnas_texto)}")
lineas.append("")

lineas.append("-" * 70)
lineas.append("OBJETIVO DE NEGOCIO Y CIBERSEGURIDAD")
lineas.append("-" * 70)
lineas.append(
    "Analizar el contenido de texto no estructurado y categórico de los logs para "
    "identificar los principales proveedores de infraestructura (ISPs/Organizaciones) "
    "utilizados por los atacantes para desplegar tráfico malicioso."
)
lineas.append("")

lineas.append("-" * 70)
lineas.append("TOP 10 ORGANIZACIONES / ISPs CON MAYOR ACTIVIDAD MALICIOSA")
lineas.append("-" * 70)
lineas.append(frecuencia_org.head(10).to_string(index=False))
lineas.append("")

lineas.append("-" * 70)
lineas.append("TOP 10 PAÍSES DE ORIGEN")
lineas.append("-" * 70)
lineas.append(frecuencia_pais.head(10).to_string(index=False))
lineas.append("")

lineas.append("-" * 70)
lineas.append("INTERPRETACIÓN DE HALLAZGOS (EJE DE PRESENTACIÓN)")
lineas.append("-" * 70)
lineas.append(
    "El análisis exploratorio de texto y la Nube de Palabras revelan una alta "
    "concentración de eventos en proveedores de hosting y telecomunicaciones "
    "específicos (como VDI-NETWORK, OVH SAS y Tianfeng HK). "
    "Visualizar la prominencia de estos términos ayuda al equipo de ciberseguridad "
    "a identificar ASN/ISPs sospechosos y solicitar bloqueos preventivos de rangos de IP."
)
lineas.append("")

with open(OUTPUT_DIR / "07_reporte_semana14.txt", "w", encoding="utf-8") as archivo:
    archivo.write("\n".join(lineas))


# ============================================================
# 8. MENSAJE FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)
print(f"\nTodos los resultados se guardaron en:\n{OUTPUT_DIR}\n")
print("Archivos generados:")
for archivo in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {archivo.name}")