# ============================================================
# VISUALIZACIÓN DE DATOS AUTOMATIZADA - HONEYNET (SEMANA 6)
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. CONFIGURACIÓN Y RUTAS DE SALIDA
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "practica1" / "HoneyNetEvents_Clean.csv"
OUTPUT_DIR = BASE_DIR / "resultados" / "resultados_practica3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("SEMANA 6 - VISUALIZACIÓN DE DATOS Y AUTOMATIZACIÓN")
print("=" * 70)
print(f"Registros analizados: {len(df):,}")

# ------------------------------------------------------------
# 2. GENERACIÓN DE 5 TIPOS DE GRÁFICAS DISTINTAS
# ------------------------------------------------------------

# GRÁFICA 1: DIAGRAMA DE PASTEL (Distribución de Amenazas)
plt.figure(figsize=(8, 6))
df["attackType"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=140,
    colors=["#ff9999", "#66b3ff", "#99ff99"]
)
plt.title("1. Distribución Porcentual de Eventos por Tipo de Ataque")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_pastel_tipo_ataque.png", dpi=150)
plt.close()


# GRÁFICA 2: HISTOGRAMA (Distribución Horaria del Tráfico Malicioso)
plt.figure(figsize=(10, 5))
plt.hist(
    df["hour"],
    bins=24,
    color="steelblue",
    edgecolor="black",
    alpha=0.7
)
plt.title("2. Histograma de Frecuencia Horaria de Ataques")
plt.xlabel("Hora del día (0-23)")
plt.ylabel("Frecuencia de eventos")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_histograma_horas.png", dpi=150)
plt.close()


# GRÁFICA 3: DIAGRAMA DE CAJA / BOXPLOT (Distribución de Hora por Tipo de Ataque)
plt.figure(figsize=(10, 6))
df.boxplot(
    column="hour",
    by="attackType",
    patch_artist=True,
    grid=True
)
plt.title("3. Distribución Horaria (Boxplot) según Tipo de Ataque")
plt.suptitle("")
plt.xlabel("Tipo de Ataque")
plt.ylabel("Hora del día")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_boxplot_hora_por_ataque.png", dpi=150)
plt.close()


# GRÁFICA 4: DIAGRAMA DE DISPERSIÓN / SCATTER (Mapa Geográfico de Orígenes)
datos_geo = df[["srcLon", "srcLat"]].dropna()
plt.figure(figsize=(10, 6))
plt.scatter(
    datos_geo["srcLon"],
    datos_geo["srcLat"],
    alpha=0.2,
    s=12,
    c="crimson"
)
plt.title("4. Dispersión Geográfica de IP de Origen (Longitud vs Latitud)")
plt.xlabel("Longitud (srcLon)")
plt.ylabel("Latitud (srcLat)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_dispersion_geografica.png", dpi=150)
plt.close()


# GRÁFICA 5: GRÁFICA DE LÍNEA Y ÁREA (Tendencia Temporal de Eventos por Hora)
eventos_hora = df.groupby("hour").size()
plt.figure(figsize=(10, 5))
plt.plot(
    eventos_hora.index,
    eventos_hora.values,
    marker="o",
    color="purple",
    linewidth=2
)
plt.fill_between(eventos_hora.index, eventos_hora.values, color="purple", alpha=0.1)
plt.title("5. Tendencia Temporal de Eventos a lo largo de las 24 Horas")
plt.xlabel("Hora del día")
plt.ylabel("Volumen de Ataques")
plt.xticks(range(24))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_linea_tendencia_hora.png", dpi=150)
plt.close()


# ------------------------------------------------------------
# 3. AUTOMATIZACIÓN CON CICLOS (REQUISITO SEMANA 6)
# ------------------------------------------------------------
# Generamos automáticamente gráficas de barras para el Top 10 de
# variables categóricas clave usando un ciclo FOR.

variables_analizar = [
    ("attackType", "Eventos por Tipo de Ataque"),
    ("protocol", "Top 10 Protocolos de Red Atacados"),
    ("weekday", "Distribución de Ataques por Día de la Semana"),
    ("srcCountryName", "Top 10 Países de Origen de los Ataques")
]

print("\nGenerando gráficas automatizadas en ciclo...")

for idx, (columna, titulo) in enumerate(variables_analizar, start=6):
    datos = df[columna].value_counts().head(10)
    
    plt.figure(figsize=(10, 5))
    datos.plot(kind="bar", color="darkcyan", edgecolor="black")
    plt.title(f"{idx}. Automatizado: {titulo}")
    plt.xlabel(columna)
    plt.ylabel("Cantidad de Eventos")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # Guardado automático por variable
    plt.savefig(OUTPUT_DIR / f"{idx:02d}_auto_barras_{columna}.png", dpi=150)
    plt.close()

print(f"\n¡Éxito! Todas las 9 gráficas de la Semana 6 se guardaron automáticamente en:\n{OUTPUT_DIR}")