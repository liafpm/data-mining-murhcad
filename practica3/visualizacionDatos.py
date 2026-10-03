# ============================================================
# VISUALIZACIÓN DE DATOS - HONEYNET EVENTS
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CARGAR EL ARCHIVO CSV
# ============================================================

csv_path = Path(__file__).resolve().parent.parent / "practica1" / "raw" / "HoneyNetEvents_Clean.csv"
df = pd.read_csv(csv_path)

print("Datos cargados correctamente.")
print("Número de filas:", len(df))
print("Número de columnas:", len(df.columns))


# ============================================================
# 2. REVISAR LA INFORMACIÓN DEL DATASET
# ============================================================

print("\n========== COLUMNAS ==========")
print(df.columns.tolist())

print("\n========== TIPOS DE DATOS ==========")
print(df.dtypes)

print("\n========== VALORES FALTANTES ==========")
print(df.isnull().sum())

# ============================================================
# 3. VISUALIZACIÓN DE DATOS - SEMANA 6
# ============================================================

print("\n========== VISUALIZACIÓN DE DATOS - SEMANA 6 ==========")


# ------------------------------------------------------------
# GRÁFICA 1 - DIAGRAMA DE PASTEL
# Distribución de eventos por tipo de ataque
# ------------------------------------------------------------

ataques = df["attackType"].value_counts()

plt.figure(figsize=(8, 6))
ataques.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Distribución de eventos por tipo de ataque")
plt.ylabel("")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRÁFICA 2 - HISTOGRAMA
# Distribución de los puertos de origen
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df["srcPort"].dropna(),
    bins=30,
    edgecolor="black"
)

plt.title("Distribución de puertos de origen")
plt.xlabel("Puerto de origen")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRÁFICA 3 - DIAGRAMA DE CAJA
# Distribución de puertos según tipo de ataque
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df.boxplot(
    column="srcPort",
    by="attackType"
)

plt.title("Distribución de puertos por tipo de ataque")
plt.suptitle("")
plt.xlabel("Tipo de ataque")
plt.ylabel("Puerto de origen")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRÁFICA 4 - DIAGRAMA DE DISPERSIÓN
# Ubicación geográfica de las IP de origen
# ------------------------------------------------------------

datos_geo = df[["srcLon", "srcLat"]].dropna()

plt.figure(figsize=(10, 6))

plt.scatter(
    datos_geo["srcLon"],
    datos_geo["srcLat"],
    alpha=0.2,
    s=10
)

plt.title("Ubicación geográfica de las IP de origen")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# GRÁFICA 5 - GRÁFICA DE LÍNEA
# Cantidad de eventos por hora
# ------------------------------------------------------------

eventos_hora = df.groupby("hour").size()

plt.figure(figsize=(10, 5))

plt.plot(
    eventos_hora.index,
    eventos_hora.values,
    marker="o"
)

plt.title("Cantidad de eventos por hora")
plt.xlabel("Hora del día")
plt.ylabel("Cantidad de eventos")
plt.xticks(range(24))
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 4. AUTOMATIZACIÓN CON CICLOS
# ============================================================

# Generamos automáticamente gráficas de barras para
# diferentes variables categóricas.

columnas_categoricas = [
    "attackType",
    "protocol",
    "weekday"
]

titulos = [
    "Eventos por tipo de ataque",
    "Eventos por protocolo",
    "Eventos por día de la semana"
]


for columna, titulo in zip(columnas_categoricas, titulos):

    datos = df[columna].value_counts().head(10)

    plt.figure(figsize=(10, 5))

    datos.plot(kind="bar")

    plt.title(titulo)
    plt.xlabel(columna)
    plt.ylabel("Cantidad de eventos")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


print("\nVisualizaciones de Semana 6 generadas correctamente.")


# ============================================================
# FIN
# ============================================================

print("\n========== ANÁLISIS TERMINADO ==========")