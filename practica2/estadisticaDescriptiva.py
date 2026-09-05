# ============================================================
# ESTADÍSTICA DESCRIPTIVA - HONEYNET EVENTS
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
# 3. ESTADÍSTICA DESCRIPTIVA
# ============================================================


print("\n========== ESTADÍSTICA DESCRIPTIVA ==========")
print(df.describe())


# ============================================================
# 4. FRECUENCIA DE LOS TIPOS DE ATAQUE
# ============================================================

print("\n========== TIPOS DE ATAQUE ==========")

ataques = df["attackType"].value_counts()

print(ataques)


# GRÁFICA DE BARRAS
ataques.plot(kind="bar")

plt.title("Cantidad de eventos por tipo de ataque")
plt.xlabel("Tipo de ataque")
plt.ylabel("Cantidad de eventos")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 5. IDENTIFICAR ENTIDADES Y RELACIONES
# ============================================================

# En este dataset podemos identificar principalmente:

# ENTIDAD 1 = ORIGEN
# srcIp = IP que genera el evento

# ENTIDAD 2 = DESTINO
# dstIp = IP que recibe el evento

# ENTIDAD 3 = EVENTO
# attackType = tipo de ataque

# La relación sería:
#
# ORIGEN ----> EVENTO ----> DESTINO
#
# Por ejemplo:
# una IP de origen realiza un ataque
# contra una IP de destino.


print("\n========== ENTIDADES ==========")

print("Entidad ORIGEN:")
print("- srcIp")
print("- srcCountryName")
print("- srcOrg")

print("\nEntidad EVENTO:")
print("- attackType")
print("- protocol")
print("- timestamp")

print("\nEntidad DESTINO:")
print("- dstIp")
print("- dstPort")
print("- dstHostname")


# ============================================================
# 6. MOSTRAR LAS RELACIONES MÁS FRECUENTES
# ============================================================

# Agrupamos IP de origen, IP de destino y tipo de ataque.
# Esto nos dice qué combinaciones aparecen más veces.

relaciones = (
    df.groupby(["srcIp", "dstIp", "attackType"])
    .size()
    .reset_index(name="cantidad")
    .sort_values("cantidad", ascending=False)
)

print("\n========== RELACIONES MÁS FRECUENTES ==========")
print(relaciones.head(10))


# ============================================================
# 7. DIAGRAMA DE RELACIONES
# ============================================================

# 10 relaciones más frecuentes
top_relaciones = relaciones.head(10)

plt.figure(figsize=(16, 10))

for posicion, (_, fila) in enumerate(top_relaciones.iterrows()):

    origen = fila["srcIp"]
    destino = fila["dstIp"]
    cantidad = fila["cantidad"]

    y = len(top_relaciones) - posicion

    # ORIGEN ----------> DESTINO
    plt.annotate(
        "",
        xy=(1, y),
        xytext=(0, y),
        arrowprops=dict(arrowstyle="->", linewidth=1.5)
    )

    # IP de origen
    plt.text(
        -0.05,
        y,
        origen,
        ha="right",
        va="center",
        fontsize=10
    )

    # IP de destino
    plt.text(
        1.05,
        y,
        destino,
        ha="left",
        va="center",
        fontsize=10
    )

    # Cuántos eventos existen
    plt.text(
        0.5,
        y + 0.15,
        f"{cantidad:,} eventos",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xlim(-0.8, 1.8)
plt.ylim(0, len(top_relaciones) + 1)

plt.yticks([])

plt.xticks(
    [0, 1],
    ["ORIGEN (srcIp)", "DESTINO (dstIp)"]
)

plt.title(
    "Diagrama de las 10 relaciones más frecuentes",
    fontsize=16
)
plt.subplots_adjust(
    left=0.25,
    right=0.75,
    top=0.90,
    bottom=0.10
)

plt.show()

# ============================================================
# 8. MÉTRICAS DE DATOS AGRUPADOS
# ============================================================

# Aquí agrupamos los datos por tipo de ataque.

# Calculamos:
# - cantidad de registros
# - promedio del puerto de origen
# - mediana
# - desviación estándar
# - mínimo
# - máximo

metricas = (
    df.groupby("attackType")
    .agg(
        cantidad=("attackType", "count"),
        promedio_srcPort=("srcPort", "mean"),
        mediana_srcPort=("srcPort", "median"),
        desviacion_srcPort=("srcPort", "std"),
        minimo_srcPort=("srcPort", "min"),
        maximo_srcPort=("srcPort", "max")
    )
)

print("\n========== MÉTRICAS AGRUPADAS POR TIPO DE ATAQUE ==========")
print(metricas)


# ============================================================
# 9. MÉTRICAS AGRUPADAS POR HORA
# ============================================================

# También podemos saber cuántos eventos ocurren
# durante cada hora del día.

eventos_hora = df.groupby("hour").size()

print("\n========== EVENTOS POR HORA ==========")
print(eventos_hora)


# GRÁFICA
eventos_hora.plot(kind="bar", figsize=(10, 5))

plt.title("Cantidad de eventos por hora")
plt.xlabel("Hora")
plt.ylabel("Cantidad de eventos")

plt.tight_layout()
plt.show()


# ============================================================
# 10. GUARDAR LAS MÉTRICAS EN UN CSV
# ============================================================

metricas.to_csv("metricas_ataques.csv")

print("\nArchivo 'metricas_ataques.csv' creado correctamente.")


# ============================================================
# FIN
# ============================================================

print("\n========== ANÁLISIS TERMINADO ==========")