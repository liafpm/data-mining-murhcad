# ============================================================
# ESTADÍSTICA DESCRIPTIVA - HONEYNET EVENTS (SEMANA 5)
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. CONFIGURACIÓN Y RUTAS DE SALIDA
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "practica1" / "HoneyNetEvents_Clean.csv"
OUTPUT_DIR = BASE_DIR / "resultados" / "resultados_practica2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("SEMANA 5 - ESTADÍSTICA DESCRIPTIVA Y ENTIDADES")
print("=" * 70)
print(f"Registros cargados: {len(df):,}")

# ------------------------------------------------------------
# 2. INFORMACIÓN BÁSICA Y ESTADÍSTICA DESCRIPTIVA
# ------------------------------------------------------------
print("\n========== RESUMEN GENERAL ==========")
print(df.describe())

# ------------------------------------------------------------
# 3. FRECUENCIA DE ATAQUES Y GRAFICADO
# ------------------------------------------------------------
ataques = df["attackType"].value_counts()
print("\n========== FRECUENCIA POR TIPO DE ATAQUE ==========")
print(ataques)

plt.figure(figsize=(8, 5))
ataques.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Cantidad de eventos por tipo de ataque")
plt.xlabel("Tipo de ataque")
plt.ylabel("Cantidad de eventos")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_eventos_por_tipo_ataque.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 4. ENTIDADES Y RELACIONES MÁS FRECUENTES
# ------------------------------------------------------------
relaciones = (
    df.groupby(["srcIp", "dstIp", "attackType"])
    .size()
    .reset_index(name="cantidad")
    .sort_values("cantidad", ascending=False)
)

print("\n========== RELACIONES MÁS FRECUENTES (TOP 10) ==========")
print(relaciones.head(10).to_string(index=False))

# Diagrama visual de las 10 relaciones principales
top_relaciones = relaciones.head(10)
plt.figure(figsize=(14, 8))

for posicion, (_, fila) in enumerate(top_relaciones.iterrows()):
    origen = fila["srcIp"]
    destino = fila["dstIp"]
    cantidad = fila["cantidad"]
    y = len(top_relaciones) - posicion

    plt.annotate(
        "",
        xy=(1, y),
        xytext=(0, y),
        arrowprops=dict(arrowstyle="->", linewidth=1.5, color="navy")
    )
    plt.text(-0.05, y, origen, ha="right", va="center", fontsize=10, weight="bold")
    plt.text(1.05, y, destino, ha="left", va="center", fontsize=10, weight="bold")
    plt.text(0.5, y + 0.15, f"{cantidad:,} eventos ({fila['attackType']})", ha="center", va="bottom", fontsize=9)

plt.xlim(-0.8, 1.8)
plt.ylim(0, len(top_relaciones) + 1)
plt.yticks([])
plt.xticks([0, 1], ["ORIGEN (srcIp)", "DESTINO (dstIp)"], fontsize=12, weight="bold")
plt.title("Diagrama de las 10 relaciones de flujo más frecuentes", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_diagrama_relaciones_top10.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 5. MÉTRICAS DE DATOS AGRUPADOS POR ATAQUE Y HORA
# ------------------------------------------------------------
metricas = (
    df.groupby("attackType")
    .agg(
        cantidad=("attackType", "count"),
        promedio_hora=("hour", "mean"),
        mediana_hora=("hour", "median"),
        desviacion_hora=("hour", "std"),
        minimo_hora=("hour", "min"),
        maximo_hora=("hour", "max")
    )
)

print("\n========== MÉTRICAS AGRUPADAS POR TIPO DE ATAQUE ==========")
print(metricas)
metricas.to_csv(OUTPUT_DIR / "metricas_ataques.csv")

eventos_hora = df.groupby("hour").size()
plt.figure(figsize=(10, 5))
eventos_hora.plot(kind="bar", color="teal", edgecolor="black")
plt.title("Cantidad de eventos por hora del día")
plt.xlabel("Hora (0-23)")
plt.ylabel("Cantidad de eventos")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_eventos_por_hora.png", dpi=150)
plt.close()

print(f"\nProceso completado. Todos los gráficos e informes se guardaron en:\n{OUTPUT_DIR}")