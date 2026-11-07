"""
Semana 9 - Modelos Lineales y Correlación

Objetivo:
Generar un modelo lineal entre la latitud de origen (srcLat) y 
la latitud de destino (dstLat), crear las gráficas correspondientes
y obtener la métrica R^2.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "practica1" / "HoneyNetEvents_Clean.csv"

# Carpeta donde se guardarán todos los resultados.
OUTPUT_DIR = BASE_DIR / "resultados" / "resultados_practica5"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. CARGAR DATASET
# ============================================================

print("=" * 70)
print("SEMANA 9 - MODELOS LINEALES Y CORRELACIÓN")
print("=" * 70)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nNo se encontró el archivo CSV:\n{CSV_PATH}\n"
    )

df = pd.read_csv(CSV_PATH)

print("\nDataset cargado correctamente.")
print(f"Registros: {len(df):,}")
print(f"Columnas: {len(df.columns)}")


# ============================================================
# 3. VALIDAR COLUMNAS
# ============================================================

columnas_necesarias = ["srcLat", "dstLat"]

faltantes = [
    col for col in columnas_necesarias
    if col not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan las siguientes columnas en el CSV: {faltantes}"
    )


# ============================================================
# 4. PREPARAR DATOS
# ============================================================

datos = df[["srcLat", "dstLat"]].copy()

# Convertir las variables a numéricas por seguridad.
datos["srcLat"] = pd.to_numeric(
    datos["srcLat"],
    errors="coerce"
)

datos["dstLat"] = pd.to_numeric(
    datos["dstLat"],
    errors="coerce"
)

# Eliminar filas que no tengan alguno de los valores.
datos = datos.dropna(
    subset=["srcLat", "dstLat"]
)

print(f"\nRegistros utilizados para el modelo: {len(datos):,}")


# ============================================================
# 5. MODELO LINEAL Y CORRELACIÓN
# ============================================================

print("\n" + "-" * 70)
print("MODELO LINEAL Y CORRELACIÓN")
print("-" * 70)

x = datos["srcLat"]
y = datos["dstLat"]

# Regresión lineal.
modelo = stats.linregress(x, y)

pendiente = modelo.slope
intercepto = modelo.intercept
correlacion = modelo.rvalue
p_valor = modelo.pvalue

# R^2 se obtiene elevando al cuadrado el coeficiente de correlación.
r2 = correlacion ** 2

print("\nVariable independiente (X): srcLat")
print("Variable dependiente (Y): dstLat")

print(f"\nPendiente: {pendiente:.6f}")
print(f"Intercepto: {intercepto:.6f}")
print(f"Correlación de Pearson (r): {correlacion:.6f}")
print(f"R^2: {r2:.6f}")
print(f"p-valor: {p_valor:.6g}")

print(
    f"\nEcuación del modelo:"
    f" dstLat = {intercepto:.4f} "
    f"+ ({pendiente:.4f} * srcLat)"
)


# ============================================================
# 6. GUARDAR RESULTADOS
# ============================================================

resultados = pd.DataFrame(
    [
        {
            "variable_independiente": "srcLat",
            "variable_dependiente": "dstLat",
            "pendiente": pendiente,
            "intercepto": intercepto,
            "correlacion_pearson": correlacion,
            "R2": r2,
            "p_valor": p_valor,
            "registros": len(datos),
        }
    ]
)

resultados.to_csv(
    OUTPUT_DIR / "01_modelo_lineal.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 7. GRÁFICA DE REGRESIÓN LINEAL
# ============================================================

print("\nGenerando gráfica de regresión lineal...")

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    alpha=0.15,
    s=8,
    label="Datos"
)

# Dos puntos son suficientes para dibujar la recta.
x_linea = pd.Series(
    [x.min(), x.max()]
)

y_linea = intercepto + pendiente * x_linea

plt.plot(
    x_linea,
    y_linea,
    linewidth=2,
    color="red",
    label=f"Regresión lineal (R² = {r2:.4f})"
)

plt.xlabel("Latitud de origen (srcLat)")
plt.ylabel("Latitud de destino (dstLat)")
plt.title("Modelo lineal: Latitud Origen vs Latitud Destino")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_regresion_lineal.png",
    dpi=150
)

plt.close()


# ============================================================
# 8. GRÁFICA DE RESIDUALES
# ============================================================

print("Generando gráfica de residuales...")

valores_predichos = intercepto + pendiente * x
residuales = y - valores_predichos

plt.figure(figsize=(10, 6))

plt.scatter(
    valores_predichos,
    residuales,
    alpha=0.15,
    s=8
)

plt.axhline(
    0,
    color="red",
    linewidth=2
)

plt.xlabel("Valores predichos")
plt.ylabel("Residuales")
plt.title("Residuales del modelo lineal (srcLat vs dstLat)")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_residuales.png",
    dpi=150
)

plt.close()


# ============================================================
# 9. INTERPRETACIÓN
# ============================================================

if abs(correlacion) < 0.3:
    fuerza = "débil"
elif abs(correlacion) < 0.7:
    fuerza = "moderada"
else:
    fuerza = "fuerte"

if correlacion > 0:
    direccion = "positiva"
elif correlacion < 0:
    direccion = "negativa"
else:
    direccion = "nula"

conclusion = (
    f"La correlación entre srcLat y dstLat es {direccion} y "
    f"{fuerza} (r = {correlacion:.4f}). "
    f"El modelo lineal obtiene un R^2 de {r2:.4f}, lo que indica que "
    f"la ubicación geográfica de los atacantes no mantiene una relación lineal "
    f"directa con la ubicación del servidor destino, evidenciando una amenaza global y distribuida."
)

print("\n" + "-" * 70)
print("INTERPRETACIÓN")
print("-" * 70)
print(conclusion)


conclusion_df = pd.DataFrame(
    [
        {
            "correlacion": correlacion,
            "R2": r2,
            "conclusion": conclusion,
        }
    ]
)

conclusion_df.to_csv(
    OUTPUT_DIR / "04_conclusion.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. REPORTE TXT
# ============================================================

lineas = []

lineas.append("=" * 70)
lineas.append("REPORTE - SEMANA 9: MODELOS LINEALES Y CORRELACIÓN")
lineas.append("=" * 70)
lineas.append("")
lineas.append(f"Dataset: {CSV_PATH.name}")
lineas.append(f"Registros originales: {len(df):,}")
lineas.append(f"Registros analizados: {len(datos):,}")
lineas.append("")

lineas.append("-" * 70)
lineas.append("OBJETIVO")
lineas.append("-" * 70)
lineas.append(
    "Generar un modelo lineal entre srcLat y dstLat, "
    "crear las gráficas correspondientes y obtener R^2."
)
lineas.append("")

lineas.append("-" * 70)
lineas.append("MODELO")
lineas.append("-" * 70)
lineas.append("Variable independiente: srcLat")
lineas.append("Variable dependiente: dstLat")
lineas.append(
    f"Ecuación: dstLat = {intercepto:.6f} "
    f"+ ({pendiente:.6f} * srcLat)"
)
lineas.append("")

lineas.append("-" * 70)
lineas.append("RESULTADOS")
lineas.append("-" * 70)
lineas.append(f"Correlación de Pearson (r): {correlacion:.6f}")
lineas.append(f"R^2: {r2:.6f}")
lineas.append(f"p-valor: {p_valor:.6g}")
lineas.append("")

lineas.append("-" * 70)
lineas.append("CONCLUSIÓN")
lineas.append("-" * 70)
lineas.append(conclusion)

with open(
    OUTPUT_DIR / "05_reporte_semana9.txt",
    "w",
    encoding="utf-8"
) as archivo:
    archivo.write("\n".join(lineas))


# ============================================================
# 11. MENSAJE FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)

print(f"\nTodos los resultados fueron guardados en:")
print(OUTPUT_DIR)

print("\nArchivos generados:")

for archivo in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {archivo.name}")

print(
    "\nListo. Puedes revisar las gráficas y los resultados "
    "generados para la práctica."
)