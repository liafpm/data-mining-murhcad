"""
Semana 7 - Pruebas Estadísticas

Objetivo:
Comprobar si existen diferencias estadísticamente significativas
en el puerto de origen (srcPort) entre los diferentes tipos de
ataque (attackType), utilizando:

1. ANOVA de una vía.
2. Pruebas t de Welch por pares, con corrección de Bonferroni.

"""

from pathlib import Path
from itertools import combinations

import pandas as pd
from scipy import stats


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "practica1" / "raw" / "HoneyNetEvents_Clean.csv"

# Carpeta donde se guardarán todos los resultados.
OUTPUT_DIR = BASE_DIR / "resultados_practica4"
OUTPUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.05


# ============================================================
# 2. CARGAR DATASET
# ============================================================

print("=" * 70)
print("SEMANA 7 - PRUEBAS ESTADÍSTICAS")
print("=" * 70)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nNo se encontró el archivo CSV:\n{CSV_PATH}\n\n"            
    )

df = pd.read_csv(CSV_PATH)

print(f"\nDataset cargado correctamente.")
print(f"Registros: {len(df):,}")
print(f"Columnas: {len(df.columns)}")


# ============================================================
# 3. VALIDAR COLUMNAS
# ============================================================

columnas_necesarias = ["attackType", "srcPort"]
faltantes = [col for col in columnas_necesarias if col not in df.columns]

if faltantes:
    raise ValueError(
        f"Faltan las siguientes columnas en el CSV: {faltantes}"
    )


# ============================================================
# 4. PREPARAR DATOS
# ============================================================

datos = df[["attackType", "srcPort"]].copy()

# Convertir srcPort a numérico por seguridad.
datos["srcPort"] = pd.to_numeric(datos["srcPort"], errors="coerce")

# Eliminar filas sin attackType o srcPort.
datos = datos.dropna(subset=["attackType", "srcPort"])

# Eliminar categorías vacías.
datos["attackType"] = datos["attackType"].astype(str).str.strip()
datos = datos[datos["attackType"] != ""]

print(f"\nRegistros utilizados para las pruebas: {len(datos):,}")

grupos = {}

for attack_type, grupo in datos.groupby("attackType"):
    valores = grupo["srcPort"].dropna()

    # Para una prueba t se necesitan al menos 2 observaciones.
    if len(valores) >= 2:
        grupos[attack_type] = valores

print("\nGrupos encontrados:")
for nombre, valores in grupos.items():
    print(f"  - {nombre}: {len(valores):,} observaciones")


# ============================================================
# 5. ESTADÍSTICA DESCRIPTIVA POR GRUPO
# ============================================================

resumen = (
    datos.groupby("attackType")["srcPort"]
    .agg(
        cantidad="count",
        media="mean",
        mediana="median",
        desviacion_estandar="std",
        minimo="min",
        maximo="max",
    )
    .reset_index()
)

resumen = resumen.sort_values("attackType")

resumen.to_csv(
    OUTPUT_DIR / "01_resumen_descriptivo.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "-" * 70)
print("RESUMEN DESCRIPTIVO")
print("-" * 70)
print(resumen.to_string(index=False))


# ============================================================
# 6. HIPÓTESIS DEL ANOVA
# ============================================================

print("\n" + "-" * 70)
print("ANOVA DE UNA VÍA")
print("-" * 70)

print("\nH0: Las medias de srcPort son iguales entre los tipos de ataque.")
print("H1: Al menos una media de srcPort es diferente.")
print(f"Nivel de significancia: α = {ALPHA}")


# ============================================================
# 7. EJECUTAR ANOVA
# ============================================================

if len(grupos) < 2:
    raise ValueError(
        "Se necesitan al menos dos grupos para ejecutar ANOVA."
    )

nombres_grupos = list(grupos.keys())
valores_grupos = list(grupos.values())

f_stat, p_anova = stats.f_oneway(*valores_grupos)

anova_resultado = pd.DataFrame(
    [
        {
            "prueba": "ANOVA de una vía",
            "variable_numerica": "srcPort",
            "variable_categorica": "attackType",
            "F": f_stat,
            "p_valor": p_anova,
            "alpha": ALPHA,
            "decision": (
                "Rechazar H0"
                if p_anova < ALPHA
                else "No rechazar H0"
            ),
            "resultado": (
                "Diferencia estadísticamente significativa"
                if p_anova < ALPHA
                else "No se encontró diferencia estadísticamente significativa"
            ),
        }
    ]
)

anova_resultado.to_csv(
    OUTPUT_DIR / "02_resultado_anova.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"\nF = {f_stat:.6f}")
print(f"p = {p_anova:.10g}")
print(f"α = {ALPHA}")

if p_anova < ALPHA:
    print("\nResultado: se rechaza H0.")
    print(
        "Existe evidencia estadísticamente significativa de que "
        "no todas las medias de srcPort son iguales."
    )
else:
    print("\nResultado: no se rechaza H0.")
    print(
        "No existe evidencia estadísticamente significativa suficiente "
        "para afirmar que las medias sean diferentes."
    )


# ============================================================
# 8. PRUEBAS T DE WELCH POR PARES
# ============================================================

print("\n" + "-" * 70)
print("PRUEBAS T DE WELCH POR PARES")
print("-" * 70)

pares = list(combinations(nombres_grupos, 2))
numero_comparaciones = len(pares)

resultados_t = []

for grupo_a, grupo_b in pares:
    valores_a = grupos[grupo_a]
    valores_b = grupos[grupo_b]

    # Welch: no supone varianzas iguales.
    t_stat, p_original = stats.ttest_ind(
        valores_a,
        valores_b,
        equal_var=False,
    )

    # Corrección de Bonferroni.
    p_ajustado = min(
        p_original * numero_comparaciones,
        1.0,
    )

    significativo = p_ajustado < ALPHA

    resultados_t.append(
        {
            "grupo_1": grupo_a,
            "grupo_2": grupo_b,
            "t": t_stat,
            "p_valor_original": p_original,
            "p_valor_bonferroni": p_ajustado,
            "alpha": ALPHA,
            "significativo": "Sí" if significativo else "No",
        }
    )

resultados_t_df = pd.DataFrame(resultados_t)

resultados_t_df.to_csv(
    OUTPUT_DIR / "03_pruebas_t_welch_bonferroni.csv",
    index=False,
    encoding="utf-8-sig",
)

print(resultados_t_df.to_string(index=False))


# ============================================================
# 9. INTERPRETACIÓN AUTOMÁTICA DE LAS PRUEBAS T
# ============================================================

interpretaciones = []

for resultado in resultados_t:
    if resultado["p_valor_bonferroni"] < ALPHA:
        texto = (
            f"Existe una diferencia estadísticamente significativa "
            f"entre {resultado['grupo_1']} y {resultado['grupo_2']}."
        )
    else:
        texto = (
            f"No se encontró una diferencia estadísticamente significativa "
            f"entre {resultado['grupo_1']} y {resultado['grupo_2']}."
        )

    interpretaciones.append(
        {
            "grupo_1": resultado["grupo_1"],
            "grupo_2": resultado["grupo_2"],
            "interpretacion": texto,
        }
    )

interpretaciones_df = pd.DataFrame(interpretaciones)

interpretaciones_df.to_csv(
    OUTPUT_DIR / "04_interpretaciones_pares.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. CONCLUSIÓN AUTOMÁTICA
# ============================================================

if p_anova < ALPHA:
    conclusion = (
        "El ANOVA de una vía mostró una diferencia estadísticamente "
        "significativa en srcPort entre los tipos de ataque "
        f"(F = {f_stat:.2f}, p < 0.001). "
        "Por lo tanto, se rechaza la hipótesis nula de igualdad "
        "de medias. Las pruebas t de Welch con corrección de "
        "Bonferroni permiten identificar qué pares de grupos "
        "presentan diferencias estadísticamente significativas."
    )
else:
    conclusion = (
        "El ANOVA de una vía no mostró una diferencia estadísticamente "
        f"significativa en srcPort entre los tipos de ataque "
        f"(F = {f_stat:.2f}, p = {p_anova:.6g}). "
        "Por lo tanto, no se rechaza la hipótesis nula."
    )

conclusion_df = pd.DataFrame(
    [
        {
            "alpha": ALPHA,
            "F_anova": f_stat,
            "p_anova": p_anova,
            "conclusion": conclusion,
        }
    ]
)

conclusion_df.to_csv(
    OUTPUT_DIR / "05_conclusion.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 11. GENERAR REPORTE TXT
# ============================================================

lineas = []

lineas.append("=" * 70)
lineas.append("REPORTE - SEMANA 7: PRUEBAS ESTADÍSTICAS")
lineas.append("=" * 70)
lineas.append("")
lineas.append(f"Dataset: {CSV_PATH.name}")
lineas.append(f"Registros originales: {len(df):,}")
lineas.append(f"Registros analizados: {len(datos):,}")
lineas.append(f"Nivel de significancia: α = {ALPHA}")
lineas.append("")

lineas.append("-" * 70)
lineas.append("OBJETIVO")
lineas.append("-" * 70)
lineas.append(
    "Comprobar si existen diferencias estadísticamente significativas "
    "en srcPort entre los diferentes valores de attackType."
)
lineas.append("")

lineas.append("-" * 70)
lineas.append("HIPÓTESIS")
lineas.append("-" * 70)
lineas.append(
    "H0: Las medias de srcPort son iguales entre los tipos de ataque."
)
lineas.append(
    "H1: Al menos una media de srcPort es diferente."
)
lineas.append("")

lineas.append("-" * 70)
lineas.append("ANOVA DE UNA VÍA")
lineas.append("-" * 70)
lineas.append(f"F = {f_stat:.6f}")
lineas.append(f"p = {p_anova:.10g}")
lineas.append(
    "Decisión: "
    + ("Rechazar H0" if p_anova < ALPHA else "No rechazar H0")
)
lineas.append("")
lineas.append(conclusion)
lineas.append("")

lineas.append("-" * 70)
lineas.append("PRUEBAS T DE WELCH + BONFERRONI")
lineas.append("-" * 70)

for resultado in resultados_t:
    lineas.append(
        f"{resultado['grupo_1']} vs {resultado['grupo_2']}: "
        f"t = {resultado['t']:.6f}, "
        f"p original = {resultado['p_valor_original']:.10g}, "
        f"p Bonferroni = {resultado['p_valor_bonferroni']:.10g}, "
        f"significativo = {resultado['significativo']}"
    )

lineas.append("")
lineas.append("-" * 70)
lineas.append("RESUMEN DESCRIPTIVO")
lineas.append("-" * 70)
lineas.append(resumen.to_string(index=False))
lineas.append("")

with open(
    OUTPUT_DIR / "06_reporte_semana7.txt",
    "w",
    encoding="utf-8",
) as archivo:
    archivo.write("\n".join(lineas))


# ============================================================
# 12. MENSAJE FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)

print(f"\nTodos los resultados fueron guardados en:")
print(OUTPUT_DIR)

print("\nArchivos generados:")

for archivo in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {archivo.name}")

print("\nListo. Puedes revisar los archivos generados para ver los resultados de las pruebas estadísticas.")
