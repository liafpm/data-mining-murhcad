
import os
import subprocess
import sys


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))


# Prácticas actualmente disponibles
PRACTICAS_DISPONIBLES = [2, 3, 4]

# ============================================================
# FUNCIONES
# ============================================================

def encontrar_archivo_python(numero_practica):
    """
    Busca automáticamente el único archivo .py dentro
    de la carpeta practica#.
    """

    carpeta = os.path.join(
        CARPETA_BASE,
        f"practica{numero_practica}"
    )

    if not os.path.isdir(carpeta):
        print(f"\n[!] No existe la carpeta: practica{numero_practica}")
        return None

    archivos_python = [
        archivo
        for archivo in os.listdir(carpeta)
        if archivo.endswith(".py")
        and archivo != "__init__.py"
    ]

    if not archivos_python:
        print(
            f"\n[!] No se encontró ningún archivo .py "
            f"en practica{numero_practica}"
        )
        return None

    if len(archivos_python) > 1:
        print(
            f"\n[!] Hay más de un archivo .py "
            f"en practica{numero_practica}:"
        )

        for archivo in archivos_python:
            print(f"    - {archivo}")

        print(
            "\n    Se esperaba solamente un archivo .py "
            "por práctica."
        )

        return None

    return os.path.join(carpeta, archivos_python[0])


def ejecutar_practica(numero_practica):
    """
    Ejecuta una práctica y devuelve True si terminó
    correctamente, o False si ocurrió un error.
    """

    archivo = encontrar_archivo_python(numero_practica)

    if archivo is None:
        return False

    print("\n" + "=" * 65)
    print(f"  EJECUTANDO PRACTICA {numero_practica}")
    print("=" * 65)
    print(f"  Archivo: {os.path.basename(archivo)}")
    print(f"  Carpeta: practica{numero_practica}")
    print("=" * 65)
    print()

    resultado = subprocess.run(
        [sys.executable, archivo],
        cwd=os.path.dirname(archivo)
    )

    print("\n" + "=" * 65)

    if resultado.returncode == 0:
        print(
            f"  ✓ Practica {numero_practica} "
            "terminada correctamente."
        )
        print("=" * 65)
        return True

    print(
        f"  ✗ Practica {numero_practica} "
        f"terminó con código de error: "
        f"{resultado.returncode}"
    )
    print("=" * 65)

    return False


def ejecutar_todas():
    """
    Ejecuta todas las prácticas disponibles en orden.

    Si una práctica falla, se detiene la ejecución para
    evitar continuar con resultados posiblemente incorrectos.
    """

    print("\n" + "#" * 65)
    print("              EJECUTANDO TODAS LAS PRACTICAS")
    print("#" * 65)

    for numero_practica in PRACTICAS_DISPONIBLES:

        print(
            f"\n>>> Iniciando practica "
            f"{numero_practica}..."
        )

        resultado = ejecutar_practica(numero_practica)

        if not resultado:
            print("\n" + "!" * 65)
            print(
                f"  EJECUCION DETENIDA EN PRACTICA "
                f"{numero_practica}"
            )
            print(
                "  Corrige el error antes de continuar."
            )
            print("!" * 65)
            return

    print("\n" + "#" * 65)
    print("       ✓ TODAS LAS PRACTICAS TERMINARON")
    print("#" * 65)


# ============================================================
# MENÚ
# ============================================================

def mostrar_menu():

    print("\n")
    print("=" * 65)
    print("                 PRACTICAS DE DATOS")
    print("=" * 65)

    print("\nPracticas disponibles:\n")

    print("  2. Estadistica Descriptiva")
    print("  3. Visualizacion de Datos")
    print("  4. Pruebas Estadisticas")

    # ========================================================
    # PRÓXIMAS PRÁCTICAS
    # ========================================================

    # print("  5. Modelos Lineales y Correlacion")
    # print("  6. Clasificacion de Datos")
    # print("  7. Agrupamiento de Datos (Clustering)")
    # print("  8. Pronostico (Forecasting)")
    # print("  9. Analisis de Texto")

    print("\n  A. Ejecutar todas")
    print("  0. Salir")

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():

    while True:

        mostrar_menu()

        opcion = input("\nSelecciona una opcion: ").strip().lower()

        # ----------------------------------------------------
        # SALIR
        # ----------------------------------------------------

        if opcion == "0":
            print("\nSaliendo del programa...")
            break

        # ----------------------------------------------------
        # EJECUTAR TODAS
        # ----------------------------------------------------

        elif opcion == "a":
            ejecutar_todas()

        # ----------------------------------------------------
        # EJECUTAR UNA PRACTICA
        # ----------------------------------------------------

        elif opcion.isdigit():

            numero_practica = int(opcion)

            if numero_practica in PRACTICAS_DISPONIBLES:
                ejecutar_practica(numero_practica)
            else:
                print("\n[!] Practica no disponible.")

        else:
            print("\n[!] Opcion no valida.")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()