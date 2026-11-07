"""
Semana 13 - Pronóstico de Series de Tiempo (Forecasting)

Objetivo:
Construir y evaluar un modelo de regresión de series de tiempo (Forecasting)
para predecir la cantidad futura de eventos maliciosos por hora en la Honeynet,
utilizando características de rezago (Lag Features) y tendencias temporales.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. CONFIGURACIÓN Y RUTAS DE SALIDA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "practica1" / "HoneyNetEvents_Clean.csv"

# Carpeta de salida para practica 8 / semana 13
OUTPUT_DIR = BASE_DIR / "resultados_practica8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGAR Y ESTRUCTURAR LA SERIE DE TIEMPO
# ============================================================

print("=" * 70)
print("SEMANA 13 - PRONÓSTICO DE SERIES DE TIEMPO (FORECASTING)")
print("=" * 70)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"\nNo se encontró el archivo CSV:\n{CSV_PATH}\n")

df = pd.read_csv(CSV_PATH)

print(f"\nDataset cargado correctamente.")
print(f"Registros totales: {len(df):,}")

# Convertir a datetime y agrupar por hora
df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='ISO8601')
ts_hourly = df.set_index('timestamp_dt').resample('h').size().reset_index(name='event_count')

print(f"Total de periodos de tiempo (Horas registradas): {len(ts_hourly)}")


# ============================================================
# 3. INGENIERÍA DE CARACTERÍSTICAS PARA FORECASTING
# ============================================================

df_ts = ts_hourly.copy()

# Variables de Rezagos (Lags)
df_ts['lag_1'] = df_ts['event_count'].shift(1)
df_ts['lag_2'] = df_ts['event_count'].shift(2)

# Promedio móvil de 3 horas
df_ts['rolling_3'] = df_ts['event_count'].shift(1).rolling(3).mean()

# Hora del día (Componente Cíclica)
df_ts['hour'] = df_ts['timestamp_dt'].dt.hour
df_ts['time_index'] = np.arange(len(df_ts))

# Eliminar nulos generados por los rezagos
df_clean = df_ts.dropna().copy()

features = ['time_index', 'hour', 'lag_1', 'lag_2', 'rolling_3']
X = df_clean[features]
y = df_clean['event_count']


# ============================================================
# 4. DIVISIÓN TEMPORAL (ENTRENAMIENTO Y TEST)
# ============================================================

# En series de tiempo NO se usa shuffle; la división es estrictamente secuencial
split_point = int(len(df_clean) * 0.75)

X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
dates_test = df_clean['timestamp_dt'].iloc[split_point:]

print(f"\nPeriodos de Entrenamiento: {len(X_train)} horas")
print(f"Periodos de Prueba (Test): {len(X_test)} horas")


# ============================================================
# 5. ENTRENAMIENTO DEL MODELO DE REGRESIÓN DE TIEMPO
# ============================================================

modelo_forecast = Ridge(alpha=1.0)
modelo_forecast.fit(X_train, y_train)

# Predicción en el conjunto de test
predicciones = modelo_forecast.predict(X_test)
predicciones = np.clip(predicciones, a_min=0, a_max=None) # No puede haber eventos negativos


# ============================================================
# 6. EVALUACIÓN DE MÉTRICAS DE PRONÓSTICO
# ============================================================

mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

print("\n" + "-" * 70)
print("MÉTRICAS DE RENDIMIENTO DEL PRONÓSTICO")
print("-" * 70)
print(f"Error Absoluto Medio (MAE): {mae:.2f} eventos por hora")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"Coeficiente de Determinación (R^2): {r2:.4f}")


# ============================================================
# 7. PROYECTAR PRONÓSTICO DE HORAS FUTURAS
# ============================================================

# Proyectar las siguientes 6 horas
ultimos_datos = df_clean.iloc[-1].copy()
proyecciones_futuras = []

ultimo_time_index = ultimos_datos['time_index']
ultimo_lag1 = ultimos_datos['event_count']
ultimo_lag2 = ultimos_datos['lag_1']
ultimas_obs = list(df_clean['event_count'].tail(3))

for step in range(1, 7):
    siguiente_hora = (int(ultimos_datos['hour']) + step) % 24
    time_idx = ultimo_time_index + step
    roll_3 = np.mean(ultimas_obs[-3:])
    
    feats = pd.DataFrame([{
        'time_index': time_idx,
        'hour': siguiente_hora,
        'lag_1': ultimo_lag1,
        'lag_2': ultimo_lag2,
        'rolling_3': roll_3
    }])
    
    pred_futura = max(0, modelo_forecast.predict(feats)[0])
    proyecciones_futuras.append({
        'hora_futura_paso': f"+{step}h",
        'hora_del_dia': siguiente_hora,
        'volumen_predicho_eventos': int(pred_futura)
    })
    
    # Actualizar rezagos para el siguiente paso
    ultimo_lag2 = ultimo_lag1
    ultimo_lag1 = pred_futura
    ultimas_obs.append(pred_futura)

df_futuro = pd.DataFrame(proyecciones_futuras)
print("\n========== PRONÓSTICO PRÓXIMAS 6 HORAS ==========")
print(df_futuro.to_string(index=False))


# ============================================================
# 8. GUARDAR DATOS EN CSV
# ============================================================

df_evaluacion = pd.DataFrame({
    'timestamp': dates_test.values,
    'volumen_real': y_test.values,
    'volumen_predicho': predicciones
})
df_evaluacion.to_csv(OUTPUT_DIR / "01_evaluacion_pronostico.csv", index=False, encoding="utf-8-sig")
df_futuro.to_csv(OUTPUT_DIR / "02_proyeccion_futura_6h.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 9. GENERACIÓN DE GRÁFICAS (GUARDADO AUTOMÁTICO)
# ============================================================

print("\nGenerando gráficas de pronóstico...")

# GRÁFICA 1: Serie de Tiempo Real vs Predicha
plt.figure(figsize=(12, 6))
plt.plot(df_clean['timestamp_dt'], df_clean['event_count'], label='Datos Históricos Reales', color='steelblue', alpha=0.6)
plt.plot(dates_test, y_test, label='Real (Periodo Prueba)', color='navy', linewidth=2)
plt.plot(dates_test, predicciones, label='Pronóstico del Modelo', color='crimson', linestyle='--', linewidth=2)
plt.title("Pronóstico de Volumen de Ataques por Hora (Forecasting)", fontsize=13)
plt.xlabel("Fecha y Hora", fontsize=11)
plt.ylabel("Cantidad de Eventos / Hora", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_serie_tiempo_real_vs_prediccion.png", dpi=150)
plt.close()


# GRÁFICA 2: Proyección de las Próximas 6 Horas
plt.figure(figsize=(8, 5))
sns.barplot(data=df_futuro, x="hora_futura_paso", y="volumen_predicho_eventos", palette="magma")
plt.title("Proyección Estimada de Ataques para las Próximas 6 Horas", fontsize=13)
plt.xlabel("Paso Futuro (Horas)", fontsize=11)
plt.ylabel("Eventos Estimados", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.7)

for idx, row in df_futuro.iterrows():
    plt.text(idx, row["volumen_predicho_eventos"] + 100, f"{row['volumen_predicho_eventos']:,}", ha='center', weight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_proyeccion_proximas_horas.png", dpi=150)
plt.close()


# GRÁFICA 3: Análisis de Residuales (Errores)
residuales = y_test.values - predicciones
plt.figure(figsize=(9, 5))
plt.scatter(predicciones, residuales, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title("Gráfica de Residuales del Modelo de Pronóstico", fontsize=13)
plt.xlabel("Valores Predichos", fontsize=11)
plt.ylabel("Residuales (Real - Predicho)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_residuales_pronostico.png", dpi=150)
plt.close()


# ============================================================
# 10. GENERAR REPORTE TXT PARA ENTREGA
# ============================================================

lineas = []

lineas.append("=" * 70)
lineas.append("REPORTE - SEMANA 13: PRONÓSTICO DE SERIES DE TIEMPO (FORECASTING)")
lineas.append("=" * 70)
lineas.append("")
lineas.append(f"Dataset: {CSV_PATH.name}")
lineas.append(f"Periodos analizados: {len(df_clean)} horas")
lineas.append(f"Variables predictoras: {', '.join(features)}")
lineas.append("")

lineas.append("-" * 70)
lineas.append("OBJETIVO DE NEGOCIO Y CIBERSEGURIDAD")
lineas.append("-" * 70)
lineas.append(
    "Predecir la volumetría de tráfico malicioso en las siguientes horas "
    "para anticipar sobrecargas en la infraestructura Honeynet y planificar la "
    "capacidad del almacenamiento de logs."
)
lineas.append("")

lineas.append("-" * 70)
lineas.append("MÉTRICAS DE RENDIMIENTO")
lineas.append("-" * 70)
lineas.append(f"MAE (Error Absoluto Medio): {mae:.2f} eventos por hora")
lineas.append(f"RMSE (Raíz de Error Cuadrático Medio): {rmse:.2f}")
lineas.append(f"R^2 Score: {r2:.4f}")
lineas.append("")

lineas.append("-" * 70)
lineas.append("PROYECCIÓN PRÓXIMAS 6 HORAS")
lineas.append("-" * 70)
lineas.append(df_futuro.to_string(index=False))
lineas.append("")

lineas.append("-" * 70)
lineas.append("INTERPRETACIÓN DE HALLAZGOS (EJE DE PRESENTACIÓN)")
lineas.append("-" * 70)
lineas.append(
    "El modelo de regresión de series de tiempo logró capturar la tendencia de "
    "las ráfagas de ataques por hora. La inclusión de rezagos (lag_1, lag_2) "
    "permite anticipar con alta fidelidad las subidas y bajadas del tráfico malicioso "
    "facilitando la gestión proactiva de infraestructura en el SOC."
)
lineas.append("")

with open(OUTPUT_DIR / "06_reporte_semana13.txt", "w", encoding="utf-8") as archivo:
    archivo.write("\n".join(lineas))


# ============================================================
# 11. MENSAJE FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)
print(f"\nTodos los resultados se guardaron en:\n{OUTPUT_DIR}\n")
print("Archivos generados:")
for archivo in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {archivo.name}")