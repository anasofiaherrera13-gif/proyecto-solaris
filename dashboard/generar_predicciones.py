from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# Rutas del proyecto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "solaris_datos_limpios.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "solaris_rf_v1_20260916.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "dashboard"
    / "data"
    / "solaris_predicciones.csv"
)


# ============================================================
# Cargar datos y modelo
# ============================================================

df = pd.read_csv(DATA_PATH)

modelo = joblib.load(MODEL_PATH)


# ============================================================
# Mantener solamente la población del modelo
# ============================================================

df_modelo = df[
    df["estado_cuenta"].isin(["Al día", "Moroso"])
].copy()


# ============================================================
# Variables utilizadas por SOLARIS
# ============================================================

features = [
    "edad",
    "ingreso_mensual",
    "saldo_actual",
    "cuotas_pagadas",
    "cuotas_totales",
    "score_externo",
    "nivel_educativo",
    "ciudad",
]

X = df_modelo[features].copy()


# ============================================================
# Generar predicciones
# ============================================================

probabilidad_mora = modelo.predict_proba(X)[:, 1]

prediccion = modelo.predict(X)


# ============================================================
# Agregar resultados al dataset
# ============================================================

df_modelo["probabilidad_mora"] = probabilidad_mora

df_modelo["prediccion_mora"] = prediccion

df_modelo["prediccion_estado"] = df_modelo[
    "prediccion_mora"
].map({
    0: "Al día",
    1: "Moroso"
})


# ============================================================
# Crear nivel de riesgo para el dashboard
# ============================================================

def clasificar_riesgo(probabilidad):
    if probabilidad >= 0.70:
        return "Alto"
    elif probabilidad >= 0.40:
        return "Medio"
    else:
        return "Bajo"


df_modelo["nivel_riesgo"] = df_modelo[
    "probabilidad_mora"
].apply(clasificar_riesgo)


# ============================================================
# Acción sugerida
# ============================================================

def accion_sugerida(fila):

    if fila["probabilidad_mora"] >= 0.70:
        return "Revisión prioritaria"

    elif fila["probabilidad_mora"] >= 0.40:
        return "Monitorear"

    else:
        return "Seguimiento habitual"


df_modelo["accion_sugerida"] = df_modelo.apply(
    accion_sugerida,
    axis=1
)


# ============================================================
# Guardar archivo
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df_modelo.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Verificación
# ============================================================

print("Archivo creado correctamente:")
print(OUTPUT_PATH)

print()
print("Número de registros:", len(df_modelo))

print()
print("Distribución del nivel de riesgo:")
print(
    df_modelo["nivel_riesgo"].value_counts()
)

print()
print("Columnas creadas:")
print(
    [
        "probabilidad_mora",
        "prediccion_mora",
        "prediccion_estado",
        "nivel_riesgo",
        "accion_sugerida",
    ]
)
