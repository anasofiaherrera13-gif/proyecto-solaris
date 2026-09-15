# ============================================================
# PROYECTO SOLARIS - QUANTUM ANALYTICS
# src/pipeline.py
# Pipeline ML end-to-end reproducible
# ============================================================

import os
import random
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

RANDOM_SEED = 42
MODEL_VERSION = "v1"
MODEL_DATE = "20260915"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RUTA_DATOS = DATA_DIR / "solaris_datos_limpios.csv"
RUTA_CONFUSION = FIGURES_DIR / "confusion_matrix_s08.png"
RUTA_PIPELINE = MODELS_DIR / f"solaris_rf_{MODEL_VERSION}_{MODEL_DATE}.joblib"
RUTA_METRICAS = REPORTS_DIR / "metricas_s08.json"


def configurar_reproducibilidad():
    """Configura las semillas globales para reproducibilidad.

    Args:
        None.

    Returns:
        None: Ajusta PYTHONHASHSEED, random y NumPy.

    Example:
        >>> configurar_reproducibilidad()
    """
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def cargar_datos(ruta_datos):
    """Carga y valida el dataset procesado de SOLARIS.

    Args:
        ruta_datos (Path): Ruta al CSV limpio utilizado por el pipeline.

    Returns:
        pd.DataFrame: Dataset cargado con el esquema minimo validado.

    Example:
        >>> df = cargar_datos(Path("data/processed/solaris_datos_limpios.csv"))
        >>> df.shape
        (15000, 11)
    """
    columnas_requeridas = {
        "edad", "ingreso_mensual", "saldo_actual", "cuotas_pagadas",
        "cuotas_totales", "score_externo", "nivel_educativo", "ciudad",
        "estado_cuenta",
    }

    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos: {ruta_datos}")

    df = pd.read_csv(ruta_datos)
    columnas_faltantes = sorted(columnas_requeridas - set(df.columns))

    if columnas_faltantes:
        raise ValueError(
            "El dataset no contiene todas las columnas requeridas. "
            f"Faltan: {columnas_faltantes}"
        )

    print("=" * 72)
    print("PROYECTO SOLARIS - PIPELINE ML END-TO-END")
    print("=" * 72)
    print("\nCARGA DE DATOS LIMPIOS")
    print("=" * 72)
    print("Ruta:", ruta_datos)
    print("Shape original:", df.shape)
    print("Datos cargados correctamente.")

    return df


def definir_features_target(df):
    """Define las variables predictoras y construye el target binario.

    Args:
        df (pd.DataFrame): Dataset limpio con la columna estado_cuenta.

    Returns:
        tuple: X, y, lista de variables numericas y lista de categoricas.

    Example:
        >>> X, y, num_cols, cat_cols = definir_features_target(df)
        >>> X.shape[1]
        8
    """
    df_modelo = df[df["estado_cuenta"].isin(["Al día", "Moroso"])].copy()
    target = "mora"
    df_modelo[target] = df_modelo["estado_cuenta"].map({"Al día": 0, "Moroso": 1})

    numeric_features = [
        "edad", "ingreso_mensual", "saldo_actual", "cuotas_pagadas",
        "cuotas_totales", "score_externo",
    ]
    categorical_features = ["nivel_educativo", "ciudad"]

    # estado_cuenta se excluye porque define directamente el target.
    # id_cliente se excluye por ser identificador unico.
    # fecha_vinculacion no se usa porque no es fecha de snapshot/evento.
    X = df_modelo[numeric_features + categorical_features].copy()
    y = df_modelo[target].copy()

    print("\nDEFINICION DE FEATURES Y TARGET")
    print("=" * 72)
    print("Shape de X:", X.shape)
    print("Shape de y:", y.shape)
    print("\nFeatures numericas:", numeric_features)
    print("Features categoricas:", categorical_features)
    print("\nDistribucion del target:")
    print(y.value_counts())
    print("\nProporcion del target:")
    print(y.value_counts(normalize=True))

    proporcion_morosos = float(y.mean())
    print(
        "\nConclusion: la clase Moroso representa "
        f"{proporcion_morosos:.2%} de las observaciones. Se mantiene "
        "la proporcion con stratify=y y se priorizan ROC-AUC, precision, "
        "recall y F1 sobre el accuracy aislado."
    )

    return X, y, numeric_features, categorical_features


def auditar_leakage():
    """Audita individualmente las features antes del split.

    Args:
        None.

    Returns:
        dict: Resultado de auditoria para cada feature.

    Raises:
        RuntimeError: Si se detecta una feature con leakage confirmado.

    Example:
        >>> auditoria = auditar_leakage()
        >>> auditoria["edad"]["decision"]
        'Sin leakage detectado'
    """
    auditoria = {
        "edad": {
            "existe_al_predecir": True,
            "depende_target_o_postevento": False,
            "usa_informacion_futura": False,
            "decision": "Sin leakage detectado",
        },
        "ingreso_mensual": {
            "existe_al_predecir": True,
            "depende_target_o_postevento": False,
            "usa_informacion_futura": False,
            "decision": "Sin leakage detectado, siempre que represente el ingreso disponible al scoring",
        },
        "saldo_actual": {
            "existe_al_predecir": True,
            "depende_target_o_postevento": False,
            "usa_informacion_futura": False,
            "decision": "Sin leakage detectado bajo el supuesto de saldo conocido al scoring",
        },
        "cuotas_pagadas": {
            "existe_al_predecir": True,
            "depende_target_o_postevento": False,
            "usa_informacion_futura": False,
            "decision": "Sin leakage detectado si solo incluye pagos previos al scoring",
        },
        "cuotas_totales": {
            "existe_al_predecir": True,
            "depende_target_o_postevento": False,
            "usa_informacion_futura": False,
            "decision": "Sin leakage detectado; condicion contractual conocida",
        },
        "score_externo": {
            "existe_al_predecir": True,
            "depende_target_o_postevento": False,
            "usa_informacion_futura": False,
            "decision": "Sin leakage detectado si el score existe antes de la decision",
        },
        "nivel_educativo": {
            "existe_al_predecir": True,
            "depende_target_o_postevento": False,
            "usa_informacion_futura": False,
            "decision": "Sin leakage detectado",
        },
        "ciudad": {
            "existe_al_predecir": True,
            "depende_target_o_postevento": False,
            "usa_informacion_futura": False,
            "decision": "Sin leakage detectado",
        },
    }

    features_con_leakage = []

    print("\nAUDITORIA PRE-SPLIT DE DATA LEAKAGE")
    print("=" * 72)

    for variable, info in auditoria.items():
        riesgo_confirmado = (
            not info["existe_al_predecir"]
            or info["depende_target_o_postevento"]
            or info["usa_informacion_futura"]
        )
        if riesgo_confirmado:
            features_con_leakage.append(variable)
        print(f"{variable}: {info['decision']}")

    print("\nColumnas excluidas:")
    print("- estado_cuenta: contiene directamente la informacion del target.")
    print("- id_cliente: identificador unico sin valor predictivo generalizable.")
    print("- fecha_vinculacion: no se usa directamente en esta version.")

    if features_con_leakage:
        raise RuntimeError(
            "Se detecto leakage confirmado en: " + ", ".join(features_con_leakage)
        )

    print(
        "\nConclusion: no se detecto data leakage en las features "
        "seleccionadas bajo los supuestos temporales documentados."
    )
    return auditoria


def dividir_datos(X, y):
    """Divide los datos en train y test conservando la clase objetivo.

    Args:
        X (pd.DataFrame): Variables predictoras.
        y (pd.Series): Target binario.

    Returns:
        tuple: X_train, X_test, y_train y y_test.

    Example:
        >>> X_train, X_test, y_train, y_test = dividir_datos(X, y)
        >>> X_train.shape[0]
        9840
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
        shuffle=True,
    )

    print("\nSPLIT TRAIN / TEST")
    print("=" * 72)
    print("Shape X_train:", X_train.shape)
    print("Shape X_test :", X_test.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape y_test :", y_test.shape)
    print("\nProporcion del target en TRAIN:")
    print(y_train.value_counts(normalize=True))
    print("\nProporcion del target en TEST:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def construir_preprocesador(numeric_features, categorical_features):
    """Construye el ColumnTransformer de variables numericas y categoricas.

    Args:
        numeric_features (list[str]): Variables numericas.
        categorical_features (list[str]): Variables categoricas.

    Returns:
        ColumnTransformer: Preprocesador listo para el Pipeline.

    Example:
        >>> preprocessor = construir_preprocesador(num_cols, cat_cols)
        >>> isinstance(preprocessor, ColumnTransformer)
        True
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def construir_pipeline(preprocessor):
    """Construye el Pipeline de preprocesamiento y Random Forest.

    Args:
        preprocessor (ColumnTransformer): Transformador de entrada.

    Returns:
        Pipeline: Pipeline de sklearn listo para optimizacion.

    Example:
        >>> pipeline = construir_pipeline(preprocessor)
        >>> list(pipeline.named_steps.keys())
        ['preprocessor', 'classifier']
    """
    classifier = RandomForestClassifier(random_state=RANDOM_SEED)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("\nCOLUMNTRANSFORMER Y PIPELINE")
    print("=" * 72)
    print(pipeline)
    return pipeline


def optimizar_pipeline(pipeline, X_train, y_train):
    """Optimiza el Random Forest con GridSearchCV y ROC-AUC.

    Args:
        pipeline (Pipeline): Pipeline base.
        X_train (pd.DataFrame): Variables de entrenamiento.
        y_train (pd.Series): Target de entrenamiento.

    Returns:
        tuple: Objeto GridSearchCV ajustado y mejor Pipeline encontrado.

    Example:
        >>> search, best_pipeline = optimizar_pipeline(pipeline, X_train, y_train)
        >>> hasattr(search, "best_params_")
        True
    """
    param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("\nOPTIMIZACION DE HIPERPARAMETROS")
    print("=" * 72)
    print("24 combinaciones x 5 folds = 120 ajustes.")

    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    print("\nMejores hiperparametros:")
    print(search.best_params_)
    print("\nMejor ROC-AUC promedio de validacion cruzada:")
    print(f"{search.best_score_:.4f}")

    return search, best_pipeline


def evaluar_modelo(best_pipeline, X_test, y_test):
    """Evalua el mejor Pipeline sobre el conjunto test.

    Args:
        best_pipeline (Pipeline): Pipeline optimizado.
        X_test (pd.DataFrame): Variables de prueba.
        y_test (pd.Series): Target real de prueba.

    Returns:
        dict: Predicciones, probabilidades y metricas de test.

    Example:
        >>> resultados = evaluar_modelo(best_pipeline, X_test, y_test)
        >>> "auc_test" in resultados
        True
    """
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    resultados = {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "auc_test": roc_auc_score(y_test, y_proba),
        "precision_test": precision_score(y_test, y_pred, pos_label=1),
        "recall_test": recall_score(y_test, y_pred, pos_label=1),
        "f1_test": f1_score(y_test, y_pred, pos_label=1),
        "accuracy_test": accuracy_score(y_test, y_pred),
    }

    print("\nEVALUACION FINAL EN TEST")
    print("=" * 72)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Al día", "Moroso"],
            digits=4,
        )
    )
    print(f"ROC-AUC Test     : {resultados['auc_test']:.4f}")
    print(f"Precision Moroso : {resultados['precision_test']:.4f}")
    print(f"Recall Moroso    : {resultados['recall_test']:.4f}")
    print(f"F1 Moroso        : {resultados['f1_test']:.4f}")
    print(f"Accuracy         : {resultados['accuracy_test']:.4f}")

    return resultados


def verificar_leakage_post_entrenamiento(best_pipeline, X_train, y_train, X_test, y_test):
    """Compara ROC-AUC Train-Test y valida el umbral de 0.10.

    Args:
        best_pipeline (Pipeline): Pipeline optimizado.
        X_train (pd.DataFrame): Variables de entrenamiento.
        y_train (pd.Series): Target de entrenamiento.
        X_test (pd.DataFrame): Variables de prueba.
        y_test (pd.Series): Target de prueba.

    Returns:
        tuple: auc_train, auc_test, diferencia_auc y estado_leakage.

    Raises:
        RuntimeError: Si la diferencia Train-Test supera 0.10.

    Example:
        >>> auc_train, auc_test, diff, estado = verificar_leakage_post_entrenamiento(
        ...     best_pipeline, X_train, y_train, X_test, y_test
        ... )
        >>> diff <= 0.10
        True
    """
    y_proba_train = best_pipeline.predict_proba(X_train)[:, 1]
    y_proba_test = best_pipeline.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, y_proba_train)
    auc_test = roc_auc_score(y_test, y_proba_test)
    diferencia_auc = abs(auc_train - auc_test)

    print("\nVERIFICACION POST-ENTRENAMIENTO DE DATA LEAKAGE")
    print("=" * 72)
    print(f"ROC-AUC Train      : {auc_train:.4f}")
    print(f"ROC-AUC Test       : {auc_test:.4f}")
    print(f"Diferencia absoluta: {diferencia_auc:.4f}")

    if diferencia_auc > 0.10:
        raise RuntimeError(
            "Pipeline detenido: la diferencia AUC Train-Test supera 0.10."
        )

    estado_leakage = "Diferencia aceptable - pipeline validado"
    print("\n" + estado_leakage)
    return auc_train, auc_test, diferencia_auc, estado_leakage


def guardar_artefactos(
    best_pipeline,
    search,
    y_test,
    resultados_test,
    auc_train,
    auc_test,
    diferencia_auc,
    estado_leakage,
):
    """Guarda modelo, metricas y matriz de confusion.

    Args:
        best_pipeline (Pipeline): Pipeline optimizado completo.
        search (GridSearchCV): Busqueda ajustada.
        y_test (pd.Series): Target real del conjunto test.
        resultados_test (dict): Predicciones y metricas de test.
        auc_train (float): ROC-AUC de entrenamiento.
        auc_test (float): ROC-AUC de test.
        diferencia_auc (float): Brecha absoluta Train-Test.
        estado_leakage (str): Resultado de la validacion post-entrenamiento.

    Returns:
        dict: Metricas guardadas en metricas_s08.json.

    Example:
        >>> metricas = guardar_artefactos(
        ...     best_pipeline, search, y_test, resultados_test,
        ...     auc_train, auc_test, diferencia_auc, estado_leakage
        ... )
        >>> "auc_test" in metricas
        True
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        resultados_test["y_pred"],
        display_labels=["Al día", "Moroso"],
        values_format="d",
    )
    plt.title("Matriz de Confusión - Random Forest Optimizado")
    plt.tight_layout()
    plt.savefig(RUTA_CONFUSION, dpi=300, bbox_inches="tight")
    plt.close()

    joblib.dump(best_pipeline, RUTA_PIPELINE)

    metricas = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "auc_cv": float(search.best_score_),
        "auc_train": float(auc_train),
        "auc_test": float(auc_test),
        "diferencia_auc": float(diferencia_auc),
        "precision_moroso": float(resultados_test["precision_test"]),
        "recall_moroso": float(resultados_test["recall_test"]),
        "f1_moroso": float(resultados_test["f1_test"]),
        "accuracy": float(resultados_test["accuracy_test"]),
        "estado_leakage": estado_leakage,
        "metodo_optimizacion": "GridSearchCV",
        "mejores_hiperparametros": search.best_params_,
    }

    with open(RUTA_METRICAS, "w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, ensure_ascii=False, indent=4)

    pipeline_verificacion = joblib.load(RUTA_PIPELINE)
    if not isinstance(pipeline_verificacion, Pipeline):
        raise TypeError("El archivo serializado no contiene un sklearn Pipeline valido.")

    print("\nSERIALIZACION Y METRICAS DE ENTREGA")
    print("=" * 72)
    print("Pipeline guardado en:", RUTA_PIPELINE)
    print("Metricas guardadas en:", RUTA_METRICAS)
    print("Matriz de confusion:", RUTA_CONFUSION)
    print("Verificacion del modelo: cargado correctamente.")

    return metricas


def imprimir_resumen(search, auc_train, auc_test, diferencia_auc, resultados_test, estado_leakage):
    """Imprime el resumen final de metricas del pipeline.

    Args:
        search (GridSearchCV): Busqueda de hiperparametros ajustada.
        auc_train (float): ROC-AUC de entrenamiento.
        auc_test (float): ROC-AUC de test.
        diferencia_auc (float): Diferencia absoluta Train-Test.
        resultados_test (dict): Metricas finales del conjunto test.
        estado_leakage (str): Estado de la validacion de leakage.

    Returns:
        None.

    Example:
        >>> imprimir_resumen(search, auc_train, auc_test, diff, resultados, estado)
    """
    print("\n" + "=" * 72)
    print("RESUMEN FINAL DEL PIPELINE SOLARIS")
    print("=" * 72)
    print(f"Mejor ROC-AUC CV      : {search.best_score_:.4f}")
    print(f"ROC-AUC Train         : {auc_train:.4f}")
    print(f"ROC-AUC Test          : {auc_test:.4f}")
    print(f"Diferencia Train-Test : {diferencia_auc:.4f}")
    print(f"Precision Moroso      : {resultados_test['precision_test']:.4f}")
    print(f"Recall Moroso         : {resultados_test['recall_test']:.4f}")
    print(f"F1 Moroso             : {resultados_test['f1_test']:.4f}")
    print(f"Accuracy              : {resultados_test['accuracy_test']:.4f}")
    print(f"Estado                : {estado_leakage}")
    print("\nEjecucion finalizada correctamente.")


def main():
    """Ejecuta de principio a fin el pipeline reproducible de SOLARIS.

    Args:
        None.

    Returns:
        None: Coordina carga, preparacion, entrenamiento, optimizacion,
        evaluacion, validacion y serializacion del modelo.

    Example:
        >>> main()
        PROYECTO SOLARIS - PIPELINE ML END-TO-END
        ...
    """
    configurar_reproducibilidad()
    df = cargar_datos(RUTA_DATOS)

    X, y, numeric_features, categorical_features = definir_features_target(df)
    auditar_leakage()

    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    preprocessor = construir_preprocesador(
        numeric_features,
        categorical_features,
    )
    pipeline = construir_pipeline(preprocessor)

    search, best_pipeline = optimizar_pipeline(
        pipeline,
        X_train,
        y_train,
    )

    resultados_test = evaluar_modelo(
        best_pipeline,
        X_test,
        y_test,
    )

    auc_train, auc_test, diferencia_auc, estado_leakage = (
        verificar_leakage_post_entrenamiento(
            best_pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    guardar_artefactos(
        best_pipeline,
        search,
        y_test,
        resultados_test,
        auc_train,
        auc_test,
        diferencia_auc,
        estado_leakage,
    )

    imprimir_resumen(
        search,
        auc_train,
        auc_test,
        diferencia_auc,
        resultados_test,
        estado_leakage,
    )


if __name__ == "__main__":
    main()
