"""Feature engineering y preprocesamiento del Proyecto SOLARIS.

Las transformaciones que aprenden parámetros se encapsulan en objetos de
scikit-learn y deben ajustarse únicamente con el conjunto de entrenamiento.
"""

from __future__ import annotations

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


COLS_LOG_SCALE = [
    "saldo_actual",
    "ratio_deuda_ingreso",
]

COLS_STD_SCALE = [
    "edad",
    "ingreso_mensual",
    "cuotas_pagadas",
    "cuotas_totales",
    "score_externo",
    "avance_pago",
    "cuotas_pendientes",
    "saldo_por_cuota_pendiente",
    "ingreso_cero",
    "inconsistencia_cuotas",
]

COLS_ONE_HOT = [
    "ciudad",
]

COLS_ORDINAL = [
    "nivel_educativo",
]

ORDEN_EDUCATIVO = [
    "Primaria",
    "Secundaria",
    "Técnico",
    "Tecnólogo",
    "Universitario",
    "Posgrado",
]


def crear_features(df):
    """Crea las variables derivadas utilizadas por SOLARIS.

    La función no modifica el DataFrame recibido. Las inconsistencias de
    cuotas se conservan mediante un indicador explícito y las variables
    derivadas afectadas se dejan como ``NaN`` para que la imputación ocurra
    posteriormente dentro del pipeline.

    Args:
        df (pandas.DataFrame): Variables predictoras sin transformar.

    Returns:
        pandas.DataFrame: Copia con seis variables derivadas adicionales.

    Example:
        >>> X_fe = crear_features(X_train)
        >>> "ratio_deuda_ingreso" in X_fe.columns
        True
    """
    X = df.copy()

    X["ratio_deuda_ingreso"] = (
        X["saldo_actual"]
        / (X["ingreso_mensual"] + 1)
    )

    X["ingreso_cero"] = (
        X["ingreso_mensual"] == 0
    ).astype(int)

    X["inconsistencia_cuotas"] = (
        X["cuotas_pagadas"]
        > X["cuotas_totales"]
    ).astype(int)

    X["avance_pago"] = np.where(
        X["inconsistencia_cuotas"] == 0,
        X["cuotas_pagadas"]
        / (X["cuotas_totales"] + 1),
        np.nan,
    )

    X["cuotas_pendientes"] = np.where(
        X["inconsistencia_cuotas"] == 0,
        X["cuotas_totales"]
        - X["cuotas_pagadas"],
        np.nan,
    )

    X["saldo_por_cuota_pendiente"] = (
        X["saldo_actual"]
        / (X["cuotas_pendientes"] + 1)
    )

    return X


def build_preprocessor():
    """Construye el ``ColumnTransformer`` de SOLARIS sin ajustarlo.

    La imputación, transformación logarítmica, escalamiento y codificación
    se aprenden posteriormente y exclusivamente sobre el conjunto train.

    Args:
        None.

    Returns:
        sklearn.compose.ColumnTransformer: Preprocesador sin ajustar.

    Example:
        >>> preprocessor = build_preprocessor()
        >>> preprocessor.__class__.__name__
        'ColumnTransformer'
    """
    pipeline_log = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "log",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    pipeline_std = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    pipeline_ohe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    pipeline_ordinal = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ORDEN_EDUCATIVO],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "log_num",
                pipeline_log,
                COLS_LOG_SCALE,
            ),
            (
                "std_num",
                pipeline_std,
                COLS_STD_SCALE,
            ),
            (
                "nominal",
                pipeline_ohe,
                COLS_ONE_HOT,
            ),
            (
                "ordinal",
                pipeline_ordinal,
                COLS_ORDINAL,
            ),
        ],
        remainder="drop",
    )
