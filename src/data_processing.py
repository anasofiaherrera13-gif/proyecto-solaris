"""Preparación determinística de datos para el Proyecto SOLARIS.

Este módulo transforma la exportación cruda de FrescaMar en una versión
procesada reproducible. No realiza imputación estadística, escalamiento,
codificación One-Hot ni ninguna transformación que deba aprenderse con
datos de entrenamiento.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNAS_REQUERIDAS = [
    "id_cliente",
    "edad",
    "ingreso_mensual",
    "nivel_educativo",
    "ciudad",
    "fecha_vinculacion",
    "saldo_actual",
    "cuotas_pagadas",
    "cuotas_totales",
    "estado_cuenta",
    "score_externo",
]

CIUDADES_CANONICAS = [
    "Bogotá",
    "Medellín",
    "Cali",
    "Barranquilla",
    "Cartagena",
    "Bucaramanga",
    "Pereira",
    "Manizales",
    "Santa Marta",
    "Ibagué",
    "Villavicencio",
    "Cúcuta",
]

ALIAS_CIUDADES = {
    "ctg": "Cartagena",
    "cartagena de indias": "Cartagena",
    "bogota dc": "Bogotá",
    "bogota d c": "Bogotá",
    "santa marta magdalena": "Santa Marta",
}


def _clave_texto(valor: object) -> str:
    """Convierte un valor textual en una clave comparable sin acentos.

    Args:
        valor (object): Valor de texto a normalizar.

    Returns:
        str: Texto en minúsculas, sin acentos y con espacios uniformes.

    Example:
        >>> _clave_texto("  Bogotá D.C. ")
        'bogota d c'
    """
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()
    texto = "".join(
        caracter
        for caracter in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(caracter)
    )
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_ciudad(valor: object) -> object:
    """Estandariza formatos de ciudad sin usar información del target.

    La regla corrige diferencias de mayúsculas, minúsculas, acentos,
    sufijos departamentales separados por coma y abreviaturas documentadas.

    Args:
        valor (object): Ciudad original.

    Returns:
        object: Ciudad normalizada o ``np.nan`` si el valor era faltante.

    Example:
        >>> normalizar_ciudad("SANTA MARTA")
        'Santa Marta'
        >>> normalizar_ciudad("CTG")
        'Cartagena'
    """
    if pd.isna(valor):
        return np.nan

    texto_original = re.sub(r"\s+", " ", str(valor).strip())

    # Cuando viene "Ciudad, Departamento", se conserva la ciudad.
    if "," in texto_original:
        texto_original = texto_original.split(",", maxsplit=1)[0].strip()

    clave = _clave_texto(texto_original)

    if clave in ALIAS_CIUDADES:
        return ALIAS_CIUDADES[clave]

    for ciudad in CIUDADES_CANONICAS:
        if clave == _clave_texto(ciudad):
            return ciudad

    # Regla conservadora para variantes no contempladas explícitamente.
    return texto_original.title()


def normalizar_nivel_educativo(valor: object) -> object:
    """Estandariza las categorías conocidas de nivel educativo.

    Args:
        valor (object): Categoría original.

    Returns:
        object: Categoría canónica o ``np.nan`` si era faltante.

    Example:
        >>> normalizar_nivel_educativo("tecnólogo")
        'Tecnólogo'
    """
    if pd.isna(valor):
        return np.nan

    canonicas = {
        "primaria": "Primaria",
        "secundaria": "Secundaria",
        "tecnico": "Técnico",
        "tecnologo": "Tecnólogo",
        "universitario": "Universitario",
        "posgrado": "Posgrado",
    }

    texto = str(valor).strip()
    return canonicas.get(_clave_texto(texto), texto)


def normalizar_estado_cuenta(valor: object) -> object:
    """Estandariza las categorías del estado de cuenta.

    Args:
        valor (object): Estado de cuenta original.

    Returns:
        object: Estado canónico o ``np.nan`` si era faltante.

    Example:
        >>> normalizar_estado_cuenta("AL DÍA")
        'Al día'
    """
    if pd.isna(valor):
        return np.nan

    canonicas = {
        "al dia": "Al día",
        "moroso": "Moroso",
        "cancelado": "Cancelado",
    }

    texto = str(valor).strip()
    return canonicas.get(_clave_texto(texto), texto)


def validar_esquema(df: pd.DataFrame) -> None:
    """Valida que el dataset contenga las columnas esperadas.

    Args:
        df (pd.DataFrame): Dataset a validar.

    Returns:
        None.

    Raises:
        ValueError: Si faltan columnas requeridas.

    Example:
        >>> validar_esquema(df)
    """
    faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "El dataset no contiene todas las columnas requeridas. "
            f"Faltan: {faltantes}"
        )


def limpiar_datos_base(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza determinística previa al modelado.

    La función:
    - valida el esquema;
    - normaliza categorías de texto;
    - convierte variables numéricas a tipo numérico;
    - convierte ``fecha_vinculacion`` a fecha;
    - marca edades fuera del rango 18-100 como faltantes;
    - elimina duplicados exactos.

    No imputa ``score_externo``, no corrige cuotas inconsistentes y no
    aplica escalamiento/codificación. Esas decisiones permanecen dentro
    del pipeline de Machine Learning para evitar data leakage.

    Args:
        df (pd.DataFrame): Dataset crudo.

    Returns:
        pd.DataFrame: Dataset con limpieza base reproducible.

    Example:
        >>> limpio = limpiar_datos_base(df_crudo)
        >>> limpio.shape[1]
        11
    """
    validar_esquema(df)

    limpio = df.copy()

    # Identificador.
    limpio["id_cliente"] = (
        limpio["id_cliente"]
        .astype("string")
        .str.strip()
    )

    # Conversión numérica sin imputación.
    columnas_numericas = [
        "edad",
        "ingreso_mensual",
        "saldo_actual",
        "cuotas_pagadas",
        "cuotas_totales",
        "score_externo",
    ]

    for columna in columnas_numericas:
        limpio[columna] = pd.to_numeric(
            limpio[columna],
            errors="coerce",
        )

    # Regla de plausibilidad documentada para edad.
    edad_invalida = (
        (limpio["edad"] < 18)
        | (limpio["edad"] > 100)
    )
    limpio.loc[edad_invalida, "edad"] = np.nan

    # Categorías.
    limpio["ciudad"] = limpio["ciudad"].map(normalizar_ciudad)
    limpio["nivel_educativo"] = limpio["nivel_educativo"].map(
        normalizar_nivel_educativo
    )
    limpio["estado_cuenta"] = limpio["estado_cuenta"].map(
        normalizar_estado_cuenta
    )

    # Fecha: se convierte de manera determinística.
    limpio["fecha_vinculacion"] = pd.to_datetime(
        limpio["fecha_vinculacion"],
        errors="coerce",
    )

    # El dataset auditado no presenta duplicados, pero la regla queda
    # versionada para futuras exportaciones.
    limpio = limpio.drop_duplicates().reset_index(drop=True)

    return limpio


def guardar_datos_procesados(
    df: pd.DataFrame,
    ruta_salida: Path,
) -> Path:
    """Guarda la versión procesada del dataset en CSV.

    Args:
        df (pd.DataFrame): Dataset procesado.
        ruta_salida (Path): Ruta destino del CSV.

    Returns:
        Path: Ruta del archivo generado.

    Example:
        >>> guardar_datos_procesados(
        ...     df_limpio,
        ...     Path("data/processed/solaris_datos_limpios.csv"),
        ... )
        PosixPath('data/processed/solaris_datos_limpios.csv')
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df_salida = df.copy()

    if pd.api.types.is_datetime64_any_dtype(
        df_salida["fecha_vinculacion"]
    ):
        df_salida["fecha_vinculacion"] = (
            df_salida["fecha_vinculacion"]
            .dt.strftime("%Y-%m-%d")
        )

    df_salida.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8-sig",
    )

    return ruta_salida
