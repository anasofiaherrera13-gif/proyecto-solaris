import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import unicodedata
import re


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="SOLARIS | Riesgo de Mora",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def limpiar_texto(texto):
    """
    Convierte un texto a una forma comparable.
    Ejemplo:
    Medellín -> medellin
    B/Quilla -> b quilla
    """

    if pd.isna(texto):
        return ""

    texto = str(texto).strip().lower()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    texto = re.sub(
        r"[^a-z0-9]+",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


def normalizar_ciudad(valor):
    """
    Agrupa distintas formas de escribir una misma ciudad.
    """

    clave = limpiar_texto(valor)

    mapa_ciudades = {

        # Medellín
        "med": "Medellín",
        "mdl": "Medellín",
        "medellin": "Medellín",

        # Bogotá
        "bta": "Bogotá",
        "bog": "Bogotá",
        "bogota": "Bogotá",

        # Cali
        "cal": "Cali",
        "cali": "Cali",
        "santiago de cali": "Cali",

        # Villavicencio
        "villavo": "Villavicencio",
        "villavicencio": "Villavicencio",

        # Barranquilla
        "baq": "Barranquilla",
        "b quilla": "Barranquilla",
        "barranqilla": "Barranquilla",
        "barranquilla": "Barranquilla",

        # Bucaramanga
        "bga": "Bucaramanga",
        "b manga": "Bucaramanga",
        "bucaramanga": "Bucaramanga",

        # Pereira
        "pei": "Pereira",
        "pereira": "Pereira",

        # Santa Marta
        "smr": "Santa Marta",
        "sta marta": "Santa Marta",
        "santa marta": "Santa Marta",

        # Cúcuta
        "san jose de cucuta": "Cúcuta",
        "cucuta": "Cúcuta",

        # Manizales
        "mzl": "Manizales",
        "manizales": "Manizales",

        # Ibagué
        "ibague": "Ibagué",

        # Cartagena
        "cartagena": "Cartagena",
    }

    if clave in mapa_ciudades:
        return mapa_ciudades[clave]

    if pd.isna(valor):
        return "Sin información"

    return str(valor).strip().title()


def generar_accion(fila):
    """
    Genera una recomendación sencilla según la probabilidad de mora.
    """

    probabilidad = fila["probabilidad_mora"]

    if probabilidad >= 0.85:
        return "Revisión prioritaria"

    elif probabilidad >= 0.70:
        return "Contactar y revisar pagos"

    elif probabilidad >= 0.40:
        return "Monitorear"

    else:
        return "Seguimiento habitual"


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():

    try:
        df = pd.read_csv(
            "data/solaris_predicciones.csv"
        )

    except FileNotFoundError:

        st.error(
            "No se encontró el archivo "
            "data/solaris_predicciones.csv. "
            "Verifica que exista dentro de dashboard/data."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Identificador del distribuidor
    # --------------------------------------------------------

    if "id_distribuidor" not in df.columns:

        df = df.reset_index(
            drop=True
        )

        df["id_distribuidor"] = (
            df.index + 1
        )

    # --------------------------------------------------------
    # Normalización de ciudades
    # --------------------------------------------------------

    if "ciudad" in df.columns:

        # Conservar valor original por trazabilidad
        df["ciudad_original"] = (
            df["ciudad"]
        )

        df["ciudad"] = (
            df["ciudad"]
            .apply(normalizar_ciudad)
        )

    # --------------------------------------------------------
    # Conversión de fecha
    # --------------------------------------------------------

    if "fecha_vinculacion" in df.columns:

        df["fecha_vinculacion"] = (
            pd.to_datetime(
                df["fecha_vinculacion"],
                errors="coerce"
            )
        )

    # --------------------------------------------------------
    # Conversión de variables numéricas
    # --------------------------------------------------------

    columnas_numericas = [
        "probabilidad_mora",
        "saldo_actual",
        "cuotas_pagadas",
        "cuotas_totales",
        "score_externo",
        "edad",
        "ingreso_mensual"
    ]

    for columna in columnas_numericas:

        if columna in df.columns:

            df[columna] = (
                pd.to_numeric(
                    df[columna],
                    errors="coerce"
                )
            )

    return df


df = cargar_datos()


if df.empty:
    st.stop()


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "SOLARIS | Monitoreo de Riesgo de Mora"
)

st.caption(
    "Dashboard de apoyo a la gestión de riesgo de crédito "
    "de distribuidores de FrescaMar S.A.S."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Filtros")

    # --------------------------------------------------------
    # Ciudad
    # --------------------------------------------------------

    ciudades = (
        ["Todos"]
        + sorted(
            df["ciudad"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    ciudad_seleccionada = (
        st.selectbox(
            "Ciudad",
            ciudades
        )
    )

    # --------------------------------------------------------
    # Rango de fechas
    # --------------------------------------------------------

    rango_fechas = None

    if (
        "fecha_vinculacion" in df.columns
        and
        df["fecha_vinculacion"].notna().any()
    ):

        fecha_min = (
            df["fecha_vinculacion"]
            .min()
        )

        fecha_max = (
            df["fecha_vinculacion"]
            .max()
        )

        rango_fechas = st.date_input(
            "Rango de fecha de vinculación",
            value=(
                fecha_min.date(),
                fecha_max.date()
            ),
            min_value=fecha_min.date(),
            max_value=fecha_max.date()
        )

    # --------------------------------------------------------
    # Umbral de riesgo
    # --------------------------------------------------------

    umbral = st.slider(
        "Umbral de riesgo",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help=(
            "Un distribuidor se considera prioritario "
            "cuando su probabilidad de mora es igual "
            "o superior a este valor."
        )
    )


# ============================================================
# APLICAR FILTROS
# ============================================================

df_filtrado = df.copy()


# ------------------------------------------------------------
# Filtro de ciudad
# ------------------------------------------------------------

if ciudad_seleccionada != "Todos":

    df_filtrado = (
        df_filtrado[
            df_filtrado["ciudad"]
            == ciudad_seleccionada
        ]
    )


# ------------------------------------------------------------
# Filtro de fechas
# ------------------------------------------------------------

if (
    rango_fechas is not None
    and
    len(rango_fechas) == 2
):

    fecha_inicio = (
        pd.to_datetime(
            rango_fechas[0]
        )
    )

    fecha_fin = (
        pd.to_datetime(
            rango_fechas[1]
        )
    )

    df_filtrado = (
        df_filtrado[
            (
                df_filtrado[
                    "fecha_vinculacion"
                ]
                >= fecha_inicio
            )
            &
            (
                df_filtrado[
                    "fecha_vinculacion"
                ]
                <= fecha_fin
            )
        ]
    )


# ============================================================
# CLASIFICACIÓN DINÁMICA SEGÚN UMBRAL
# ============================================================

df_filtrado = df_filtrado.copy()

df_filtrado["en_riesgo"] = (
    df_filtrado["probabilidad_mora"]
    >= umbral
)


# ============================================================
# ZONA SUPERIOR - KPI CARDS
# ============================================================

st.subheader(
    "Resumen ejecutivo"
)


total_distribuidores = (
    len(df_filtrado)
)


distribuidores_riesgo = int(
    df_filtrado[
        "en_riesgo"
    ].sum()
)


probabilidad_promedio = (
    df_filtrado[
        "probabilidad_mora"
    ].mean()
    if total_distribuidores > 0
    else 0
)


valor_en_riesgo = (
    df_filtrado.loc[
        df_filtrado["en_riesgo"],
        "saldo_actual"
    ].sum()
)


porcentaje_riesgo = (
    distribuidores_riesgo
    / total_distribuidores
    * 100
    if total_distribuidores > 0
    else 0
)


col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        label="Distribuidores analizados",
        value=f"{total_distribuidores:,}"
    )


with col2:

    st.metric(
        label="Distribuidores en riesgo",
        value=f"{distribuidores_riesgo:,}",
        delta=(
            f"{porcentaje_riesgo:.1f}% del total"
        ),
        delta_color="inverse",
        help=(
            "Distribuidores cuya probabilidad de mora "
            "es igual o superior al umbral seleccionado."
        )
    )


with col3:

    st.metric(
        label="Probabilidad promedio de mora",
        value=f"{probabilidad_promedio:.1%}",
        help=(
            "Promedio de la probabilidad de mora "
            "para los distribuidores visibles."
        )
    )


# ============================================================
# MEJORA 1
# KPI MÁS CLARO DE EXPOSICIÓN ECONÓMICA
# ============================================================

with col4:

    st.metric(
        label="Saldo expuesto en riesgo",
        value=(
            f"COP "
            f"{valor_en_riesgo / 1_000_000_000:.2f} "
            f"mil M"
        ),
        help=(
            "Suma del saldo actual de los distribuidores "
            "cuya probabilidad de mora supera el "
            "umbral seleccionado."
        )
    )


# ============================================================
# ZONA MEDIA - GRÁFICOS
# ============================================================

st.subheader(
    "Contexto del riesgo"
)

graf_col1, graf_col2 = (
    st.columns(2)
)


# ============================================================
# MEJORA 3
# RIESGO POR CIUDAD MÁS FÁCIL DE INTERPRETAR
# ============================================================

with graf_col1:

    riesgo_ciudad = (
        df_filtrado
        .groupby(
            "ciudad",
            as_index=False
        )
        .agg(
            probabilidad_mora=(
                "probabilidad_mora",
                "mean"
            ),
            distribuidores=(
                "probabilidad_mora",
                "size"
            ),
            distribuidores_en_riesgo=(
                "en_riesgo",
                "sum"
            )
        )
    )


    riesgo_ciudad[
        "porcentaje_en_riesgo"
    ] = (
        riesgo_ciudad[
            "distribuidores_en_riesgo"
        ]
        /
        riesgo_ciudad[
            "distribuidores"
        ]
        * 100
    )


    riesgo_ciudad = (
        riesgo_ciudad
        .sort_values(
            "probabilidad_mora",
            ascending=True
        )
    )


    fig_ciudad = px.bar(
        riesgo_ciudad,
        x="probabilidad_mora",
        y="ciudad",
        orientation="h",
        color="probabilidad_mora",
        color_continuous_scale=[
            [0.0, "#27AE60"],
            [0.5, "#F39C12"],
            [1.0, "#E74C3C"]
        ],
        text=(
            riesgo_ciudad[
                "porcentaje_en_riesgo"
            ]
            .apply(
                lambda x:
                f"{x:.1f}% en riesgo"
            )
        ),
        hover_data={
            "distribuidores": True,
            "distribuidores_en_riesgo": True,
            "porcentaje_en_riesgo": ":.1f",
            "probabilidad_mora": ":.1%"
        },
        labels={
            "probabilidad_mora":
                "Probabilidad promedio de mora",
            "ciudad":
                "Ciudad",
            "distribuidores":
                "Distribuidores analizados",
            "distribuidores_en_riesgo":
                "Distribuidores en riesgo",
            "porcentaje_en_riesgo":
                "% en riesgo"
        },
        title="Riesgo de mora por ciudad"
    )


    fig_ciudad.update_traces(
        textposition="outside",
        cliponaxis=False
    )


    if not riesgo_ciudad.empty:

        max_probabilidad = (
            riesgo_ciudad[
                "probabilidad_mora"
            ].max()
        )

        limite_x = min(
            1,
            max(
                0.20,
                max_probabilidad * 1.40
            )
        )

        fig_ciudad.update_xaxes(
            range=[
                0,
                limite_x
            ]
        )


    fig_ciudad.update_layout(
        coloraxis_colorbar=dict(
            title="Riesgo"
        ),
        xaxis_tickformat=".0%",
        yaxis_title=None,
        margin=dict(
            r=90
        )
    )


    st.plotly_chart(
        fig_ciudad,
        width="stretch"
    )


# ============================================================
# MEJORA 2
# REEMPLAZAR GRÁFICO TEMPORAL POR EXPOSICIÓN ECONÓMICA
# ============================================================

with graf_col2:

    df_en_riesgo = (
        df_filtrado[
            df_filtrado["en_riesgo"]
        ]
        .copy()
    )


    if not df_en_riesgo.empty:

        saldo_riesgo_ciudad = (
            df_en_riesgo
            .groupby(
                "ciudad",
                as_index=False
            )
            .agg(
                saldo_en_riesgo=(
                    "saldo_actual",
                    "sum"
                ),
                distribuidores_en_riesgo=(
                    "probabilidad_mora",
                    "size"
                )
            )
            .sort_values(
                "saldo_en_riesgo",
                ascending=True
            )
        )


        fig_saldo = px.bar(
            saldo_riesgo_ciudad,
            x="saldo_en_riesgo",
            y="ciudad",
            orientation="h",
            color="saldo_en_riesgo",
            color_continuous_scale=[
                [0.0, "#F39C12"],
                [1.0, "#E74C3C"]
            ],
            text=(
                saldo_riesgo_ciudad[
                    "distribuidores_en_riesgo"
                ]
                .apply(
                    lambda x:
                    f"{x} casos"
                )
            ),
            hover_data={
                "distribuidores_en_riesgo": True,
                "saldo_en_riesgo": ":,.0f"
            },
            labels={
                "saldo_en_riesgo":
                    "Saldo expuesto (COP)",
                "ciudad":
                    "Ciudad",
                "distribuidores_en_riesgo":
                    "Distribuidores en riesgo"
            },
            title=(
                "Saldo expuesto en riesgo por ciudad"
            )
        )


        fig_saldo.update_traces(
            textposition="outside",
            cliponaxis=False
        )


        fig_saldo.update_layout(
            yaxis_title=None,
            xaxis_title=(
                "Saldo expuesto (COP)"
            ),
            coloraxis_colorbar=dict(
                title="Exposición"
            ),
            margin=dict(
                r=70
            )
        )


        st.plotly_chart(
            fig_saldo,
            width="stretch"
        )


    else:

        st.info(
            "No hay distribuidores que superen "
            "el umbral de riesgo seleccionado."
        )


# ============================================================
# ZONA INFERIOR - DISTRIBUIDORES PRIORITARIOS
# ============================================================

st.subheader(
    "Distribuidores prioritarios"
)


df_prioritarios = (
    df_filtrado[
        df_filtrado["en_riesgo"]
    ]
    .copy()
)


# ------------------------------------------------------------
# Ordenar primero por probabilidad y luego por saldo
# ------------------------------------------------------------

df_prioritarios = (
    df_prioritarios
    .sort_values(
        by=[
            "probabilidad_mora",
            "saldo_actual"
        ],
        ascending=[
            False,
            False
        ]
    )
)


# ------------------------------------------------------------
# Acción sugerida
# ------------------------------------------------------------

if not df_prioritarios.empty:

    df_prioritarios[
        "accion_sugerida"
    ] = (
        df_prioritarios
        .apply(
            generar_accion,
            axis=1
        )
    )


# ============================================================
# TABLA COMPACTA PARA PANTALLA
# ============================================================

if not df_prioritarios.empty:

    columnas_tabla = [
        "id_distribuidor",
        "ciudad",
        "probabilidad_mora",
        "saldo_actual",
        "nivel_riesgo",
        "accion_sugerida"
    ]


    tabla_prioritarios = (
        df_prioritarios[
            columnas_tabla
        ]
        .copy()
    )


    tabla_prioritarios = (
        tabla_prioritarios
        .rename(
            columns={
                "id_distribuidor":
                    "ID",
                "ciudad":
                    "Ciudad",
                "probabilidad_mora":
                    "Probabilidad de mora",
                "saldo_actual":
                    "Saldo actual",
                "nivel_riesgo":
                    "Nivel de riesgo",
                "accion_sugerida":
                    "Acción sugerida"
            }
        )
    )


    # ProgressColumn trabaja de 0 a 100
    tabla_prioritarios[
        "Probabilidad de mora"
    ] = (
        tabla_prioritarios[
            "Probabilidad de mora"
        ]
        * 100
    )


    st.dataframe(
        tabla_prioritarios,
        hide_index=True,
        width="stretch",
        column_config={

            "Probabilidad de mora":
                st.column_config.ProgressColumn(
                    "Probabilidad de mora",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),

            "Saldo actual":
                st.column_config.NumberColumn(
                    "Saldo actual",
                    format="COP %,.0f"
                )
        }
    )


else:

    st.info(
        "No hay distribuidores prioritarios "
        "para el umbral seleccionado."
    )


# ============================================================
# DESCARGA CSV
# ============================================================

if not df_prioritarios.empty:

    columnas_descarga = [
        "id_distribuidor",
        "ciudad",
        "probabilidad_mora",
        "saldo_actual",
        "cuotas_pagadas",
        "cuotas_totales",
        "score_externo",
        "nivel_riesgo",
        "accion_sugerida"
    ]


    # Solo tomar columnas que existan
    columnas_descarga = [
        columna
        for columna in columnas_descarga
        if columna in df_prioritarios.columns
    ]


    df_descarga = (
        df_prioritarios[
            columnas_descarga
        ]
        .copy()
    )


    if (
        "probabilidad_mora"
        in df_descarga.columns
    ):

        df_descarga[
            "probabilidad_mora"
        ] = (
            df_descarga[
                "probabilidad_mora"
            ]
            * 100
        )


    df_descarga = (
        df_descarga
        .rename(
            columns={
                "id_distribuidor":
                    "ID distribuidor",
                "ciudad":
                    "Ciudad",
                "probabilidad_mora":
                    "Probabilidad de mora (%)",
                "saldo_actual":
                    "Saldo actual",
                "cuotas_pagadas":
                    "Cuotas pagadas",
                "cuotas_totales":
                    "Cuotas totales",
                "score_externo":
                    "Score externo",
                "nivel_riesgo":
                    "Nivel de riesgo",
                "accion_sugerida":
                    "Acción sugerida"
            }
        )
    )


    if (
        "Score externo"
        in df_descarga.columns
    ):

        df_descarga[
            "Score externo"
        ] = (
            df_descarga[
                "Score externo"
            ]
            .where(
                df_descarga[
                    "Score externo"
                ].notna(),
                "Sin dato"
            )
        )


    csv_prioritarios = (
        df_descarga
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )


    st.download_button(
        label=(
            "Descargar lista de "
            "distribuidores prioritarios"
        ),
        data=csv_prioritarios,
        file_name=(
            "solaris_clientes_prioritarios.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# NOTA DE USO
# ============================================================

st.caption(
    "SOLARIS es una herramienta de apoyo a la priorización. "
    "Las predicciones no deben utilizarse como único criterio "
    "para negar, restringir o retirar crédito."
)