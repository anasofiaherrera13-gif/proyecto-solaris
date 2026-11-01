import streamlit as st
import pandas as pd
import plotly.express as px
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
# PALETA ACCESIBLE
# ============================================================
# Tonos más oscuros para mejorar contraste sobre fondo blanco.
# La información nunca depende únicamente del color.

ROJO = "#C0392B"
VERDE = "#1E8449"
NARANJA = "#AF601A"
AZUL = "#2471A3"
GRIS = "#5D6D7E"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def limpiar_texto(texto):

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

        # Otras
        "ibague": "Ibagué",
        "cartagena": "Cartagena",
    }

    if clave in mapa_ciudades:
        return mapa_ciudades[clave]

    if pd.isna(valor):
        return "Sin información"

    return str(valor).strip().title()


def generar_accion(fila):

    probabilidad = fila["probabilidad_mora"]

    if probabilidad >= 0.85:
        return "Revisión prioritaria"

    elif probabilidad >= 0.70:
        return "Contactar y revisar pagos"

    elif probabilidad >= 0.40:
        return "Monitorear"

    else:
        return "Seguimiento habitual"


def formatear_cop(valor):

    if pd.isna(valor):
        return "Sin dato"

    if valor >= 1_000_000_000:
        return f"COP {valor / 1_000_000_000:.2f} mil M"

    elif valor >= 1_000_000:
        return f"COP {valor / 1_000_000:.1f} M"

    elif valor >= 1_000:
        return f"COP {valor / 1_000:.1f} mil"

    return f"COP {valor:,.0f}"


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
            "data/solaris_predicciones.csv."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Identificador
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

        df["ciudad_original"] = (
            df["ciudad"]
        )

        df["ciudad"] = (
            df["ciudad"]
            .apply(normalizar_ciudad)
        )

    # --------------------------------------------------------
    # Variables numéricas
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
# PASO 7:
# Se mantienen únicamente los filtros que modifican
# directamente una decisión de negocio.
#
# Se elimina fecha_vinculacion porque no representa la
# fecha de scoring y no cambia claramente la priorización.
# ============================================================

with st.sidebar:

    st.header(
        "Filtros de decisión"
    )

    # --------------------------------------------------------
    # FILTRO 1: CIUDAD
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

    ciudad_seleccionada = st.selectbox(
        "Mostrando ciudad",
        ciudades,
        index=0,
        help=(
            "Selecciona una ciudad para concentrar "
            "el análisis geográfico. "
            "Por defecto se muestra toda la cartera."
        )
    )

    # --------------------------------------------------------
    # FILTRO 2: UMBRAL DE RIESGO
    # --------------------------------------------------------

    umbral = st.slider(
        "Umbral para definir prioridad",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        format="%.2f",
        help=(
            "Un distribuidor se considera prioritario "
            "cuando su probabilidad de mora es igual "
            "o superior a este valor."
        )
    )

    st.caption(
        f"Mostrando: "
        f"{'todas las ciudades' if ciudad_seleccionada == 'Todos' else ciudad_seleccionada} "
        f"· Prioridad desde {umbral:.0%}"
    )


# ============================================================
# APLICAR FILTROS
# ============================================================

df_filtrado = df.copy()


if ciudad_seleccionada != "Todos":

    df_filtrado = (
        df_filtrado[
            df_filtrado["ciudad"]
            == ciudad_seleccionada
        ]
    )


# ============================================================
# CLASIFICACIÓN SEGÚN UMBRAL
# ============================================================

df_filtrado = df_filtrado.copy()

df_filtrado["en_riesgo"] = (
    df_filtrado["probabilidad_mora"]
    >= umbral
)


# ============================================================
# 5 VISTAS FUNCIONALES
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Identificar riesgo",
        "Comparar ciudades",
        "Priorizar exposición",
        "Revisar distribuidor",
        "Evaluar modelo"
    ]
)


# ============================================================
# VISTA 1
# PREGUNTA:
# ¿Cuántos distribuidores requieren atención y cuáles primero?
# ============================================================

with tab1:

    st.header(
        "Identificar distribuidores en riesgo"
    )

    st.caption(
        "Esta vista muestra el volumen y los casos con mayor "
        "probabilidad de mora para ayudarte a decidir qué "
        "distribuidores deben revisarse primero."
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

    saldo_expuesto = (
        df_filtrado.loc[
            df_filtrado[
                "en_riesgo"
            ],
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
            "Distribuidores analizados",
            f"{total_distribuidores:,}"
        )

    with col2:

        st.metric(
            "Distribuidores en riesgo",
            f"{distribuidores_riesgo:,}",
            delta=(
                f"{porcentaje_riesgo:.1f}% "
                f"del total"
            ),
            delta_color="inverse",
            help=(
                "▲ Un valor mayor implica más "
                "distribuidores que requieren atención."
            )
        )

    with col3:

        st.metric(
            "Probabilidad promedio de mora",
            f"{probabilidad_promedio:.1%}"
        )

    with col4:

        st.metric(
            "Saldo expuesto en riesgo",
            formatear_cop(
                saldo_expuesto
            ),
            help=(
                "Suma del saldo actual de los distribuidores "
                "que superan el umbral de prioridad."
            )
        )

    st.subheader(
        "Casos que requieren atención"
    )

    df_prioritarios = (
        df_filtrado[
            df_filtrado["en_riesgo"]
        ]
        .copy()
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

        tabla_prioritarios = (
            df_prioritarios[
                [
                    "id_distribuidor",
                    "ciudad",
                    "probabilidad_mora",
                    "saldo_actual",
                    "nivel_riesgo",
                    "accion_sugerida"
                ]
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

        csv_prioritarios = (
            tabla_prioritarios
            .to_csv(
                index=False
            )
            .encode(
                "utf-8-sig"
            )
        )

        st.download_button(
            label=(
                "Descargar distribuidores "
                "prioritarios"
            ),
            data=csv_prioritarios,
            file_name=(
                "solaris_clientes_prioritarios.csv"
            ),
            mime="text/csv"
        )

    else:

        st.info(
            "✓ No hay distribuidores que superen "
            "el umbral seleccionado."
        )


# ============================================================
# VISTA 2
# PREGUNTA:
# ¿En qué ciudades se concentra un mayor riesgo?
# ============================================================

with tab2:

    st.header(
        "Comparar ciudades"
    )

    st.caption(
        "Esta vista muestra cómo cambia el riesgo entre "
        "ciudades para ayudarte a decidir dónde concentrar "
        "el seguimiento de cartera."
    )

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

        # Colores semánticos con tonos más oscuros
        color_continuous_scale=[
            [0.0, VERDE],
            [0.5, NARANJA],
            [1.0, ROJO]
        ],

        text=(
            riesgo_ciudad[
                "porcentaje_en_riesgo"
            ]
            .apply(
                lambda x:
                f"{x:.1f}% supera el umbral"
            )
        ),

        hover_data={
            "distribuidores":
                True,
            "distribuidores_en_riesgo":
                True,
            "porcentaje_en_riesgo":
                ":.1f",
            "probabilidad_mora":
                ":.1%"
        },

        labels={
            "probabilidad_mora":
                "Probabilidad promedio de mora",
            "ciudad":
                "Ciudad",
            "distribuidores":
                "Distribuidores analizados",
            "distribuidores_en_riesgo":
                "Distribuidores que superan el umbral",
            "porcentaje_en_riesgo":
                "% que supera el umbral"
        },

        title=(
            "Probabilidad promedio de mora por ciudad"
        )
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
                max_probabilidad * 1.55
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
            title="Nivel de riesgo"
        ),
        xaxis_tickformat=".0%",
        yaxis_title=None,
        margin=dict(
            r=160
        )
    )

    st.plotly_chart(
        fig_ciudad,
        width="stretch"
    )

    if not riesgo_ciudad.empty:

        ciudad_mayor_riesgo = (
            riesgo_ciudad
            .sort_values(
                "probabilidad_mora",
                ascending=False
            )
            .iloc[0]
        )

        st.warning(
            f"⚠ Atención prioritaria: "
            f"{ciudad_mayor_riesgo['ciudad']} "
            f"presenta la mayor probabilidad promedio "
            f"de mora "
            f"({ciudad_mayor_riesgo['probabilidad_mora']:.1%}). "
            f"Además, "
            f"{ciudad_mayor_riesgo['porcentaje_en_riesgo']:.1f}% "
            f"de sus distribuidores supera el "
            f"umbral actual de {umbral:.0%}."
        )


# ============================================================
# VISTA 3
# PREGUNTA:
# ¿Dónde está concentrado el mayor saldo expuesto?
# ============================================================

with tab3:

    st.header(
        "Priorizar exposición económica"
    )

    st.caption(
        "Esta vista muestra cuánto saldo está asociado a "
        "distribuidores en riesgo para ayudarte a decidir "
        "dónde priorizar recursos de seguimiento."
    )

    df_en_riesgo = (
        df_filtrado[
            df_filtrado[
                "en_riesgo"
            ]
        ]
        .copy()
    )

    if not df_en_riesgo.empty:

        saldo_ciudad = (
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
            saldo_ciudad,
            x="saldo_en_riesgo",
            y="ciudad",
            orientation="h",
            color="saldo_en_riesgo",

            color_continuous_scale=[
                [0.0, NARANJA],
                [1.0, ROJO]
            ],

            text=(
                saldo_ciudad[
                    "distribuidores_en_riesgo"
                ]
                .apply(
                    lambda x:
                    f"{x} casos prioritarios"
                )
            ),

            hover_data={
                "distribuidores_en_riesgo":
                    True,
                "saldo_en_riesgo":
                    ":,.0f"
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
                r=100
            )
        )

        st.plotly_chart(
            fig_saldo,
            width="stretch"
        )

        ciudad_mayor_exposicion = (
            saldo_ciudad
            .sort_values(
                "saldo_en_riesgo",
                ascending=False
            )
            .iloc[0]
        )

        st.warning(
            f"⚠ Mayor exposición económica: "
            f"{ciudad_mayor_exposicion['ciudad']} "
            f"concentra "
            f"{formatear_cop(ciudad_mayor_exposicion['saldo_en_riesgo'])} "
            f"y "
            f"{int(ciudad_mayor_exposicion['distribuidores_en_riesgo'])} "
            f"casos prioritarios."
        )

    else:

        st.success(
            "✓ No existen distribuidores que superen "
            "el umbral seleccionado."
        )


# ============================================================
# VISTA 4
# PREGUNTA:
# ¿Un distribuidor específico requiere revisión prioritaria?
# ============================================================

with tab4:

    st.header(
        "Revisar un distribuidor"
    )

    st.caption(
        "Esta vista muestra la situación individual de un "
        "distribuidor para ayudarte a decidir si requiere "
        "una revisión prioritaria."
    )

    distribuidores_ordenados = (
        df_filtrado
        .sort_values(
            "probabilidad_mora",
            ascending=False
        )
        .copy()
    )

    ids_disponibles = (
        distribuidores_ordenados[
            "id_distribuidor"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    if len(ids_disponibles) > 0:

        id_seleccionado = (
            st.selectbox(
                "Selecciona un distribuidor",
                ids_disponibles,
                index=0,
                help=(
                    "La lista aparece ordenada desde "
                    "el distribuidor con mayor probabilidad "
                    "de mora."
                )
            )
        )

        caso = (
            df_filtrado[
                df_filtrado[
                    "id_distribuidor"
                ]
                == id_seleccionado
            ]
            .iloc[0]
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "Probabilidad de mora",
                f"{caso['probabilidad_mora']:.1%}"
            )

        with c2:

            st.metric(
                "Saldo actual",
                formatear_cop(
                    caso["saldo_actual"]
                )
            )

        with c3:

            cuotas_pagadas = (
                caso.get(
                    "cuotas_pagadas",
                    np.nan
                )
            )

            cuotas_totales = (
                caso.get(
                    "cuotas_totales",
                    np.nan
                )
            )

            if (
                pd.notna(cuotas_pagadas)
                and
                pd.notna(cuotas_totales)
            ):

                valor_cuotas = (
                    f"{cuotas_pagadas:.0f} "
                    f"de "
                    f"{cuotas_totales:.0f}"
                )

            else:

                valor_cuotas = (
                    "Sin dato"
                )

            st.metric(
                "Cuotas pagadas",
                valor_cuotas
            )

        with c4:

            score = (
                caso.get(
                    "score_externo",
                    np.nan
                )
            )

            if pd.notna(score):
                score_mostrar = f"{score:.0f}"

            else:
                score_mostrar = "Sin dato"

            st.metric(
                "Score externo",
                score_mostrar
            )

        st.subheader(
            "Perfil del distribuidor"
        )

        perfil = pd.DataFrame(
            {
                "Variable": [
                    "Ciudad",
                    "Edad",
                    "Nivel educativo",
                    "Ingreso mensual",
                    "Saldo actual",
                    "Cuotas pagadas",
                    "Cuotas totales",
                    "Probabilidad de mora"
                ],

                "Valor": [

                    caso.get(
                        "ciudad",
                        "Sin dato"
                    ),

                    (
                        f"{caso['edad']:.0f}"
                        if pd.notna(
                            caso.get(
                                "edad",
                                np.nan
                            )
                        )
                        else "Sin dato"
                    ),

                    caso.get(
                        "nivel_educativo",
                        "Sin dato"
                    ),

                    (
                        formatear_cop(
                            caso[
                                "ingreso_mensual"
                            ]
                        )
                        if pd.notna(
                            caso.get(
                                "ingreso_mensual",
                                np.nan
                            )
                        )
                        else "Sin dato"
                    ),

                    formatear_cop(
                        caso[
                            "saldo_actual"
                        ]
                    ),

                    (
                        f"{caso['cuotas_pagadas']:.0f}"
                        if pd.notna(
                            caso.get(
                                "cuotas_pagadas",
                                np.nan
                            )
                        )
                        else "Sin dato"
                    ),

                    (
                        f"{caso['cuotas_totales']:.0f}"
                        if pd.notna(
                            caso.get(
                                "cuotas_totales",
                                np.nan
                            )
                        )
                        else "Sin dato"
                    ),

                    f"{caso['probabilidad_mora']:.1%}"
                ]
            }
        )

        st.dataframe(
            perfil,
            hide_index=True,
            width="stretch"
        )

        accion = (
            generar_accion(
                caso
            )
        )

        if (
            caso[
                "probabilidad_mora"
            ]
            >= umbral
        ):

            st.error(
                f"⚠ REQUIERE ATENCIÓN. "
                f"Acción sugerida: {accion}. "
                f"Este distribuidor supera el "
                f"umbral actual de {umbral:.0%}. "
                f"La decisión final requiere revisión humana."
            )

        else:

            st.success(
                f"✓ POR DEBAJO DEL UMBRAL. "
                f"Acción sugerida: {accion}. "
                f"Este distribuidor está por debajo "
                f"del umbral actual de {umbral:.0%}."
            )

    else:

        st.info(
            "No hay distribuidores disponibles "
            "con los filtros seleccionados."
        )


# ============================================================
# VISTA 5
# PREGUNTA:
# ¿El modelo tiene desempeño suficiente para apoyar decisiones?
# ============================================================

with tab5:

    st.header(
        "Evaluar el modelo"
    )

    st.caption(
        "Esta vista muestra el desempeño validado de SOLARIS "
        "para ayudarte a decidir bajo qué condiciones puede "
        "utilizarse como apoyo a la gestión de cartera."
    )

    # --------------------------------------------------------
    # Métricas obtenidas en Módulo 4
    # --------------------------------------------------------

    auc = 0.9588
    precision = 0.9076
    recall = 0.6278
    meta_recall = 0.75

    m1, m2, m3, m4 = (
        st.columns(4)
    )

    with m1:

        st.metric(
            "ROC-AUC",
            f"{auc:.1%}"
        )

    with m2:

        st.metric(
            "Precisión en Moroso",
            f"{precision:.1%}"
        )

    with m3:

        st.metric(
            "Recall en Moroso",
            f"{recall:.1%}"
        )

    with m4:

        st.metric(
            "Meta de recall",
            f"{meta_recall:.0%}"
        )

    st.caption(
        "Leyenda accesible: "
        "⚠ = requiere atención · "
        "✓ = objetivo de negocio"
    )

    # ========================================================
    # ACCESIBILIDAD
    # No dependemos únicamente de rojo/verde.
    # Las categorías contienen iconos y texto explícito.
    # ========================================================

    comparacion_recall = pd.DataFrame(
        {
            "Indicador": [
                "⚠ Recall actual",
                "✓ Meta de negocio"
            ],

            "Valor": [
                recall * 100,
                meta_recall * 100
            ]
        }
    )

    fig_modelo = px.bar(
        comparacion_recall,
        x="Indicador",
        y="Valor",
        color="Indicador",

        color_discrete_map={
            "⚠ Recall actual":
                ROJO,
            "✓ Meta de negocio":
                VERDE
        },

        text="Valor",

        labels={
            "Valor":
                "Porcentaje (%)"
        },

        title=(
            "Recall actual vs meta de negocio"
        )
    )

    fig_modelo.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_modelo.update_yaxes(
        range=[
            0,
            100
        ]
    )

    fig_modelo.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_modelo,
        width="stretch"
    )

    if recall < meta_recall:

        diferencia = (
            meta_recall
            - recall
        )

        st.warning(
            f"⚠ El recall actual está "
            f"{diferencia:.1%} por debajo de la meta. "
            f"Decisión recomendada: utilizar SOLARIS "
            f"como herramienta de priorización y apoyo "
            f"a la revisión humana, pero no como "
            f"mecanismo automático de rechazo o "
            f"restricción de crédito."
        )

    else:

        st.success(
            "✓ El recall alcanza la meta de negocio."
        )


# ============================================================
# NOTA GENERAL
# ============================================================

st.divider()

st.caption(
    "SOLARIS es una herramienta de apoyo a la priorización. "
    "Las predicciones no deben utilizarse como único criterio "
    "para negar, restringir o retirar crédito."
)