import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import unicodedata
import re


# ============================================================
# Configuración general
# ============================================================

st.set_page_config(
    page_title="SOLARIS | Riesgo de Mora",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Funciones auxiliares
# ============================================================

def limpiar_texto(texto):
    """
    Convierte texto a una forma comparable:
    Medellín -> medellin
    B/Quilla -> b quilla
    """
    if pd.isna(texto):
        return ""

    texto = str(texto).strip().lower()

    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

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

    # Barranquilla
    "baq": "Barranquilla",
    "b quilla": "Barranquilla",
    "barranqilla": "Barranquilla",
    "barranquilla": "Barranquilla",

    # Bucaramanga
    "bga": "Bucaramanga",
    "b manga": "Bucaramanga",
    "bucaramanga": "Bucaramanga",

    # Cali
    "cal": "Cali",
    "cali": "Cali",
    "santiago de cali": "Cali",

    # Villavicencio
    "villavo": "Villavicencio",
    "villavicencio": "Villavicencio",

    # Barranquilla
    "b quilla": "Barranquilla",
    "barranquilla": "Barranquilla",

    # Bucaramanga
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

    # Si aparece una ciudad nueva que no está en el mapa,
    # conserva el texto original de forma legible.
    if pd.isna(valor):
        return "Sin información"

    return str(valor).strip().title()


def formatear_pesos(valor):
    """
    Formatea cifras grandes para que no se corten en st.metric.
    """

    if valor >= 1_000_000_000:
        numero = valor / 1_000_000_000
        return f"${numero:.2f}B"

    elif valor >= 1_000_000:
        numero = valor / 1_000_000
        return f"${numero:.1f} M"

    elif valor >= 1_000:
        numero = valor / 1_000
        return f"${numero:.1f} mil"

    return f"${valor:,.0f}"


# ============================================================
# Carga de datos
# ============================================================

@st.cache_data
def cargar_datos():

    try:
        df = pd.read_csv("data/solaris_predicciones.csv")

    except FileNotFoundError:
        st.error(
            "No se encontró data/solaris_predicciones.csv. "
            "Verifica que el archivo esté dentro de dashboard/data."
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Normalización de ciudades
    # --------------------------------------------------------

    if "ciudad" in df.columns:

        # Conservamos la ciudad original por trazabilidad
        df["ciudad_original"] = df["ciudad"]

        # Esta será la ciudad usada por el dashboard
        df["ciudad"] = df["ciudad"].apply(normalizar_ciudad)

    # --------------------------------------------------------
    # Fecha
    # --------------------------------------------------------

    if "fecha_vinculacion" in df.columns:
        df["fecha_vinculacion"] = pd.to_datetime(
            df["fecha_vinculacion"],
            errors="coerce"
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
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )

    return df


df = cargar_datos()


# Evitar seguir ejecutando si no se pudieron cargar datos
if df.empty:
    st.stop()


# ============================================================
# Encabezado
# ============================================================

st.title("SOLARIS | Monitoreo de Riesgo de Mora")

st.caption(
    "Dashboard de apoyo a la gestión de riesgo de crédito "
    "de distribuidores de FrescaMar S.A.S."
)


# ============================================================
# Sidebar
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

    ciudad_seleccionada = st.selectbox(
        "Ciudad",
        ciudades
    )

    # --------------------------------------------------------
    # Rango de fechas
    # --------------------------------------------------------

    rango_fechas = None

    if (
        "fecha_vinculacion" in df.columns
        and df["fecha_vinculacion"].notna().any()
    ):

        fecha_min = df["fecha_vinculacion"].min()
        fecha_max = df["fecha_vinculacion"].max()

        rango_fechas = st.date_input(
            "Rango de fechas",
            value=(
                fecha_min.date(),
                fecha_max.date()
            ),
            min_value=fecha_min.date(),
            max_value=fecha_max.date()
        )

    # --------------------------------------------------------
    # Umbral
    # --------------------------------------------------------

    umbral = st.slider(
        "Umbral de riesgo",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05
    )


# ============================================================
# Aplicar filtros
# ============================================================

df_filtrado = df.copy()


# Ciudad
if ciudad_seleccionada != "Todos":

    df_filtrado = df_filtrado[
        df_filtrado["ciudad"] == ciudad_seleccionada
    ]


# Fecha
if (
    rango_fechas is not None
    and len(rango_fechas) == 2
):

    fecha_inicio = pd.to_datetime(
        rango_fechas[0]
    )

    fecha_fin = pd.to_datetime(
        rango_fechas[1]
    )

    df_filtrado = df_filtrado[
        (
            df_filtrado["fecha_vinculacion"]
            >= fecha_inicio
        )
        &
        (
            df_filtrado["fecha_vinculacion"]
            <= fecha_fin
        )
    ]


# ============================================================
# Clasificación dinámica según umbral
# ============================================================

df_filtrado = df_filtrado.copy()

df_filtrado["en_riesgo"] = (
    df_filtrado["probabilidad_mora"]
    >= umbral
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("Resumen ejecutivo")


total_distribuidores = len(
    df_filtrado
)


distribuidores_riesgo = int(
    df_filtrado["en_riesgo"].sum()
)


probabilidad_promedio = (
    df_filtrado["probabilidad_mora"].mean()
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


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        label="Distribuidores analizados",
        value=f"{total_distribuidores:,}"
    )


with col2:

    st.metric(
        label="Distribuidores en riesgo",
        value=f"{distribuidores_riesgo:,}",
        delta=f"{porcentaje_riesgo:.1f}% del total",
        delta_color="inverse"
    )


with col3:

    st.metric(
        label="Probabilidad promedio de mora",
        value=f"{probabilidad_promedio:.1%}"
    )


with col4:

    st.metric(
        label="Valor en riesgo",
        value=formatear_pesos(
            valor_en_riesgo
        )
    )


# ============================================================
# GRÁFICOS
# ============================================================

st.subheader("Contexto del riesgo")

graf_col1, graf_col2 = st.columns(2)


# ============================================================
# Riesgo promedio por ciudad
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
            )
        )
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
        hover_data={
            "distribuidores": True,
            "probabilidad_mora": ":.1%"
        },
        labels={
            "probabilidad_mora":
                "Probabilidad promedio de mora",
            "ciudad":
                "Ciudad",
            "distribuidores":
                "Distribuidores"
        },
        title="Riesgo promedio por ciudad"
    )


    fig_ciudad.update_layout(
        coloraxis_colorbar=dict(
            title="Riesgo"
        ),
        xaxis_tickformat=".0%",
        yaxis_title=None
    )


    st.plotly_chart(
        fig_ciudad,
        use_container_width=True
    )


# ============================================================
# Riesgo según fecha de vinculación
# ============================================================

with graf_col2:

    if (
        "fecha_vinculacion"
        in df_filtrado.columns
        and
        df_filtrado[
            "fecha_vinculacion"
        ].notna().any()
    ):

        df_temporal = (
            df_filtrado
            .dropna(
                subset=[
                    "fecha_vinculacion"
                ]
            )
            .copy()
        )


        # Para evitar una línea demasiado ruidosa,
        # agrupamos por trimestre.
        df_temporal[
            "periodo_vinculacion"
        ] = (
            df_temporal[
                "fecha_vinculacion"
            ]
            .dt.to_period("Q")
            .dt.to_timestamp()
        )


        riesgo_temporal = (
            df_temporal
            .groupby(
                "periodo_vinculacion",
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
                )
            )
        )


        fig_temporal = px.line(
            riesgo_temporal,
            x="periodo_vinculacion",
            y="probabilidad_mora",
            markers=True,
            hover_data={
                "distribuidores": True,
                "probabilidad_mora": ":.1%"
            },
            labels={
                "periodo_vinculacion":
                    "Fecha de vinculación",
                "probabilidad_mora":
                    "Probabilidad promedio de mora",
                "distribuidores":
                    "Distribuidores"
            },
            title=(
                "Riesgo promedio según "
                "fecha de vinculación"
            )
        )


        fig_temporal.update_traces(
            line_color="#2980B9"
        )


        fig_temporal.add_hline(
            y=umbral,
            line_dash="dash",
            line_color="#E74C3C",
            annotation_text=(
                f"Umbral: {umbral:.0%}"
            ),
            annotation_position="top right"
        )


        fig_temporal.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title=(
                "Probabilidad promedio de mora"
            ),
            xaxis_title=(
                "Fecha de vinculación"
            )
        )


        st.plotly_chart(
            fig_temporal,
            use_container_width=True
        )


    else:

        st.info(
            "No existen fechas suficientes "
            "para construir el gráfico temporal."
        )


# ============================================================
# TABLA DE DISTRIBUIDORES PRIORITARIOS
# ============================================================

st.subheader("Distribuidores prioritarios")


# Crear identificador visual
if "id_distribuidor" not in df_filtrado.columns:

    df_filtrado = (
        df_filtrado
        .reset_index(drop=False)
        .rename(
            columns={
                "index":
                    "id_distribuidor"
            }
        )
    )

    df_filtrado[
        "id_distribuidor"
    ] = (
        df_filtrado[
            "id_distribuidor"
        ]
        + 1
    )


# Solo casos que superan el umbral
df_prioritarios = (
    df_filtrado[
        df_filtrado["en_riesgo"]
    ]
    .copy()
)


# Ordenar por probabilidad y saldo
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


# ============================================================
# Acción sugerida dinámica
# ============================================================

def generar_accion(fila):

    probabilidad = (
        fila["probabilidad_mora"]
    )

    if probabilidad >= 0.85:
        return "Revisión prioritaria"

    elif probabilidad >= 0.70:
        return "Contactar y revisar pagos"

    else:
        return "Monitorear"


if not df_prioritarios.empty:

    df_prioritarios[
        "accion_sugerida"
    ] = (
        df_prioritarios.apply(
            generar_accion,
            axis=1
        )
    )


# ============================================================
# Tabla compacta para pantalla
# ============================================================

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
    tabla_prioritarios.rename(
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
    use_container_width=True,
    hide_index=True,
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
                format="$ %,.0f"
            )
    }
)


# ============================================================
# CSV COMPLETO PARA DESCARGA
# ============================================================

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


df_descarga = (
    df_prioritarios[
        columnas_descarga
    ]
    .copy()
)


df_descarga[
    "probabilidad_mora"
] = (
    df_descarga[
        "probabilidad_mora"
    ]
    * 100
)


df_descarga = (
    df_descarga.rename(
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


# Mostrar los faltantes de score como texto
if "Score externo" in df_descarga.columns:

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