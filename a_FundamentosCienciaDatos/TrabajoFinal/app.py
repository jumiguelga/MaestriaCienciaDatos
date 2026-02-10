# =============================================================================
# IMPORTS & CONFIG
# =============================================================================
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def donut_issue_chart(count_issue: int, total: int, titulo: str, etiqueta_issue: str, etiqueta_ok: str) -> None:
    """
    Muestra un gráfico tipo donut (pie chart con agujero) para resumir
    la proporción de filas con problema vs filas sin problema.
    """
    if total <= 0:
        return

    count_issue = int(count_issue)
    total = int(total)
    count_ok = max(total - count_issue, 0)

    data = pd.DataFrame(
        {
            "estado": [etiqueta_issue, etiqueta_ok],
            "filas": [count_issue, count_ok],
        }
    )

    fig = px.pie(
        data,
        names="estado",
        values="filas",
        hole=0.6,
        color="estado",
        color_discrete_map={
            etiqueta_issue: "#e74c3c",  # rojo para filas con problema
            etiqueta_ok: "#2ecc71",     # verde para filas sin problema
        },
    )
    fig.update_traces(textinfo="percent", textfont_size=14)
    fig.update_layout(
        title=titulo,
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)


# Configuración de la página
st.set_page_config(
    page_title="Panel inteligente",
    page_icon="🤖",
    layout="wide"
)

# =============================================================================
# SIDEBAR: DATA LOADING
# =============================================================================

# Sidebar for data input
st.sidebar.title("📁 Carga de Datos")

# Option selector
data_source = st.sidebar.radio(
    "Selecciona el tipo de archivo:",
    ["Archivo CSV", "Archivo JSON", "URL"]
)

df = None

# Handle CSV file upload
if data_source == "Archivo CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Cargar archivo CSV",
        type=['csv'],
        help="Cargar un archivo CSV para analizar"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Archivo CSV cargado exitosamente!")
        except Exception as e:
            st.sidebar.error(f"Error al cargar el archivo CSV: {str(e)}")

# Handle JSON file upload
elif data_source == "Archivo JSON":
    uploaded_file = st.sidebar.file_uploader(
        "Cargar archivo JSON",
        type=['json'],
        help="Cargar un archivo JSON para analizar"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_json(uploaded_file)
            st.sidebar.success("Archivo JSON cargado exitosamente!")
        except Exception as e:
            st.sidebar.error(f"Error al cargar el archivo JSON: {str(e)}")

# Handle URL input
elif data_source == "URL":
    url = st.sidebar.text_input(
        "Ingresar URL",
        placeholder="https://example.com/data.csv",
        help="Ingresar URL a un archivo CSV o JSON"
    )
    if url:
        try:
            if url.endswith('.json'):
                df = pd.read_json(url)
            else:
                df = pd.read_csv(url)
            st.sidebar.success("Datos cargados desde la URL exitosamente!")
        except Exception as e:
            st.sidebar.error(f"Error al cargar desde la URL: {str(e)}")

# =============================================================================
# SIDEBAR: GLOBAL FILTERS / DATA TREATMENT
# =============================================================================

delete_duplicates = False
imputation_method = "Sin imputación"
treat_outliers = False

if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Filtros y tratamiento de datos")

    delete_duplicates = st.sidebar.checkbox(
        "Eliminar filas duplicadas",
        value=False,
        help="Si se marca, se eliminarán las filas detectadas como duplicadas.",
    )

    imputation_method = st.sidebar.selectbox(
        "Método de imputación numérica",
        options=["Sin imputación", "Media", "Mediana", "Cero"],
        help="Cómo imputar valores numéricos faltantes (NaN).",
    )

    treat_outliers = st.sidebar.checkbox(
        "Eliminar filas con valores atípicos",
        value=False,
        help="Si se marca, se eliminarán las filas marcadas con valores atípicos.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Columna índice / identificador")
    index_column = st.sidebar.selectbox(
        "Selecciona la columna identificadora (no se usará para análisis estadístico).",
        options=df.columns.tolist(),
        index=0,
    )
    st.session_state["index_column"] = index_column

# =============================================================================
# MAIN: DISPLAY (df.info, df.head)
# =============================================================================

# Área principal
st.title("📊 Panel inteligente")

if df is not None:
    tab_ingesta, tab_visualizacion, tab_insights_ia = st.tabs(["Ingesta y Procesamiento de Datos (ETL)", "Visualización Dinámica (EDA)", "Insights de IA"])

    with tab_ingesta:
        # 1) Reemplazar vacíos por NaN y 2) Sanitizar todas las columnas de texto (trim + minúsculas)
        df_proc = df.replace("", np.nan).replace(r"^\s*$", np.nan, regex=True)
        # Incluir object, string y category; además incluir explícitamente dtype == object por compatibilidad
        object_cols = [
            c for c in df_proc.columns
            if (
                df_proc[c].dtype == object
                or df_proc[c].dtype.name == "object"
                or pd.api.types.is_string_dtype(df_proc[c])
                or str(df_proc[c].dtype) == "category"
            )
        ]
        for col in object_cols:
            s = df_proc[col]
            if s.dtype.name == "category":
                s = s.astype(object)
            # Sanitizar todo: convertir a string, strip y lower (así se unifican "Fiber optic" y " Fiber optic ")
            s = s.astype(str).str.strip().str.lower()
            s = s.replace("nan", np.nan)
            df_proc[col] = s

        st.subheader("1. Reemplazo de valores vacíos por NaN")
        st.caption("Se convirtieron cadenas vacías y espacios en blanco a NaN.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas", df_proc.shape[0])
        with col2:
            st.metric("Columnas", df_proc.shape[1])
        with col3:
            st.metric("Valores NaN tras reemplazo", int(df_proc.isna().sum().sum()))

        st.subheader("2. Sanitización de columnas categóricas")
        st.caption("Se aplicó trim (espacios al inicio/fin) y conversión a minúsculas en todas las columnas no numéricas.")
        st.metric("Columnas sanitizadas", len(object_cols))
        if object_cols:
            st.caption(f"Columnas: {', '.join(object_cols)}.")

        # Resumen del dataset (sobre df_proc ya sanitizado). Identificar booleanas por valores únicos (genérico).
        st.subheader("Información del dataset")
        n_rows, n_cols = len(df_proc), len(df_proc.columns)
        numeric_cols = set(df_proc.select_dtypes(include=[np.number]).columns)
        bool_cols = set(df_proc.select_dtypes(include=[bool]).columns)
        other_cols = [c for c in df_proc.columns if c not in numeric_cols and c not in bool_cols]
        # Valores a ignorar al contar únicos (no cuentan como "valor real")
        skip_vals = {"", "nan", "n/a", "none", "na"}
        def get_effective_unique_set(ser):
            """Conjunto de valores únicos normalizados (strip, lower), sin skip_vals. Para numéricos devuelve set de enteros."""
            if ser.dtype.kind in "iufb":
                return set(pd.Series(ser.dropna().astype(int).unique()).astype(str))
            s = ser.dropna().astype(str).str.strip().str.lower()
            s = s[~s.isin(skip_vals)]
            return set(s.unique())
        # Booleana = exactamente 2 valores únicos efectivos (yes/no, sí/no, 0/1, etc.) — sin nombres quemados
        infer_bool_cols = [c for c in other_cols if len(get_effective_unique_set(df_proc[c])) == 2]
        numeric_bool_cols = [c for c in numeric_cols if get_effective_unique_set(df_proc[c]) == {"0", "1"}]
        categorical_cols = [c for c in other_cols if c not in infer_bool_cols]
        n_numeric = len(numeric_cols)
        n_bool = len(bool_cols) + len(infer_bool_cols) + len(numeric_bool_cols)
        n_categorical = len(categorical_cols)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total columnas", n_cols)
        with c2:
            st.metric("Total filas", n_rows)
        with c3:
            st.metric("Columnas categóricas", n_categorical)
        with c4:
            st.metric("Columnas numéricas", n_numeric)
        with c5:
            st.metric("Columnas booleanas", n_bool)
        if infer_bool_cols or numeric_bool_cols:
            parts = []
            if infer_bool_cols:
                parts.append(f"Inferidas (yes/no, sí/no, etc.): {', '.join(infer_bool_cols)}.")
            if numeric_bool_cols:
                parts.append(f"Numéricas 0/1: {', '.join(numeric_bool_cols)}.")
            st.caption("Se consideran booleanas las columnas con solo dos valores. " + " ".join(parts))

        # Tabla: convertir columnas inferidas como booleanas a 1 y 0 (orden estable: sorted -> 0, 1)
        df_display = df_proc.copy()
        for col in infer_bool_cols:
            unicos = sorted(df_display[col].dropna().unique().tolist(), key=str)
            if len(unicos) == 2:
                df_display[col] = df_display[col].map({unicos[0]: 0, unicos[1]: 1})
        st.subheader("Primeras 5 filas (tras sanitización y conversión a 1/0 en booleanas)")
        st.dataframe(df_display.head(), use_container_width=True)

        # Debug: mismos valores que la tabla anterior (sanitizados + booleanas como 0/1)
        with st.expander("🔍 Debug: valores únicos por columna (valor → conteo)"):
            st.caption("Valores actualizados: sanitización aplicada y columnas booleanas mostradas como 0 y 1.")
            for col in df_display.columns:
                st.markdown(f"**{col}**")
                counts = df_display[col].value_counts(dropna=False)
                st.dataframe(counts.rename_axis("valor").reset_index(name="conteo"), use_container_width=True, hide_index=True)
                st.divider()

        # Columna Nulos: True si alguna columna es nula en esa fila
        df_proc["Nulos"] = df_proc.drop(columns=["Nulos"], errors="ignore").isna().any(axis=1)
        n_filas_nulas = df_proc["Nulos"].sum()

        st.subheader("3. Detección de nulos por fila")
        st.caption('Columna "Nulos": True si la fila tiene al menos un valor nulo.')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas con al menos un nulo", int(n_filas_nulas))
        with col2:
            st.metric("Filas sin nulos", int((~df_proc["Nulos"]).sum()))
        with col3:
            pct = 100 * n_filas_nulas / len(df_proc) if len(df_proc) else 0
            st.metric("% filas con nulos", f"{pct:.1f}%")
        
        # Columna Duplicados
        df_proc["Duplicados"] = df_proc.drop(columns=["Nulos", "Duplicados"], errors="ignore").duplicated(keep=False)
        n_dup = df_proc["Duplicados"].sum()

        st.subheader("4. Detección de duplicados por fila")
        st.caption('Columna "Duplicados": True si la fila está duplicada (considerando todas las columnas originales).')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas duplicadas", int(n_dup))
        with col2:
            st.metric("Filas únicas", int((~df_proc["Duplicados"]).sum()))
        with col3:
            pct_dup = 100 * n_dup / len(df_proc) if len(df_proc) else 0
            st.metric("% filas duplicadas", f"{pct_dup:.1f}%")
        
        # Valores atípicos (solo columnas numéricas, método IQR)
        numericas = [c for c in df_proc.select_dtypes(include=[np.number]).columns if c not in ("Nulos", "Duplicados", "Valores Atípicos")]
        atipico_fila = pd.Series(False, index=df_proc.index)
        if numericas:
            for col in numericas:
                Q1 = df_proc[col].quantile(0.25)
                Q3 = df_proc[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    fuera = (df_proc[col] < Q1 - 1.5 * IQR) | (df_proc[col] > Q3 + 1.5 * IQR)
                    atipico_fila = atipico_fila | fuera
        df_proc["Valores Atípicos"] = atipico_fila
        n_atipicos = df_proc["Valores Atípicos"].sum()

        st.subheader("5. Detección de valores atípicos por fila")
        st.caption('Columna "Valores Atípicos": True si en esa fila algún valor numérico es atípico (método IQR).')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas con al menos un atípico", int(n_atipicos))
        with col2:
            st.metric("Filas sin atípicos", int((~df_proc["Valores Atípicos"]).sum()))
        with col3:
            pct_at = 100 * n_atipicos / len(df_proc) if len(df_proc) else 0
            st.metric("% filas con atípicos", f"{pct_at:.1f}%")

        # Gráficos resumen por tipo de problema en una sola fila
        c_nulos, c_dup, c_atip = st.columns(3)
        with c_nulos:
            donut_issue_chart(
                count_issue=n_filas_nulas,
                total=len(df_proc),
                titulo="Distribución de filas con nulos",
                etiqueta_issue="Con nulos",
                etiqueta_ok="Sin nulos",
            )
        with c_dup:
            donut_issue_chart(
                count_issue=n_dup,
                total=len(df_proc),
                titulo="Distribución de filas duplicadas",
                etiqueta_issue="Duplicadas",
                etiqueta_ok="Únicas",
            )
        with c_atip:
            donut_issue_chart(
                count_issue=n_atipicos,
                total=len(df_proc),
                titulo="Distribución de filas con valores atípicos",
                etiqueta_issue="Con atípicos",
                etiqueta_ok="Sin atípicos",
            )

        # Aplicar filtros globales configurados en la barra lateral
        df_filtrado = df_proc.copy()

        # Imputación numérica
        valores_imputados = 0
        if imputation_method != "Sin imputación":
            columnas_numericas_tratadas = [
                c
                for c in df_filtrado.select_dtypes(include=[np.number]).columns
                if c not in ("Nulos", "Duplicados", "Valores Atípicos")
            ]
            for col in columnas_numericas_tratadas:
                n_nan_antes = df_filtrado[col].isna().sum()
                if n_nan_antes == 0:
                    continue
                if imputation_method == "Media":
                    valor = df_filtrado[col].mean()
                elif imputation_method == "Mediana":
                    valor = df_filtrado[col].median()
                else:  # "Cero"
                    valor = 0
                df_filtrado[col] = df_filtrado[col].fillna(valor)
                valores_imputados += n_nan_antes

        filas_antes_filtros = len(df_filtrado)

        # Eliminación de duplicados
        filas_eliminadas_duplicados = 0
        if delete_duplicates:
            if "Duplicados" in df_filtrado.columns:
                filas_eliminadas_duplicados = int(df_filtrado["Duplicados"].sum())
                df_filtrado = df_filtrado[~df_filtrado["Duplicados"]]

        # Eliminación de filas con valores atípicos
        filas_eliminadas_atipicos = 0
        if treat_outliers:
            if "Valores Atípicos" in df_filtrado.columns:
                filas_eliminadas_atipicos = int(df_filtrado["Valores Atípicos"].sum())
                df_filtrado = df_filtrado[~df_filtrado["Valores Atípicos"]]

        filas_despues_filtros = len(df_filtrado)

        # Guardar dataset filtrado en sesión para futuras pestañas
        st.session_state["df_filtrado"] = df_filtrado

        st.divider()
        st.subheader("Resumen del procesamiento")
        st.success(
            f"Dataset procesado: {df_proc.shape[0]} filas originales, {df_filtrado.shape[0]} filas tras filtros, "
            f"{df_proc.shape[1]} columnas. "
            "Se aplicó: reemplazo de vacíos por NaN; sanitización de categóricas (trim de espacios y conversión a minúsculas); "
            "detección de nulos, duplicados y valores atípicos; columnas con solo dos valores (ej. sí/no) tratadas como booleanas y mostradas como 1/0. "
            "Revisa las secciones anteriores para los detalles."
        )

        st.markdown("**Filtros de tratamiento seleccionados**")
        if (
            imputation_method == "Sin imputación"
            and not delete_duplicates
            and not treat_outliers
        ):
            st.caption("No se aplicó ningún filtro adicional sobre el dataset.")
        else:
            if imputation_method != "Sin imputación":
                st.markdown(
                    f"- Imputación numérica: **{imputation_method}** "
                    f"(valores imputados: {valores_imputados})."
                )
            if delete_duplicates:
                st.markdown(
                    f"- Eliminación de duplicados: **{filas_eliminadas_duplicados}** filas marcadas como duplicadas."
                )
            if treat_outliers:
                st.markdown(
                    f"- Eliminación de valores atípicos: **{filas_eliminadas_atipicos}** filas marcadas con valores atípicos."
                )
            if filas_antes_filtros != filas_despues_filtros:
                st.markdown(
                    f"- Filas totales antes de filtros: **{filas_antes_filtros}**; "
                    f"después de filtros: **{filas_despues_filtros}**."
                )

else:
    st.info("👈 Selecciona una fuente de datos en la barra lateral y carga tu dataset para comenzar.")