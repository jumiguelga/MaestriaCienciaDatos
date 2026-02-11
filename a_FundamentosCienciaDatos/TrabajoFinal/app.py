# =============================================================================
# IMPORTS & CONFIG
# =============================================================================
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_MODEL = "llama-3.3-70b-versatile"


def _call_groq(client: "Groq", system_prompt: str, user_content: str) -> str:
    """Llama a la API de Groq y retorna el texto de la respuesta."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error al llamar a Groq: {e}"


def _try_convert_to_numeric(ser: pd.Series):
    """
    Intenta convertir una serie de strings a numérico.
    Soporta: "986,101", "8,874,909" (coma como miles); "1,234.56" (US); "1.234,56" (europeo).
    Retorna (serie_numerica, proporcion_exitosos) o (None, 0) si no convierte bien.
    """
    s = ser.astype(str).str.strip()
    empty = s.isin(["", "nan", "n/a", "none", "na"])
    non_empty = ~empty & s.notna()
    n_non_empty = non_empty.sum()
    if n_non_empty == 0:
        return None, 0.0

    def try_format(cleaner_fn):
        cleaned = s.copy()
        cleaned = cleaner_fn(cleaned)
        converted = pd.to_numeric(cleaned, errors="coerce")
        ok = converted.notna() & non_empty
        return converted, ok.sum() / n_non_empty

    # Formato 1: coma como miles (986,101 -> 986101; 1,234.56 -> 1234.56)
    conv1, rate1 = try_format(lambda x: x.str.replace(",", "", regex=False))
    # Formato 2: coma como decimal (1.234,56 -> 1234.56)
    conv2, rate2 = try_format(lambda x: x.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))

    if rate1 >= rate2 and rate1 >= 0.8:
        return conv1, rate1
    if rate2 >= 0.8:
        return conv2, rate2
    return None, 0.0


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
            etiqueta_issue: "#FFB903",  # secundario: filas con problema
            etiqueta_ok: "#004B85",     # principal: filas sin problema
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

# API Groq para Insights de IA (siempre visible)
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Groq IA (Insights)")
groq_api_key = st.sidebar.text_input(
    "API Key de Groq",
    type="password",
    placeholder="gsk_...",
    help="Obtén tu API key en console.groq.com. Se usa para generar insights automáticos en la pestaña Insights de IA.",
    key="groq_api_key",
)
if groq_api_key:
    st.sidebar.caption("API Key configurada.")

# =============================================================================
# MAIN: DISPLAY (df.info, df.head)
# =============================================================================

# Área principal
st.title("📊 Panel inteligente")

if df is not None:
    tab_ingesta, tab_visualizacion, tab_insights_ia = st.tabs(["Ingesta y Procesamiento de Datos (ETL)", "Visualización Dinámica (EDA)", "Insights de IA"])

    with tab_ingesta:
        # 1) Reemplazar vacíos por NaN
        df_proc = df.replace("", np.nan).replace(r"^\s*$", np.nan, regex=True)
        # Incluir object, string y category
        object_cols = [
            c for c in df_proc.columns
            if (
                df_proc[c].dtype == object
                or df_proc[c].dtype.name == "object"
                or pd.api.types.is_string_dtype(df_proc[c])
                or str(df_proc[c].dtype) == "category"
            )
        ]
        # 2a) Intentar convertir columnas objeto que parecen numéricas (ej. "986,101", "8,874,909")
        cols_converted_to_numeric = []
        for col in object_cols:
            s = df_proc[col]
            if s.dtype.name == "category":
                s = s.astype(object)
            num_ser, rate = _try_convert_to_numeric(s)
            if num_ser is not None:
                df_proc[col] = num_ser
                cols_converted_to_numeric.append(col)
        object_cols = [c for c in object_cols if c not in cols_converted_to_numeric]
        # 2b) Sanitizar el resto: trim y minúsculas (unifica "Fiber optic" y " Fiber optic ")
        for col in object_cols:
            s = df_proc[col]
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

        st.subheader("2. Sanitización y conversión numérica")
        st.caption(
            "Se detectaron strings con formato numérico (ej. 986,101, 8,874,909) y se convirtieron a numérico. "
            "El resto de columnas texto: trim y minúsculas."
        )
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Columnas convertidas a numérico (formato localizado)", len(cols_converted_to_numeric))
        with col_s2:
            st.metric("Columnas categóricas sanitizadas", len(object_cols))
        if cols_converted_to_numeric:
            st.caption(f"Convertidas a numérico: {', '.join(cols_converted_to_numeric)}.")
        if object_cols:
            st.caption(f"Columnas categóricas: {', '.join(object_cols)}.")

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

        # Column metadata para EDA: excluir index_column y columnas auxiliares
        exclude_cols = {"Nulos", "Duplicados", "Valores Atípicos"}
        if "index_column" in st.session_state:
            exclude_cols.add(st.session_state["index_column"])

        # Numeric: puras (excluir 0/1 binarias que son booleanas)
        numeric_cols_for_eda = [
            c for c in df_filtrado.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols and c not in numeric_bool_cols
        ]
        # Categorical
        categorical_cols_for_eda = [c for c in categorical_cols if c not in exclude_cols]
        # Boolean: inferidas + numéricas 0/1 + nativas
        all_bool_cols = list(infer_bool_cols) + list(numeric_bool_cols) + list(bool_cols)
        boolean_cols_for_eda = [c for c in all_bool_cols if c not in exclude_cols]
        # Datetime: detectar columnas de fecha nativas
        datetime_cols_for_eda = [
            c for c in df_filtrado.columns
            if c not in exclude_cols and pd.api.types.is_datetime64_any_dtype(df_filtrado[c])
        ]

        st.session_state["column_metadata"] = {
            "numeric": numeric_cols_for_eda,
            "categorical": categorical_cols_for_eda,
            "boolean": boolean_cols_for_eda,
            "datetime": datetime_cols_for_eda,
        }

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

    with tab_visualizacion:
        if "df_filtrado" not in st.session_state:
            st.info("Ejecuta primero el procesamiento en la pestaña **Ingesta y Procesamiento de Datos (ETL)** para habilitar la visualización.")
        else:
            df_viz = st.session_state["df_filtrado"].copy()
            meta = st.session_state.get("column_metadata", {})
            numeric_cols = meta.get("numeric", [])
            categorical_cols = meta.get("categorical", [])
            boolean_cols = meta.get("boolean", [])
            datetime_cols = meta.get("datetime", [])

            # ========== FILTROS GLOBALES ==========
            st.subheader("Filtros globales")
            with st.expander("Configurar filtros", expanded=False):
                # Rango de fechas
                if datetime_cols:
                    date_col = st.selectbox("Columna de fecha para filtrar", datetime_cols, key="eda_date_col")
                    if date_col and date_col in df_viz.columns:
                        col_ser = pd.to_datetime(df_viz[date_col], errors="coerce").dropna()
                        if len(col_ser) > 0:
                            min_d = col_ser.min()
                            max_d = col_ser.max()
                            min_date = min_d.date() if hasattr(min_d, "date") else min_d
                            max_date = max_d.date() if hasattr(max_d, "date") else max_d
                            d1, d2 = st.columns(2)
                            with d1:
                                date_start = st.date_input("Desde", value=min_date, min_value=min_date, max_value=max_date, key="eda_date_start")
                            with d2:
                                date_end = st.date_input("Hasta", value=max_date, min_value=min_date, max_value=max_date, key="eda_date_end")
                            if date_start and date_end and date_start <= date_end:
                                df_viz[date_col] = pd.to_datetime(df_viz[date_col], errors="coerce")
                                df_viz = df_viz[(df_viz[date_col].dt.date >= date_start) & (df_viz[date_col].dt.date <= date_end)]
                else:
                    st.caption("No se encontraron columnas de fecha para filtrar.")

                # Categorías
                if categorical_cols:
                    cat_filter_col = st.selectbox("Columna categórica a filtrar", ["(Ninguna)"] + categorical_cols, key="eda_cat_col")
                    if cat_filter_col != "(Ninguna)" and cat_filter_col in df_viz.columns:
                        vals = df_viz[cat_filter_col].dropna().unique().tolist()
                        selected_cats = st.multiselect("Valores a incluir (vacío = todos)", vals, default=vals, key="eda_cat_vals")
                        if selected_cats:
                            df_viz = df_viz[df_viz[cat_filter_col].isin(selected_cats)]

                # Sliders numéricos
                if numeric_cols:
                    num_filter_col = st.selectbox("Columna numérica a filtrar", ["(Ninguna)"] + numeric_cols, key="eda_num_col")
                    if num_filter_col != "(Ninguna)" and num_filter_col in df_viz.columns:
                        ser = df_viz[num_filter_col].dropna()
                        if len(ser) > 0:
                            lo, hi = float(ser.min()), float(ser.max())
                            rng = st.slider("Rango", lo, hi, (lo, hi), key="eda_num_range")
                            df_viz = df_viz[(df_viz[num_filter_col] >= rng[0]) & (df_viz[num_filter_col] <= rng[1])]

            if len(df_viz) == 0:
                st.warning("No hay datos tras aplicar los filtros. Ajusta los filtros o vuelve a cargar el dataset.")
            else:
                # ========== SUB-TABS: Univariado, Bivariado, Reporte ==========
                tab_uni, tab_bi, tab_rep = st.tabs(["Análisis Univariado", "Análisis Bivariado", "Reporte"])

                with tab_uni:
                    st.subheader("Análisis univariado")
                    # Numeric
                    if numeric_cols:
                        with st.expander("Columnas numéricas", expanded=True):
                            num_sel = st.selectbox("Seleccionar columna numérica", numeric_cols, key="uni_num")
                            nbins = st.slider("Bins del histograma", 5, 150, 30, key="uni_nbins")
                            c1, c2 = st.columns(2)
                            with c1:
                                fig_hist = px.histogram(df_viz, x=num_sel, nbins=nbins, title=f"Histograma: {num_sel}")
                                fig_hist.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                                st.plotly_chart(fig_hist, use_container_width=True)
                            with c2:
                                fig_box = px.box(df_viz, y=num_sel, title=f"Boxplot: {num_sel}")
                                fig_box.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                                st.plotly_chart(fig_box, use_container_width=True)
                    # Categorical
                    if categorical_cols:
                        with st.expander("Columnas categóricas", expanded=True):
                            cat_sel = st.selectbox("Seleccionar columna categórica", categorical_cols, key="uni_cat")
                            show_pct = st.checkbox("Mostrar porcentajes", value=False, key="uni_pct")
                            counts = df_viz[cat_sel].value_counts(dropna=False).reset_index()
                            counts.columns = ["categoria", "conteo"]
                            if show_pct:
                                total = counts["conteo"].sum()
                                counts["porcentaje"] = (counts["conteo"] / total * 100).round(1)
                                y_col = "porcentaje"
                            else:
                                y_col = "conteo"
                            fig_bar = px.bar(counts, x="categoria", y=y_col, title=f"Distribución: {cat_sel}")
                            fig_bar.update_layout(margin=dict(l=0, r=0, t=40, b=0), xaxis_tickangle=-45)
                            st.plotly_chart(fig_bar, use_container_width=True)
                    # Boolean
                    if boolean_cols:
                        with st.expander("Columnas booleanas", expanded=True):
                            bool_sel = st.selectbox("Seleccionar columna booleana", boolean_cols, key="uni_bool")
                            vc = df_viz[bool_sel].value_counts(dropna=False).reset_index()
                            vc.columns = ["valor", "conteo"]
                            fig_bool = px.bar(vc, x="valor", y="conteo", title=f"Distribución: {bool_sel}")
                            fig_bool.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                            st.plotly_chart(fig_bool, use_container_width=True)
                    if not numeric_cols and not categorical_cols and not boolean_cols:
                        st.caption("No hay columnas disponibles para análisis univariado.")

                with tab_bi:
                    st.subheader("Análisis bivariado")
                    # Correlaciones
                    if len(numeric_cols) >= 2:
                        with st.expander("Correlaciones (heatmap)", expanded=True):
                            corr_df = df_viz[numeric_cols].corr()
                            fig_corr = px.imshow(
                                corr_df,
                                text_auto=".2f",
                                color_continuous_scale="RdBu_r",
                                aspect="auto",
                                title="Matriz de correlación",
                            )
                            fig_corr.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=400)
                            st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.caption("Se requieren al menos 2 columnas numéricas para el heatmap de correlaciones.")
                    # Evolución temporal
                    if datetime_cols:
                        with st.expander("Evolución temporal", expanded=True):
                            dt_col = st.selectbox("Columna de fecha (eje X)", datetime_cols, key="bi_dt")
                            agg_col = st.selectbox("Columna a agregar (eje Y)", ["(Conteo)"] + numeric_cols, key="bi_agg")
                            agg_func = st.radio("Función de agregación", ["mean", "sum", "count"], horizontal=True, key="bi_agg_func")
                            df_temp = df_viz.copy()
                            df_temp[dt_col] = pd.to_datetime(df_temp[dt_col], errors="coerce")
                            df_temp = df_temp.dropna(subset=[dt_col])
                            if agg_col == "(Conteo)":
                                ts = df_temp.groupby(df_temp[dt_col].dt.to_period("D").dt.to_timestamp()).size().reset_index(name="conteo")
                                ts.columns = [dt_col, "conteo"]
                                fig_ts = px.area(ts, x=dt_col, y="conteo", title="Evolución temporal (conteo)")
                            else:
                                if agg_func == "count":
                                    ts = df_temp.groupby(df_temp[dt_col].dt.to_period("D").dt.to_timestamp())[agg_col].count().reset_index()
                                elif agg_func == "sum":
                                    ts = df_temp.groupby(df_temp[dt_col].dt.to_period("D").dt.to_timestamp())[agg_col].sum().reset_index()
                                else:
                                    ts = df_temp.groupby(df_temp[dt_col].dt.to_period("D").dt.to_timestamp())[agg_col].mean().reset_index()
                                fig_ts = px.line(ts, x=dt_col, y=agg_col, title=f"Evolución temporal ({agg_col} - {agg_func})")
                            fig_ts.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                            st.plotly_chart(fig_ts, use_container_width=True)
                    else:
                        st.caption("No se encontraron columnas de fecha para evolución temporal.")
                    # Cross-tabs opcionales
                    if numeric_cols and categorical_cols:
                        with st.expander("Numérico vs categórico (boxplot por categoría)", expanded=False):
                            bi_num = st.selectbox("Columna numérica (Y)", numeric_cols, key="bi_num")
                            bi_cat = st.selectbox("Columna categórica (X)", categorical_cols, key="bi_cat")
                            fig_cross = px.box(df_viz, x=bi_cat, y=bi_num, title=f"{bi_num} por {bi_cat}")
                            fig_cross.update_layout(margin=dict(l=0, r=0, t=40, b=0), xaxis_tickangle=-45)
                            st.plotly_chart(fig_cross, use_container_width=True)
                    if len(numeric_cols) >= 2:
                        with st.expander("Numérico vs numérico (scatter)", expanded=False):
                            sc_x = st.selectbox("Eje X", numeric_cols, key="sc_x")
                            sc_y_opts = [c for c in numeric_cols if c != sc_x] or numeric_cols
                            sc_y = st.selectbox("Eje Y", sc_y_opts, key="sc_y")
                            if sc_x and sc_y:
                                fig_sc = px.scatter(df_viz, x=sc_x, y=sc_y, title=f"{sc_x} vs {sc_y}")
                                fig_sc.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                                st.plotly_chart(fig_sc, use_container_width=True)

                with tab_rep:
                    st.subheader("Reporte")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Filas", len(df_viz))
                    with col2:
                        st.metric("Columnas", len(df_viz.columns))
                    with col3:
                        pct_miss = 100 * df_viz.isna().sum().sum() / (len(df_viz) * len(df_viz.columns)) if len(df_viz) > 0 else 0
                        st.metric("% valores faltantes", f"{pct_miss:.1f}%")
                    with col4:
                        st.metric("Columnas numéricas", len(numeric_cols))
                    st.subheader("Resumen estadístico (columnas numéricas)")
                    if numeric_cols:
                        st.dataframe(df_viz[numeric_cols].describe(), use_container_width=True)
                    else:
                        st.caption("No hay columnas numéricas.")
                    st.subheader("Moda por columna categórica/booleana")
                    cat_bool = categorical_cols + boolean_cols
                    if cat_bool:
                        modas = {c: df_viz[c].mode().iloc[0] if len(df_viz[c].mode()) > 0 else "—" for c in cat_bool}
                        st.dataframe(pd.DataFrame({"columna": list(modas.keys()), "moda": list(modas.values())}), use_container_width=True, hide_index=True)
                    csv = df_viz.to_csv(index=False).encode("utf-8")
                    st.download_button("Descargar dataset filtrado (CSV)", csv, "dataset_filtrado.csv", "text/csv", key="rep_dl")

    with tab_insights_ia:
        st.subheader("Insights de IA")
        groq_key = st.session_state.get("groq_api_key", "")
        if not GROQ_AVAILABLE:
            st.warning("La librería `groq` no está instalada. Ejecuta: `pip install groq`")
        elif not groq_key:
            st.session_state.pop("groq_client", None)
            st.info("Configura tu API Key de Groq en la barra lateral para habilitar los insights generados por IA.")
        else:
            try:
                groq_client = Groq(api_key=groq_key)
                st.session_state["groq_client"] = groq_client
            except Exception as e:
                st.session_state.pop("groq_client", None)
                st.error(f"Error al configurar Groq: {e}")
                groq_client = None

            if groq_client and "df_filtrado" in st.session_state:
                df_ins = st.session_state["df_filtrado"].copy()
                meta = st.session_state.get("column_metadata", {})
                numeric_cols = meta.get("numeric", [])
                categorical_cols = meta.get("categorical", [])
                boolean_cols = meta.get("boolean", [])

                # Layout: main (75%) | chat (25%)
                col_main, col_chat = st.columns([3, 1])
                with col_main:
                    # Resumen del dataset para Groq
                    n_rows, n_cols = len(df_ins), len(df_ins.columns)
                    missing = df_ins.isna().sum()
                    missing_pct = (missing / n_rows * 100).round(1)
                    cols_with_missing = [c for c in df_ins.columns if missing[c] > 0]
                    missing_summary = "\n".join(
                        [f"- {c}: {int(missing[c])} nulos ({missing_pct[c]}%)" for c in cols_with_missing[:20]]
                    ) if cols_with_missing else "Ninguna columna con valores faltantes."
                    desc = df_ins[numeric_cols].describe().round(2).to_string() if numeric_cols else "Sin columnas numéricas."

                    dataset_context = f"""
Dataset: {n_rows} filas, {n_cols} columnas.
Columnas numéricas: {numeric_cols[:15]}
Columnas categóricas: {categorical_cols[:15]}
Columnas booleanas: {boolean_cols[:10]}
Valores faltantes por columna:
{missing_summary}
Estadísticas descriptivas (numéricas):
{desc}
"""

                    # 1. Análisis general y recomendaciones
                    st.markdown("#### Análisis general del dataset")
                    if st.button("Generar análisis general", key="insights_general"):
                        with st.spinner("Analizando..."):
                            sys_prompt = """Eres un experto en ciencia de datos. Analiza el resumen del dataset y proporciona recomendaciones concisas en español.
Incluye: situaciones generales en las columnas, patrones que observes, y sugerencias de mejora. Responde en español, con viñetas claras."""
                            resp = _call_groq(groq_client, sys_prompt, dataset_context)
                            st.session_state["insights_general_text"] = resp
                    if "insights_general_text" in st.session_state:
                        st.markdown(st.session_state["insights_general_text"])

                    # 2. Heatmap de correlación + notas de Groq
                    if len(numeric_cols) >= 2:
                        st.markdown("#### Correlaciones")
                        corr_df = df_ins[numeric_cols].corr()
                        fig_corr = px.imshow(
                            corr_df,
                            text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            aspect="auto",
                            title="Matriz de correlación",
                        )
                        fig_corr.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=350)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        corr_context = f"Matriz de correlación (columnas): {list(corr_df.columns)}\nValores:\n{corr_df.round(2).to_string()}"
                        if st.button("Generar notas sobre correlaciones", key="insights_corr"):
                            with st.spinner("Analizando correlaciones..."):
                                sys_prompt = """Eres un experto en estadística. Analiza la matriz de correlación y escribe notas breves en español.
Identifica: correlaciones fuertes (>0.7 o <-0.7), posibles multicolinealidad, y qué pares de variables podrían ser útiles o redundantes. Responde en español con viñetas."""
                                resp = _call_groq(groq_client, sys_prompt, corr_context)
                                st.session_state["insights_corr_text"] = resp
                        if "insights_corr_text" in st.session_state:
                            with st.expander("Notas sobre correlaciones"):
                                st.markdown(st.session_state["insights_corr_text"])

                    # 3. Imputación
                    st.markdown("#### Valores faltantes e imputación")
                    if cols_with_missing:
                        imp_context = f"""Columnas con valores faltantes y porcentajes:
{missing_summary}
Columnas numéricas: {numeric_cols}
Columnas categóricas: {categorical_cols}"""
                        if st.button("Sugerir método de imputación", key="insights_imp"):
                            with st.spinner("Analizando valores faltantes..."):
                                sys_prompt = """Eres un experto en ciencia de datos. El usuario tiene un dataset con valores faltantes.
Sugiere qué método de imputación usar (media, mediana, moda, etc.) para cada tipo de columna, y por qué.
Responde en español con recomendaciones claras y viñetas."""
                                resp = _call_groq(groq_client, sys_prompt, imp_context)
                                st.session_state["insights_imp_text"] = resp
                        if "insights_imp_text" in st.session_state:
                            with st.expander("Recomendaciones de imputación"):
                                st.markdown(st.session_state["insights_imp_text"])
                    else:
                        st.caption("No se detectaron valores faltantes significativos.")

                with col_chat:
                    st.markdown("##### Chat")
                    if "insights_chat" not in st.session_state:
                        st.session_state["insights_chat"] = []
                    for msg in st.session_state["insights_chat"]:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                    if prompt := st.chat_input("Pregunta sobre el dataset..."):
                        st.session_state["insights_chat"].append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.chat_message("assistant"):
                            chat_context = f"Contexto del dataset: {dataset_context}\n\nConversación reciente: " + "\n".join(
                                [f"{m['role']}: {m['content']}" for m in st.session_state["insights_chat"][-6:]]
                            )
                            sys_prompt = "Eres un asistente experto en análisis de datos. Responde en español de forma concisa y útil."
                            resp = _call_groq(groq_client, sys_prompt, chat_context)
                            st.markdown(resp)
                        st.session_state["insights_chat"].append({"role": "assistant", "content": resp})
                        st.rerun()
            elif groq_client:
                st.info("Ejecuta primero el procesamiento en la pestaña **Ingesta y Procesamiento de Datos (ETL)** para generar insights.")

else:
    st.info("👈 Selecciona una fuente de datos en la barra lateral y carga tu dataset para comenzar.")