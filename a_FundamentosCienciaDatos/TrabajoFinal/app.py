# =============================================================================
# IMPORTS & CONFIG
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np

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
# MAIN: DISPLAY (df.info, df.head)
# =============================================================================

# Área principal
st.title("📊 Panel inteligente")

if df is not None:
    tab_ingesta, tab_visualizacion, tab_insights_ia = st.tabs(["Ingesta y Procesamiento de Datos (ETL)", "Visualización Dinámica (EDA)", "Insights de IA"])

    with tab_ingesta:
        # Resumen del dataset (en lugar de df.info)
        st.subheader("Información del dataset")
        n_rows, n_cols = len(df), len(df.columns)
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        bool_cols = set(df.select_dtypes(include=[bool]).columns)
        other_cols = [c for c in df.columns if c not in numeric_cols and c not in bool_cols]
        # Columnas con solo 2 valores únicos se consideran booleanas (ej. Yes/No -> 1/0)
        infer_bool_cols = [c for c in other_cols if df[c].dropna().nunique() == 2]
        categorical_cols = [c for c in other_cols if c not in infer_bool_cols]
        n_numeric = len(numeric_cols)
        n_bool = len(bool_cols) + len(infer_bool_cols)
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
        if infer_bool_cols:
            st.caption(f"Se consideran booleanas las columnas con solo dos valores (ej. Sí/No). Inferidas: {', '.join(infer_bool_cols)}.")

        # Para la tabla, convertir columnas tipo Yes/No a 1 y 0
        df_display = df.copy()
        for col in infer_bool_cols:
            unicos = df_display[col].dropna().unique().tolist()
            if len(unicos) == 2:
                mapping = {unicos[0]: 1, unicos[1]: 0}
                df_display[col] = df_display[col].map(mapping)
        st.subheader("Primeras 5 filas")
        st.dataframe(df_display.head(), use_container_width=True)

        # Reemplazar cadenas vacías y espacios en blanco por NaN
        df_proc = df.replace("", np.nan).replace(r"^\s*$", np.nan, regex=True)

        st.subheader("1. Reemplazo de valores vacíos por NaN")
        st.caption("Se convirtieron cadenas vacías y espacios en blanco a NaN.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas", df_proc.shape[0])
        with col2:
            st.metric("Columnas", df_proc.shape[1])
        with col3:
            st.metric("Valores NaN tras reemplazo", int(df_proc.isna().sum().sum()))
        
        # Columna Nulos: True si alguna columna es nula en esa fila
        df_proc["Nulos"] = df_proc.drop(columns=["Nulos"], errors="ignore").isna().any(axis=1)
        n_filas_nulas = df_proc["Nulos"].sum()

        st.subheader("2. Detección de nulos por fila")
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

        st.subheader("3. Detección de duplicados por fila")
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

        st.subheader("4. Detección de valores atípicos por fila")
        st.caption('Columna "Valores Atípicos": True si en esa fila algún valor numérico es atípico (método IQR).')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas con al menos un atípico", int(n_atipicos))
        with col2:
            st.metric("Filas sin atípicos", int((~df_proc["Valores Atípicos"]).sum()))
        with col3:
            pct_at = 100 * n_atipicos / len(df_proc) if len(df_proc) else 0
            st.metric("% filas con atípicos", f"{pct_at:.1f}%")

        st.divider()
        st.subheader("Resumen del procesamiento")
        st.success(f"Dataset procesado: {df_proc.shape[0]} filas, {df_proc.shape[1]} columnas. Revisa las secciones anteriores para los detalles.")

else:
    st.info("👈 Selecciona una fuente de datos en la barra lateral y carga tu dataset para comenzar.")