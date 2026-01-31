import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
from pandasai import SmartDataframe
from pandasai.llm import Groq

st.set_page_config(page_title="EDA Dinámico", layout="wide")

st.title("Exploratory Data Analysis (EDA) Dinámico")

st.markdown("""
Esta aplicación permite realizar un análisis exploratorio de datos (EDA) sobre cualquier conjunto de datos en formato CSV.
Carga tu archivo para comenzar a explorar las estadísticas, la calidad de los datos y generar gráficos interactivos.
""")

# Sidebar for file upload
st.sidebar.header("Configuración")

# Groq API Key
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Ingresa tu API Key de Groq para habilitar el Asistente AI")
if not groq_api_key:
    st.sidebar.warning("Ingresa una Groq API Key para usar el chat con datos.")

# Option to use example data
use_example = st.sidebar.checkbox("Usar datos de ejemplo si no hay archivo", value=True)

uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])

def load_data(file):
    try:
        df = pd.read_csv(file)
        # Dynamic date conversion: try to convert columns that look like dates
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Check if it's likely a date string (basic check)
                    if df[col].astype(str).str.contains('-').any() or df[col].astype(str).str.contains('/').any():
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Attempt to load data
df = None

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.sidebar.success("Archivo cargado exitosamente.")
elif use_example:
    # Try to find any available dataset for demonstration if no file is uploaded
    example_datasets = [
        "a_FundamentosCienciaDatos/datasets/energia_renovable.csv",
        "a_FundamentosCienciaDatos/datasets/agro_colombia.csv",
        "a_FundamentosCienciaDatos/datasets/monitoreo_ambiental.csv",
        "a_FundamentosCienciaDatos/Taller01/monitoreo_ambiental.csv",
        "monitoreo_ambiental.csv"
    ]
    for path in example_datasets:
        try:
            df = load_data(path)
            if df is not None:
                st.sidebar.info(f"Usando datos de ejemplo: `{path.split('/')[-1]}`")
                break
        except:
            continue

if df is not None:
    # Quantity of rows selector
    st.sidebar.subheader("Selección de Filas")
    max_rows = len(df)
    row_count = st.sidebar.slider("Cantidad de filas a analizar", 
                                  min_value=min(10, max_rows), 
                                  max_value=max_rows, 
                                  value=max_rows)
    
    # Filter by selected row count (taking the first N rows)
    df = df.head(row_count)
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["Análisis Cuantitativo", "Análisis Cualitativo", "Gráficos Interactivos", "Asistente AI"])

    with tab1:
        st.header("Análisis Cuantitativo")
        
        st.subheader("Resumen Estadístico")
        st.dataframe(df.describe())

        st.subheader("Matriz de Correlación")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.write("No hay suficientes datos numéricos para una matriz de correlación.")
        
        if not numeric_df.empty:
            st.subheader("Distribución de Variables Numéricas")
            selected_num_col = st.selectbox("Selecciona una variable para ver su distribución", numeric_df.columns)
            
            # Button to show/hide box plot overlay
            show_box = st.checkbox("Superponer gráfico de caja (Box Plot)", value=True)
            
            fig_hist = px.histogram(df, x=selected_num_col, 
                                   marginal="box" if show_box else None, 
                                   title=f"Distribución de {selected_num_col}")
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.header("Análisis Cualitativo")
        
        st.subheader("Información General")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.write(f"**Total de Registros:** {df.shape[0]}")
            st.write(f"**Total de Columnas:** {df.shape[1]}")
        with col_info2:
            st.write(f"**Valores Nulos Totales:** {df.isnull().sum().sum()}")

        st.subheader("Primeras Filas")
        st.dataframe(df.head())

        st.subheader("Tipos de Datos y Valores No Nulos")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        # Categorical Analysis
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        if not categorical_cols.empty:
            st.subheader("Resumen de Variables Categóricas")
            selected_cat_col = st.selectbox("Selecciona una variable categórica", categorical_cols)
            
            cat_counts = df[selected_cat_col].value_counts().reset_index()
            cat_counts.columns = [selected_cat_col, 'Cantidad']
            
            col_cat1, col_cat2 = st.columns([1, 2])
            with col_cat1:
                st.dataframe(cat_counts)
            with col_cat2:
                fig_cat = px.bar(cat_counts, x=selected_cat_col, y='Cantidad', 
                                 title=f"Frecuencia de {selected_cat_col}",
                                 color=selected_cat_col)
                st.plotly_chart(fig_cat, use_container_width=True)

    with tab3:
        st.header("Gráficos Interactivos")
        
        st.sidebar.subheader("Filtros de Datos")
        filtered_df = df.copy()
        
        # Dynamic Filters based on categorical data
        categorical_cols_for_filter = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # We only show filters for columns with a reasonable number of unique values (e.g. < 20)
        # to avoid cluttering the sidebar.
        for col in categorical_cols_for_filter:
            unique_vals = df[col].unique()
            if 1 < len(unique_vals) <= 25:
                filter_vals = st.sidebar.multiselect(f"Filtrar por {col}", options=unique_vals, default=unique_vals)
                filtered_df = filtered_df[filtered_df[col].isin(filter_vals)]

        # Dynamic Scatter Plot
        st.subheader("Explorador de Relaciones (Scatter Plot)")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col_sc1, col_sc2, col_sc3 = st.columns(3)
            with col_sc1:
                x_axis = st.selectbox("Eje X", numeric_cols, index=0)
            with col_sc2:
                y_axis = st.selectbox("Eje Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            with col_sc3:
                color_by = st.selectbox("Color por", ['Ninguno'] + list(df.columns))
            
            fig_dynamic = px.scatter(filtered_df, x=x_axis, y=y_axis, 
                                     color=color_by if color_by != 'Ninguno' else None,
                                     hover_data=df.columns,
                                     title=f"Relación entre {x_axis} y {y_axis}")
            st.plotly_chart(fig_dynamic, use_container_width=True)
        
        # Specific predefined but interactive charts
        st.subheader("Análisis de Series Temporales")
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols and not numeric_df.empty:
            sel_date = st.selectbox("Selecciona columna de fecha", date_cols)
            sel_val = st.selectbox("Selecciona valor a graficar", numeric_cols)
            
            # Determine grouping if there's a good categorical column
            color_time = 'Ninguno'
            potential_cats = [c for c in categorical_cols_for_filter if 1 < len(df[c].unique()) <= 10]
            if potential_cats:
                 color_time = st.selectbox("Agrupar por (Color)", ['Ninguno'] + potential_cats)

            df_time = filtered_df.sort_values(sel_date)
            fig_time = px.line(df_time, x=sel_date, y=sel_val, 
                               color=color_time if color_time != 'Ninguno' else None,
                               title=f"Evolución de {sel_val} en el tiempo")
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No se encontraron columnas de fecha para análisis temporal.")

        # Boxplot analysis
        st.subheader("Distribución por Categorías (Boxplot)")
        if not categorical_cols.empty and not numeric_df.empty:
            cat_box = st.selectbox("Categoría para el eje X", categorical_cols, key='box_cat')
            num_box = st.selectbox("Variable numérica para el eje Y", numeric_cols, key='box_num')
            
            show_points = st.checkbox("Mostrar todos los puntos", value=False)
            
            fig_box = px.box(filtered_df, x=cat_box, y=num_box, color=cat_box,
                             points="all" if show_points else "outliers",
                             title=f"Distribución de {num_box} por {cat_box}")
            st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        st.header("Asistente AI - Chat con tus Datos")
        
        if not groq_api_key:
            st.info("Por favor, ingresa tu Groq API Key en la barra lateral para habilitar el asistente.")
        else:
            try:
                # Initialize LLM
                llm = Groq(api_token=groq_api_key, model="llama3-70b-8192")
                smart_df = SmartDataframe(df, config={"llm": llm})

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if prompt := st.chat_input("¿Qué quieres saber sobre tus datos?"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Analizando..."):
                            response = smart_df.chat(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error al inicializar el asistente AI: {e}")
else:
    st.info("Carga un archivo CSV para comenzar el análisis.")
    if st.sidebar.button("Limpiar/Reiniciar App"):
        if "messages" in st.session_state:
            del st.session_state.messages
        st.rerun()
