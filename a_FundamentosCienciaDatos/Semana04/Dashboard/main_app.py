import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="EDA - Energía Renovable", layout="wide")

st.title("Exploratory Data Analysis (EDA) - Proyectos de Energía Renovable")

st.markdown("""
Esta aplicación permite realizar un análisis exploratorio de datos (EDA) sobre proyectos de energía renovable.
Puedes subir tu propio archivo CSV o usar el archivo de ejemplo proporcionado.
""")

# Sidebar for file upload
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])

def load_data(file):
    try:
        df = pd.read_csv(file)
        # Convert Fecha_Entrada_Operacion to datetime
        if 'Fecha_Entrada_Operacion' in df.columns:
            df['Fecha_Entrada_Operacion'] = pd.to_datetime(df['Fecha_Entrada_Operacion'])
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Attempt to load the example data if no file is uploaded
# The absolute path in the issue was /Users/juangomez/Documents/Study/EAFIT/a.FundamentosCienciaDatos/datasets/energia_renovable.csv
# We'll try to look for it in the project structure if it was there, or just wait for upload.
# Based on project structure provided, it doesn't seem to be inside /Users/juangomez/Projects/MaestriaCienciaDatos
# except if it was recently added.
example_path = "a_FundamentosCienciaDatos/datasets/energia_renovable.csv"

df = None

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.success("Archivo cargado exitosamente.")
else:
    st.info("Por favor carga un archivo CSV (como 'energia_renovable.csv') para comenzar.")
    # For demonstration purposes, if the file existed in a known relative path:
    # try:
    #     df = load_data(example_path)
    #     st.write("Usando datos de ejemplo.")
    # except:
    #     pass

if df is not None:
    # Navigation
    tab1, tab2, tab3 = st.tabs(["Análisis Cuantitativo", "Análisis Cualitativo", "Gráficos Interactivos"])

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
            fig_hist = px.histogram(df, x=selected_num_col, marginal="box", title=f"Distribución de {selected_num_col}")
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
        
        # Dynamic Filters based on data
        if 'Tecnologia' in df.columns:
            tech_filter = st.sidebar.multiselect("Filtrar por Tecnología", options=df['Tecnologia'].unique(), default=df['Tecnologia'].unique())
            filtered_df = filtered_df[filtered_df['Tecnologia'].isin(tech_filter)]
            
        if 'Estado_Actual' in df.columns:
            status_filter = st.sidebar.multiselect("Filtrar por Estado", options=df['Estado_Actual'].unique(), default=df['Estado_Actual'].unique())
            filtered_df = filtered_df[filtered_df['Estado_Actual'].isin(status_filter)]

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
            
            df_time = filtered_df.sort_values(sel_date)
            fig_time = px.line(df_time, x=sel_date, y=sel_val, color='Tecnologia' if 'Tecnologia' in df.columns else None,
                               title=f"Evolución de {sel_val} en el tiempo")
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No se encontraron columnas de fecha para análisis temporal.")

        # Boxplot analysis
        st.subheader("Distribución por Categorías (Boxplot)")
        if not categorical_cols.empty and not numeric_df.empty:
            cat_box = st.selectbox("Categoría para el eje X", categorical_cols, key='box_cat')
            num_box = st.selectbox("Variable numérica para el eje Y", numeric_cols, key='box_num')
            fig_box = px.box(filtered_df, x=cat_box, y=num_box, color=cat_box,
                             title=f"Distribución de {num_box} por {cat_box}")
            st.plotly_chart(fig_box, use_container_width=True)
else:
    st.write("Por favor, sube un archivo CSV para visualizar el análisis.")
