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
    # Basic Information
    st.header("1. Información General del Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Primeras filas")
        st.dataframe(df.head())
        
    with col2:
        st.subheader("Resumen estadístico")
        st.dataframe(df.describe())

    st.subheader("Información de columnas y tipos de datos")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Visualizations
    st.header("2. Visualizaciones")

    # Technology distribution
    if 'Tecnologia' in df.columns:
        st.subheader("Distribución por Tecnología")
        fig_tech = px.pie(df, names='Tecnologia', title='Proporción de Proyectos por Tecnología', hole=0.3)
        st.plotly_chart(fig_tech, use_container_width=True)

    # Capacity by Technology
    if 'Tecnologia' in df.columns and 'Capacidad_Instalada_MW' in df.columns:
        st.subheader("Capacidad Instalada por Tecnología")
        fig_cap = px.box(df, x='Tecnologia', y='Capacidad_Instalada_MW', color='Tecnologia',
                         title='Distribución de Capacidad Instalada (MW) por Tecnología')
        st.plotly_chart(fig_cap, use_container_width=True)

    # Generation vs Capacity
    if 'Capacidad_Instalada_MW' in df.columns and 'Generacion_Diaria_MWh' in df.columns:
        st.subheader("Generación Diaria vs Capacidad Instalada")
        color_col = 'Tecnologia' if 'Tecnologia' in df.columns else None
        size_col = 'Inversion_Inicial_MUSD' if 'Inversion_Inicial_MUSD' in df.columns else None
        hover_col = 'ID_Proyecto' if 'ID_Proyecto' in df.columns else None
        
        fig_scatter = px.scatter(df, x='Capacidad_Instalada_MW', y='Generacion_Diaria_MWh', 
                                 color=color_col, size=size_col,
                                 hover_name=hover_col, title='Relación entre Capacidad y Generación')
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Status count
    if 'Estado_Actual' in df.columns:
        st.subheader("Estado Actual de los Proyectos")
        status_counts = df['Estado_Actual'].value_counts().reset_index()
        status_counts.columns = ['Estado', 'Cantidad']
        fig_status = px.bar(status_counts, x='Estado', y='Cantidad',
                            title='Cantidad de Proyectos por Estado Actual')
        st.plotly_chart(fig_status, use_container_width=True)

    # Efficiency analysis
    if 'Eficiencia_Planta_Pct' in df.columns:
        st.subheader("Análisis de Eficiencia")
        color_eff = 'Operador' if 'Operador' in df.columns else None
        fig_eff = px.histogram(df, x='Eficiencia_Planta_Pct', color=color_eff, barmode='overlay',
                               title='Distribución de Eficiencia de Planta')
        st.plotly_chart(fig_eff, use_container_width=True)

    # Time series
    if 'Fecha_Entrada_Operacion' in df.columns and 'Inversion_Inicial_MUSD' in df.columns:
        st.subheader("Evolución de la Inversión en el Tiempo")
        df_time = df.sort_values('Fecha_Entrada_Operacion')
        fig_time = px.line(df_time, x='Fecha_Entrada_Operacion', y='Inversion_Inicial_MUSD',
                           title='Inversión Inicial a lo largo del Tiempo')
        st.plotly_chart(fig_time, use_container_width=True)

    # Correlation Matrix
    st.subheader("Matriz de Correlación")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No hay suficientes datos numéricos para una matriz de correlación.")
else:
    st.write("Por favor, sube un archivo CSV para visualizar el análisis.")
