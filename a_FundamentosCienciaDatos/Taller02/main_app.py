import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functions_eda as feda
import dictionaries as dicts
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Configuración de la página
st.set_page_config(page_title="Data Quality & EDA Dashboard", layout="wide")

# --- ESTADO DE LA SESIÓN ---
if 'logs' not in st.session_state:
    st.session_state.logs = []

if 'fb_clean_adjusted' not in st.session_state:
    st.session_state.fb_clean_adjusted = None

if 'age_outliers_log' not in st.session_state:
    st.session_state.age_outliers_log = pd.DataFrame()

if 'user_comments' not in st.session_state:
    st.session_state.user_comments = ""

def add_log(action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append({"timestamp": timestamp, "action": action})

# --- FUNCIONES DE APOYO ---
def generate_pdf_report(user_comments=""):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, height - 50, "Reporte de Calidad y Limpieza de Datos")
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, height - 80, "Historial de Acciones:")
    
    p.setFont("Helvetica", 10)
    y = height - 100
    for log in st.session_state.logs:
        p.drawString(100, y, f"{log['timestamp']}: {log['action']}")
        y -= 15
        if y < 100:
            p.showPage()
            y = height - 50
    
    if user_comments:
        y -= 20
        if y < 150:
            p.showPage()
            y = height - 50
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y, "Comentarios del Analista:")
        y -= 20
        p.setFont("Helvetica", 10)
        # Manejo simple de multilínea
        textobject = p.beginText(100, y)
        for line in user_comments.split('\n'):
            textobject.textLine(line)
        p.drawText(textobject)
            
    p.save()
    buffer.seek(0)
    return buffer

def compute_health_score(df_raw, df_clean):
    if df_raw is None or df_raw.empty:
        return 0, 0, 0
    
    total_rows = len(df_raw)
    total_nulls = df_raw.isnull().sum().sum()
    rows_cleaned = total_rows - len(df_clean)
    
    # Simple health score formula: (1 - (nulls / total_elements)) * 100
    total_elements = df_raw.size
    null_ratio = total_nulls / total_elements if total_elements > 0 else 0
    health_score = (1 - null_ratio) * 100
    
    return health_score, total_nulls, rows_cleaned

# --- SIDEBAR: CARGA DE DATOS ---
st.sidebar.title("Configuración")
uploaded_inv = st.sidebar.file_uploader("Cargar Inventario (CSV)", type=["csv"])
uploaded_tx = st.sidebar.file_uploader("Cargar Transacciones (CSV)", type=["csv"])
uploaded_fb = st.sidebar.file_uploader("Cargar Feedback (CSV)", type=["csv"])

# Opciones de limpieza global
st.sidebar.subheader("Opciones de Limpieza")
exclude_outliers = st.sidebar.checkbox("Excluir Outliers", value=False)
exclude_nulls = st.sidebar.checkbox("Excluir Filas con Nulos", value=False)

if st.sidebar.button("Generar Log PDF"):
    pdf_buf = generate_pdf_report(st.session_state.user_comments)
    st.sidebar.download_button(label="Descargar Reporte PDF", data=pdf_buf, file_name="reporte_calidad.pdf", mime="application/pdf")

# --- LÓGICA PRINCIPAL ---
if uploaded_inv and uploaded_tx and uploaded_fb:
    # Cargar datos
    inv_raw = pd.read_csv(uploaded_inv)
    tx_raw = pd.read_csv(uploaded_tx)
    fb_raw = pd.read_csv(uploaded_fb)
    
    # Pre-detección de outliers para UI y lógica (opcional si se quiere ver antes de excluir)
    # Por ahora la exclusión global se encarga.

    # 1. Proceso de Limpieza Estándar
    inv_clean, inv_report = feda.sanitize_inventario(inv_raw)
    fb_clean = feda.limpiar_feedback_basico(fb_raw)
    
    # Aplicar ajustes persistentes de sesión si existen
    if st.session_state.fb_clean_adjusted is not None:
        fb_clean = st.session_state.fb_clean_adjusted
    
    # Logging inicial
    if 'initial_clean' not in st.session_state:
        add_log("Carga de archivos y limpieza inicial ejecutada.")
        st.session_state.initial_clean = True

    future_date_mode = st.sidebar.radio("Ventas Futuras", ["Mantener", "Corregir (Año -1)", "Excluir"])
    normalize_status = st.sidebar.checkbox("Normalizar Estado_Envio", value=False)
    
    # Aplicar limpieza estándar de transacciones
    tx_clean, tx_report = feda.sanitize_transacciones(tx_raw, normalize_status=normalize_status)
    
    # 2. Procesos de Limpieza Opcionales (Transacciones)
    st.sidebar.subheader("Limpieza Opcional Transacciones")
    
    impute_knn = st.sidebar.checkbox("Imputar Costo Envío (KNN)", value=False)
    if impute_knn:
        tx_clean = feda.imputar_costo_envio_knn(tx_clean)
        add_log("Imputación KNN de Costo Envío realizada.")
        
    exclude_neg_qty = st.sidebar.checkbox("Excluir Cantidades Negativas", value=False)
    if exclude_neg_qty:
        tx_clean = feda.excluir_ventas_cantidad_negativa(tx_clean)
        add_log("Cantidades negativas excluidas.")
        
    if future_date_mode == "Corregir (Año -1)":
        tx_clean = feda.corregir_o_excluir_ventas_futuras(tx_clean, modo="corregir")
        add_log("Fechas futuras corregidas.")
        # Actualizamos el reporte para reflejar la acción opcional
        future_mask = pd.to_datetime(tx_raw["Fecha_Venta"], format="%d/%m/%Y", errors="coerce") > pd.Timestamp(datetime.today().date())
        tx_report.loc[tx_report["Proceso"] == "Fechas futuras corregidas (año -1)", "Filas_afectadas"] = int(future_mask.sum())
    elif future_date_mode == "Excluir":
        tx_clean = feda.corregir_o_excluir_ventas_futuras(tx_clean, modo="excluir")
        add_log("Fechas futuras excluidas.")
        # Podríamos añadir una fila al reporte o reutilizar la existente
        future_mask = pd.to_datetime(tx_raw["Fecha_Venta"], format="%d/%m/%Y", errors="coerce") > pd.Timestamp(datetime.today().date())
        tx_report.loc[tx_report["Proceso"] == "Fechas futuras corregidas (año -1)", "Proceso"] = "Fechas futuras excluidas"
        tx_report.loc[tx_report["Proceso"] == "Fechas futuras excluidas", "Filas_afectadas"] = int(future_mask.sum())
    elif future_date_mode == "Mantener":
        # Si se elige mantener, ponemos a 0 en el reporte lo que hizo sanitize_transacciones por defecto
        tx_report.loc[tx_report["Proceso"] == "Fechas futuras corregidas (año -1)", "Filas_afectadas"] = 0
        
    include_ghost_skus = st.sidebar.checkbox("Incluir SKUs inexistentes en Inventario", value=True)
    tx_clean = feda.filtrar_skus_fantasma(tx_clean, inv_clean, incluir_fantasma=include_ghost_skus)
    if not include_ghost_skus:
        add_log("SKUs fantasma excluidos.")
    
    # Feedback
    exclude_fb_dupes = st.sidebar.checkbox("Excluir Feedback Duplicado", value=False)
    if exclude_fb_dupes:
        fb_clean = feda.excluir_feedback_duplicado(fb_clean)
        add_log("Feedback duplicado excluido.")

    # Exclusión global de Outliers/Nulos
    inv_clean, tx_clean, fb_clean = feda.aplicar_exclusion_global(
        inv_clean, tx_clean, fb_clean, 
        exclude_outliers=exclude_outliers, 
        exclude_nulls=exclude_nulls
    )
    if exclude_outliers:
        add_log("Outliers excluidos globalmente.")
    if exclude_nulls:
        add_log("Filas con nulos excluidas globalmente.")

    # JOIN Final para EDA
    joined_df = feda.build_join_dataset(tx_clean, inv_clean, fb_clean)
    joined_df = feda.feature_engineering(joined_df)

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["EDA General", "Salud Inventario", "Salud Transacciones", "Salud NPS", "Reporte (Dashboard)"])

    with tab1:
        st.header("Análisis Exploratorio de Datos (EDA) por Dataset")
        
        # --- EDA INVENTARIO ---
        st.subheader("1. EDA: Inventario")
        col_inv1, col_inv2 = st.columns(2)
        with col_inv1:
            st.write("**Estadísticas Cuantitativas**")
            st.write(inv_clean.describe())
        with col_inv2:
            st.write("**Estadísticas Cualitativas**")
            st.write(inv_clean.select_dtypes(include=['object', 'string']).describe())
        
        st.write("**Visualizaciones Inventario**")
        num_cols_inv = inv_clean.select_dtypes(include=[np.number]).columns
        if not num_cols_inv.empty:
            st.write("*Boxplots Variables Numéricas*")
            # Crear un grid de columnas para los boxplots (por ejemplo 3 columnas)
            cols_grid = st.columns(3)
            for i, col in enumerate(num_cols_inv):
                with cols_grid[i % 3]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.boxplot(x=inv_clean[col], ax=ax, color='skyblue')
                    ax.set_title(f"Distribución de {col}")
                    st.pyplot(fig)
        
        cat_cols_inv = inv_clean.select_dtypes(include=['object', 'string']).columns
        if 'Categoria' in cat_cols_inv:
            st.write("*Distribución por Categoría*")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(y=inv_clean['Categoria'], ax=ax, palette='viridis')
            st.pyplot(fig)

        st.divider()

        # --- EDA TRANSACCIONES ---
        st.subheader("2. EDA: Transacciones")
        col_tx1, col_tx2 = st.columns(2)
        with col_tx1:
            st.write("**Estadísticas Cuantitativas**")
            st.write(tx_clean.describe())
        with col_tx2:
            st.write("**Estadísticas Cualitativas**")
            st.write(tx_clean.select_dtypes(include=['object', 'string']).describe())

        st.write("**Visualizaciones Transacciones**")
        num_cols_tx = tx_clean.select_dtypes(include=[np.number]).columns
        if not num_cols_tx.empty:
            st.write("*Boxplots Variables Numéricas*")
            cols_grid_tx = st.columns(3)
            for i, col in enumerate(num_cols_tx):
                with cols_grid_tx[i % 3]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.boxplot(x=tx_clean[col], ax=ax, color='lightgreen')
                    ax.set_title(f"Distribución de {col}")
                    st.pyplot(fig)
        
        if 'Canal_Venta' in tx_clean.columns:
            st.write("*Distribución por Canal de Venta*")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x=tx_clean['Canal_Venta'], ax=ax, palette='magma')
            st.pyplot(fig)

        st.divider()

        # --- EDA FEEDBACK ---
        st.subheader("3. EDA: Feedback")
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            st.write("**Estadísticas Cuantitativas**")
            st.write(fb_clean.describe())
        with col_fb2:
            st.write("**Estadísticas Cualitativas**")
            st.write(fb_clean.select_dtypes(include=['object', 'string']).describe())

        st.write("**Visualizaciones Feedback**")
        num_cols_fb = fb_clean.select_dtypes(include=[np.number]).columns
        if not num_cols_fb.empty:
            st.write("*Boxplots Variables Numéricas*")
            cols_grid_fb = st.columns(3)
            for i, col in enumerate(num_cols_fb):
                with cols_grid_fb[i % 3]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.boxplot(x=fb_clean[col], ax=ax, color='salmon')
                    ax.set_title(f"Distribución de {col}")
                    st.pyplot(fig)
        
        if 'Satisfaccion_NPS_Grupo' in fb_clean.columns:
            st.write("*Distribución Grupos NPS*")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x=fb_clean['Satisfaccion_NPS_Grupo'], ax=ax, palette='rocket')
            plt.xticks(rotation=45)
            st.pyplot(fig)

    with tab2:
        st.header("Análisis de Salud: Inventario")
        h_score, nulls, cleaned = compute_health_score(inv_raw, inv_clean)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Health Score", f"{h_score:.2f}%")
        c2.metric("Nulos Totales (Raw)", nulls)
        c3.metric("Filas Removidas/Filtradas", cleaned)
        
        st.subheader("Reporte de Procesos de Limpieza")
        st.table(inv_report)
        
        st.subheader("Muestra de Datos Limpios")
        st.dataframe(inv_clean.head(20))

    with tab3:
        st.header("Análisis de Salud: Transacciones")
        
        # Gráfico de cantidades negativas
        neg_qty_df = tx_raw[tx_raw['Cantidad_Vendida'] < 0]
        if not neg_qty_df.empty:
            st.subheader("Análisis de Cantidades Negativas")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.write(f"Total de registros con cantidad negativa: {len(neg_qty_df)}")
                st.dataframe(neg_qty_df[['Transaccion_ID', 'SKU_ID', 'Cantidad_Vendida']].head(10))
            with col_v2:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.histplot(neg_qty_df['Cantidad_Vendida'], ax=ax, bins=20, color='red')
                ax.set_title("Distribución de Cantidades Negativas")
                st.pyplot(fig)
        else:
            st.success("No se encontraron cantidades negativas en los datos originales.")

        h_score, nulls, cleaned = compute_health_score(tx_raw, tx_clean)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Health Score", f"{h_score:.2f}%")
        c2.metric("Nulos Totales (Raw)", nulls)
        c3.metric("Filas Removidas/Filtradas", cleaned)
        
        # ghost SKU metrics
        st.subheader("Productos en Transacciones no existentes en Inventario")
        ghost_skus = tx_clean[tx_clean['flag_sku_fantasma'] == True]
        
        total_sales = (tx_clean['Cantidad_Vendida'] * tx_clean['Precio_Venta_Final']).sum()
        ghost_sales = (ghost_skus['Cantidad_Vendida'] * ghost_skus['Precio_Venta_Final']).sum()
        ghost_pct = (ghost_sales / total_sales * 100) if total_sales > 0 else 0
        
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("SKUs Únicos Faltantes", ghost_skus['SKU_ID'].nunique())
        gc2.metric("Transacciones Afectadas", len(ghost_skus))
        gc3.metric("Ventas Perdidas/Fantasma (USD)", f"${ghost_sales:,.2f}")
        st.write(f"Esto representa el **{ghost_pct:.2f}%** del total de ventas procesadas.")

        if not ghost_skus.empty:
            st.dataframe(ghost_skus[['SKU_ID', 'Transaccion_ID', 'Cantidad_Vendida', 'Precio_Venta_Final']].head(10))

        st.subheader("Reporte de Procesos de Limpieza")
        st.table(tx_report)

    with tab4:
        st.header("Análisis de Salud: NPS (Feedback)")
        h_score, nulls, cleaned = compute_health_score(fb_raw, fb_clean)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Health Score", f"{h_score:.2f}%")
        c2.metric("Nulos Totales (Raw)", nulls)
        c3.metric("Filas Removidas/Filtradas", cleaned)

        if 'Satisfaccion_NPS' in fb_clean.columns:
            # --- NUEVA VISUALIZACIÓN NPS PROFESIONAL ---
            st.markdown("---")
            
            # 1. Preparación de Datos
            nps_scores = fb_clean['Satisfaccion_NPS'].value_counts().reindex(range(11), fill_value=0)
            total_respuestas = len(fb_clean)
            
            detractores = fb_clean[fb_clean['Satisfaccion_NPS'] <= 0]
            pasivos = fb_clean[(fb_clean['Satisfaccion_NPS'] >= 0.1) & (fb_clean['Satisfaccion_NPS'] <= 50)]
            promotores = fb_clean[fb_clean['Satisfaccion_NPS'] >= 50.1]
            
            count_det = len(detractores)
            count_pas = len(pasivos)
            count_pro = len(promotores)
            
            pct_det = (count_det / total_respuestas * 100) if total_respuestas > 0 else 0
            pct_pas = (count_pas / total_respuestas * 100) if total_respuestas > 0 else 0
            pct_pro = (count_pro / total_respuestas * 100) if total_respuestas > 0 else 0
            
            nps_score = pct_pro - pct_det
            
            # Colores corporativos
            color_det = "#E91E63"
            color_pas = "#FFC107"
            color_pro = "#4CAF50"
            
            # --- FILA 2: Fórmula y Métricas ---
            st.markdown(f"<h3 style='text-align: center;'>NPS = <span style='color:{color_pro}'>%PROMOTORES</span> - "
                        f"<span style='color:{color_det}'>%DETRACTORES</span></h3>", unsafe_allow_html=True)
            
            col_met1, col_met2, col_met3 = st.columns([1.5, 3, 1])
            
            with col_met1:
                # Donut Chart
                fig_donut, ax_donut = plt.subplots(figsize=(4, 4))
                sizes = [max(0.1, pct_pro), max(0.1, pct_pas), max(0.1, pct_det)]
                
                ax_donut.pie(sizes, colors=[color_pro, color_pas, color_det], startangle=90, counterclock=False, 
                             wedgeprops={'width': 0.3, 'edgecolor': 'w', 'linewidth': 2})
                
                ax_donut.text(0, 0, f"{int(nps_score)}", fontsize=40, ha='center', va='center', fontweight='bold')
                st.pyplot(fig_donut)
                
            with col_met2:
                st.write("") # Espaciador
                # Estilo de las cajas de métricas
                def metric_box(label, count, pct, color):
                    st.markdown(f"""
                        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                            <div style='background-color:{color}; color:white; padding:10px 20px; border-radius:20px; width:150px; text-align:center; font-weight:bold; margin-right:10px;'>{label}</div>
                            <div style='background-color:{color}; color:white; padding:10px; border-radius:50%; width:40px; height:40px; display:flex; align-items:center; justify-content:center; font-weight:bold; margin-right:10px;'>{count}</div>
                            <div style='background-color:#F0F2F6; padding:10px 20px; border-radius:20px; width:100px; text-align:center; font-weight:bold;'>{pct:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                metric_box("detractores", count_det, pct_det, color_det)
                metric_box("pasivos", count_pas, pct_pas, color_pas)
                metric_box("promotores", count_pro, pct_pro, color_pro)
                
            with col_met3:
                st.markdown(f"""
                    <div style='text-align: center; margin-top: 50px;'>
                        <div style='background-color:#333; color:white; padding:5px 15px; border-radius:15px; display:inline-block; font-weight:bold;'>total</div>
                        <div style='background-color:#AAA; color:white; width:80px; height:80px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:24px; font-weight:bold; margin: 10px auto;'>{total_respuestas}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
        else:
            st.warning("No se encontró información de NPS procesada.")

        if 'Edad_Cliente' in fb_clean.columns:
            st.subheader("Análisis de Outliers en Edad")
            
            # Función para ajustar outliers de edad
            if st.button("Ajustar Outliers de Edad a la Mediana"):
                median_age = fb_clean[fb_clean['Edad_Cliente'] <= 100]['Edad_Cliente'].median()
                outliers_mask = fb_clean['Edad_Cliente'] > 100
                num_outliers = outliers_mask.sum()
                if num_outliers > 0:
                    # Guardar los datos outliers antes de imputar para el log
                    current_outliers = fb_clean[outliers_mask].copy()
                    st.session_state.age_outliers_log = pd.concat([st.session_state.age_outliers_log, current_outliers]).drop_duplicates()
                    
                    fb_clean.loc[outliers_mask, 'Edad_Cliente'] = median_age
                    st.session_state.fb_clean_adjusted = fb_clean.copy()
                    add_log(f"Se ajustaron {num_outliers} outliers de edad a la mediana ({median_age}).")
                    st.success(f"Se han ajustado {num_outliers} registros.")
                    # Forzamos recarga para ver cambios en gráficas
                    st.rerun()
                else:
                    st.info("No se encontraron outliers (> 100 años) para ajustar.")

            # Mostrar log de cambios de edad si existe
            if not st.session_state.age_outliers_log.empty:
                with st.expander("Ver log de outliers de edad procesados"):
                    st.write("Registros que superaban los 100 años y fueron ajustados:")
                    st.dataframe(st.session_state.age_outliers_log)

            age_outliers = feda.outlier_flag_iqr(fb_clean, 'Edad_Cliente')
            outlier_df = fb_clean[age_outliers]
            
            if not outlier_df.empty:
                st.warning(f"Se detectaron {len(outlier_df)} registros con edades fuera de rango.")
                col_age1, col_age2 = st.columns(2)
                with col_age1:
                    st.write("Muestra de Outliers:")
                    st.dataframe(outlier_df[['Feedback_ID', 'Edad_Cliente', 'Satisfaccion_NPS']].head(10))
                with col_age2:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.boxplot(x=fb_clean['Edad_Cliente'], ax=ax, color='orange')
                    ax.set_title("Boxplot de Edad (Feedback)")
                    st.pyplot(fig)
            else:
                st.success("No se detectaron outliers significativos en la edad de los encuestados.")
        else:
            # Si no existe la columna 'Edad', informamos (podría ser opcional en el CSV)
            st.info("La columna 'Edad' no está presente en el dataset de Feedback para el análisis de outliers.")

    with tab5:
        st.header("📊 Reporte de Calidad y Análisis de Negocio")
        
        # SECCIÓN 1: Métricas de Calidad de Datos
        st.subheader("1️⃣ Métricas de Calidad de Datos")
        
        def get_top_nulls(df):
            nulls = df.isnull().sum()
            return nulls[nulls > 0].sort_values(ascending=False).head(5)

        def count_outliers_multicol(df):
            # Usamos la lógica de feda para consistencia
            df_out = feda.detectar_outliers_multicolumna(df)
            total = len(df_out)
            count = df_out["Es_Outlier"].sum()
            pct = (count / total * 100) if total > 0 else 0
            return count, pct

        # Métricas Inventario
        inv_out_count, inv_out_pct = count_outliers_multicol(inv_raw)
        inv_metrics = {
            "Dataset": "Inventario",
            "Registros Totales (Raw)": len(inv_raw),
            "Registros Finales (Clean)": len(inv_clean),
            "% Nulidad General": (inv_raw.isnull().sum().sum() / inv_raw.size * 100),
            "Top 5 Nulos": str(get_top_nulls(inv_raw).to_dict()),
            "Duplicados": inv_raw.duplicated().sum(),
            "Outliers": f"{inv_out_count} ({inv_out_pct:.1f}%)"
        }
        
        # Métricas Transacciones
        tx_out_count, tx_out_pct = count_outliers_multicol(tx_raw)
        tx_metrics = {
            "Dataset": "Transacciones",
            "Registros Totales (Raw)": len(tx_raw),
            "Registros Finales (Clean)": len(tx_clean),
            "% Nulidad General": (tx_raw.isnull().sum().sum() / tx_raw.size * 100),
            "Top 5 Nulos": str(get_top_nulls(tx_raw).to_dict()),
            "Duplicados": tx_raw.duplicated().sum(),
            "Outliers": f"{tx_out_count} ({tx_out_pct:.1f}%)"
        }
        
        # Métricas Feedback
        fb_out_count, fb_out_pct = count_outliers_multicol(fb_raw)
        fb_metrics = {
            "Dataset": "Feedback",
            "Registros Totales (Raw)": len(fb_raw),
            "Registros Finales (Clean)": len(fb_clean),
            "% Nulidad General": (fb_raw.isnull().sum().sum() / fb_raw.size * 100),
            "Top 5 Nulos": str(get_top_nulls(fb_raw).to_dict()),
            "Duplicados": fb_raw.duplicated().sum(),
            "Outliers": f"{fb_out_count} ({fb_out_pct:.1f}%)"
        }
        
        df_quality = pd.DataFrame([inv_metrics, tx_metrics, fb_metrics])
        st.table(df_quality)
        
        # Visualización: Barras Apiladas
        st.write("**Registros Originales vs Limpios vs Excluidos**")
        data_viz = {
            'Dataset': ['Inventario', 'Transacciones', 'Feedback'],
            'Limpios': [len(inv_clean), len(tx_clean), len(fb_clean)],
            'Excluidos': [len(inv_raw)-len(inv_clean), len(tx_raw)-len(tx_clean), len(fb_raw)-len(fb_clean)]
        }
        df_viz = pd.DataFrame(data_viz)
        
        fig_qual, ax_qual = plt.subplots(figsize=(10, 5))
        df_viz.set_index('Dataset').plot(kind='bar', stacked=True, color=['#4CAF50', '#E91E63'], ax=ax_qual)
        ax_qual.set_title("Estado de los Registros por Dataset")
        ax_qual.set_ylabel("Cantidad de Registros")
        st.pyplot(fig_qual)
        
        st.divider()
        
        # SECCIÓN 2: Decisiones Éticas de Limpieza
        st.subheader("2️⃣ Decisiones Éticas de Limpieza")
        col_log1, col_log2 = st.columns([2, 1])
        
        with col_log1:
            st.write("**Log de Acciones de Limpieza**")
            if st.session_state.logs:
                log_df = pd.DataFrame(st.session_state.logs)
                st.dataframe(log_df, use_container_width=True)
            else:
                st.info("No hay acciones registradas aún.")
        
        with col_log2:
            st.write("**Comentarios del Analista**")
            st.session_state.user_comments = st.text_area("Justificaciones éticas y observaciones:", 
                                                         value=st.session_state.user_comments,
                                                         placeholder="Escriba aquí sus comentarios...",
                                                         height=200)

        st.write("**Resumen de Decisiones de Imputación/Limpieza**")
        imputacion_data = [
            {"Variable": "Stock_Actual", "Acción Tomada": "Conversión/Imputación", "Método": "Absoluto y Fillna(0)", "Justificación": "Stock no puede ser negativo; nulos se asumen sin existencias."},
            {"Variable": "Costo_Envio", "Acción Tomada": "Imputación (Opcional)", "Método": "KNN Imputer", "Justificación": "Recuperar datos perdidos basados en cercanía de otras variables logísticas."},
            {"Variable": "Edad_Cliente", "Acción Tomada": "Imputación", "Método": "Mediana", "Justificación": "Valores > 100 son outliers biológicos improbables; se ajustan a la tendencia central."},
            {"Variable": "Fechas_Venta", "Acción Tomada": "Corrección/Exclusión", "Método": "Año -1 / Drop", "Justificación": "Datos futuros son errores de entrada; se corrigen si es posible o se eliminan para no sesgar el histórico."}
        ]
        st.table(pd.DataFrame(imputacion_data))
        
        # Distribuciones antes/después (Ejemplo con Edad si fue ajustada)
        if st.session_state.fb_clean_adjusted is not None:
            st.write("**Impacto de la Imputación: Edad_Cliente**")
            fig_age, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(fb_raw['Edad_Cliente'].dropna(), ax=ax1, color='gray', kde=True)
            ax1.set_title("Antes (Raw)")
            sns.histplot(fb_clean['Edad_Cliente'], ax=ax2, color='salmon', kde=True)
            ax2.set_title("Después (Limpios/Ajustados)")
            st.pyplot(fig_age)

        st.divider()
        
        # SECCIÓN 3: Análisis de SKUs Fantasma
        st.subheader("3️⃣ Análisis de SKUs Fantasma")
        ghost_skus_all = tx_clean[tx_clean['flag_sku_fantasma'] == True]
        
        if not ghost_skus_all.empty:
            total_ghost_unique = ghost_skus_all['SKU_ID'].nunique()
            total_tx_ghost = len(ghost_skus_all)
            pct_tx_ghost = (total_tx_ghost / len(tx_clean) * 100)
            
            ghost_sales_val = (ghost_skus_all['Cantidad_Vendida'] * ghost_skus_all['Precio_Venta_Final']).sum()
            total_sales_val = (tx_clean['Cantidad_Vendida'] * tx_clean['Precio_Venta_Final']).sum()
            pct_sales_ghost = (ghost_sales_val / total_sales_val * 100) if total_sales_val > 0 else 0
            
            gs_c1, gs_c2, gs_c3 = st.columns(3)
            gs_c1.metric("SKUs Únicos Fantasma", total_ghost_unique)
            gs_c2.metric("Transacciones Afectadas", f"{total_tx_ghost} ({pct_tx_ghost:.2f}%)")
            gs_c3.metric("Ventas Totales Fantasma", f"${ghost_sales_val:,.2f} ({pct_sales_ghost:.2f}%)")
            
            # Gráfico de barras horizontal TOP 10
            st.write("**Top 10 SKUs Fantasma por Frecuencia**")
            top_ghost = ghost_skus_all['SKU_ID'].value_counts().head(10).reset_index()
            top_ghost.columns = ['SKU_ID', 'Frecuencia']
            
            fig_ghost, ax_ghost = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Frecuencia', y='SKU_ID', data=top_ghost, palette='Oranges_r', ax=ax_ghost)
            ax_ghost.set_title("SKUs Fantasma más frecuentes")
            st.pyplot(fig_ghost)
            
            with st.expander("Ver Tabla Detallada de SKUs Fantasma"):
                # Agrupación para tabla detallada
                ghost_details = ghost_skus_all.groupby('SKU_ID').agg({
                    'Transaccion_ID': 'count',
                    'Cantidad_Vendida': 'sum',
                    'Precio_Venta_Final': 'sum' # Esto es suma de precios, quizás mejor Ingreso
                }).reset_index()
                ghost_details.columns = ['SKU_ID', 'Transacciones', 'Cant_Total', 'Ingreso_Total']
                ghost_details = ghost_details.sort_values('Ingreso_Total', ascending=False)
                st.dataframe(ghost_details, use_container_width=True)
        else:
            st.success("No se detectaron SKUs fantasma en el dataset actual.")

        st.divider()
        
        # SECCIÓN 4: Margen Negativo (Fuga de Capital)
        st.subheader("4️⃣ Fuga de Capital: SKUs con Margen Negativo")
        
        margen_negativo = joined_df[joined_df['Margen_Neto_aprox'] < 0].copy()
        
        if not margen_negativo.empty:
            loss_unique_skus = margen_negativo['SKU_ID'].nunique()
            loss_tx_count = len(margen_negativo)
            total_loss = margen_negativo['Margen_Neto_aprox'].sum()
            
            total_revenue = joined_df['Ingreso'].sum()
            pct_loss_tx = (loss_tx_count / len(joined_df) * 100) if len(joined_df) > 0 else 0
            
            mn_c1, mn_c2, mn_c3 = st.columns(3)
            mn_c1.metric("SKUs con Margen Negativo", loss_unique_skus)
            mn_c2.metric("Transacciones con Pérdida", f"{loss_tx_count} ({pct_loss_tx:.2f}%)")
            mn_c3.metric("Pérdida Total Acumulada", f"${total_loss:,.2f}", delta_color="inverse")
            
            # Gráfico Top 10 Pérdidas
            st.write("**Top 10 SKUs con Mayor Pérdida Acumulada**")
            perdidas_por_sku = margen_negativo.groupby('SKU_ID').agg({
                'Margen_Neto_aprox': 'sum',
                'Transaccion_ID': 'count',
                'Canal_Venta': lambda x: x.mode()[0] if not x.empty else 'Desconocido'
            }).reset_index()
            perdidas_por_sku.columns = ['SKU_ID', 'Perdida_Total', 'Num_Transacciones', 'Canal_Principal']
            top_perdidas = perdidas_por_sku.sort_values('Perdida_Total').head(10) # Más negativo
            
            fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
            sns.barplot(x=top_perdidas['Perdida_Total'].abs(), y=top_perdidas['SKU_ID'], palette='Reds_r', ax=ax_loss)
            ax_loss.set_title("Top 10 SKUs con mayor Fuga de Capital")
            ax_loss.set_xlabel("Pérdida Acumulada (Valor Absoluto)")
            st.pyplot(fig_loss)
            
            st.write("**Distribución de Margen Negativo por Canal de Venta**")
            canal_loss = margen_negativo.groupby('Canal_Venta')['Margen_Neto_aprox'].sum().reset_index()
            st.table(canal_loss.sort_values('Margen_Neto_aprox'))
        else:
            st.success("No se detectaron transacciones con margen neto negativo.")

        st.divider()
        
        # SECCIÓN 5: Crisis Logística y Correlación NPS
        st.subheader("5️⃣ Crisis Logística: Correlación Tiempo de Entrega vs NPS")
        
        # Filtrar datos válidos
        analisis_log = joined_df[
            (joined_df['Tiempo_Entrega_Real'].notna()) & 
            (joined_df['Satisfaccion_NPS'].notna()) &
            (joined_df['Bodega_Origen'].notna()) &
            (joined_df['Ciudad_Destino'].notna())
        ].copy()
        
        if not analisis_log.empty:
            # Correlación por Bodega-Ciudad
            def get_corr(x):
                if len(x) > 5:
                    return x[['Tiempo_Entrega_Real', 'Satisfaccion_NPS']].corr().iloc[0, 1]
                return np.nan

            res_corr = analisis_log.groupby(['Bodega_Origen', 'Ciudad_Destino']).apply(get_corr).reset_index(name='Correlacion')
            
            metricas_log = analisis_log.groupby(['Bodega_Origen', 'Ciudad_Destino']).agg({
                'Tiempo_Entrega_Real': 'mean',
                'Satisfaccion_NPS': 'mean',
                'Transaccion_ID': 'count'
            }).reset_index()
            
            resultado_log = res_corr.merge(metricas_log, on=['Bodega_Origen', 'Ciudad_Destino'])
            resultado_log = resultado_log.sort_values('Correlacion')
            
            # Heatmap
            st.write("**Heatmap de Correlación: Tiempo Entrega vs NPS**")
            pivot_corr = resultado_log.pivot(index="Ciudad_Destino", columns="Bodega_Origen", values="Correlacion")
            
            fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot_corr, annot=True, cmap='RdYlGn', center=0, ax=ax_heat)
            ax_heat.set_title("Correlación de Pearson (Negativo = Rojo)")
            st.pyplot(fig_heat)
            
            # Identificación zona crítica
            criticos = resultado_log[(resultado_log['Correlacion'] < -0.5) & (resultado_log['Transaccion_ID'] > 2)]
            if not criticos.empty:
                peor = criticos.iloc[0]
                st.error(f"🚨 **ZONA CRÍTICA DETECTADA:** La ruta desde **{peor['Bodega_Origen']}** hacia **{peor['Ciudad_Destino']}** tiene una correlación de **{peor['Correlacion']:.2f}**. Requiere cambio inmediato de operador logístico.")
            
            with st.expander("Ver Tabla Completa de Correlación Logística"):
                st.dataframe(resultado_log, use_container_width=True)
        else:
            st.info("No hay suficientes datos cruzados (Entrega + NPS) para realizar el análisis de correlación.")

else:
    st.info("Por favor, carga los tres archivos CSV en el panel lateral para comenzar.")
    
    # Placeholder for structure info if user needs help
    with st.expander("Ver estructura esperada de archivos"):
        st.write("**Inventario:** SKU_ID, Stock_Actual, Costo_Unitario_USD, Categoria, Bodega_Origen, Lead_Time_Dias, Ultima_Revision")
        st.write("**Transacciones:** Transaccion_ID, SKU_ID, Fecha_Venta, Cantidad_Vendida, Precio_Venta_Final, Ciudad_Destino, Canal_Venta, Estado_Envio")
        st.write("**Feedback:** Feedback_ID, Transaccion_ID, Satisfaccion_NPS, Comentario_Texto, Recomienda_Marca")
