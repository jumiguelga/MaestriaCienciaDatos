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

def add_log(action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append({"timestamp": timestamp, "action": action})

# --- FUNCIONES DE APOYO ---
def generate_pdf_report():
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, height - 50, "Reporte de Limpieza de Datos")
    
    p.setFont("Helvetica", 12)
    y = height - 80
    for log in st.session_state.logs:
        p.drawString(100, y, f"{log['timestamp']}: {log['action']}")
        y -= 20
        if y < 50:
            p.showPage()
            y = height - 50
            
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
    pdf_buf = generate_pdf_report()
    st.sidebar.download_button(label="Descargar Reporte PDF", data=pdf_buf, file_name="cleaning_log.pdf", mime="application/pdf")

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
    tab1, tab2, tab3, tab4 = st.tabs(["EDA General", "Salud Inventario", "Salud Transacciones", "Salud NPS"])

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
            
            detractores = fb_clean[fb_clean['Satisfaccion_NPS'] <= 6]
            pasivos = fb_clean[(fb_clean['Satisfaccion_NPS'] >= 7) & (fb_clean['Satisfaccion_NPS'] <= 8)]
            promotores = fb_clean[fb_clean['Satisfaccion_NPS'] >= 9]
            
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
            
            st.markdown("<h2 style='text-align: center;'>NET PROMOTER SCORE</h2>", unsafe_allow_html=True)
            
            # --- FILA 1: Distribución Detallada por Puntuación (0-10) ---
            fig_detail, ax_detail = plt.subplots(figsize=(12, 4))
            ax_detail.set_xlim(-0.5, 10.5)
            ax_detail.set_ylim(0, 4)
            ax_detail.axis('off')
            
            emojis = ["😡", "😠", "☹️", "🙁", "😐", "😕", "😶", "😌", "😊", "😀", "😁"]
            
            for i in range(11):
                # Determinar color
                color = color_det if i <= 6 else (color_pas if i <= 8 else color_pro)
                
                # Dibujar círculo de emoji
                ax_detail.text(i, 3, emojis[i], fontsize=30, ha='center', va='center', 
                               bbox=dict(facecolor=color, edgecolor='none', boxstyle='circle', pad=0.3))
                
                # Cantidad de respuestas
                ax_detail.text(i, 2.2, f"{nps_scores[i]}", fontsize=12, ha='center', fontweight='bold',
                               bbox=dict(facecolor='#333', edgecolor='none', boxstyle='round,pad=0.3'),
                               color='white')
                
                # Círculo con el número de puntuación
                ax_detail.text(i, 1.4, f"{i}", fontsize=14, ha='center', va='center', color='white', fontweight='bold',
                               bbox=dict(facecolor=color, edgecolor='none', boxstyle='circle', pad=0.4))

            # Etiquetas de categorías
            ax_detail.text(3, 0.7, "DETRACTORS", fontsize=10, ha='center', fontweight='bold', color='#555')
            ax_detail.text(7.5, 0.7, "PASSIVES", fontsize=10, ha='center', fontweight='bold', color='#555')
            ax_detail.text(9.5, 0.7, "PROMOTERS", fontsize=10, ha='center', fontweight='bold', color='#555')
            
            # Líneas de escala (barras de colores debajo)
            ax_detail.plot([-0.2, 6.2], [1, 1], color=color_det, lw=8, solid_capstyle='round', alpha=0.3)
            ax_detail.plot([6.8, 8.2], [1, 1], color=color_pas, lw=8, solid_capstyle='round', alpha=0.3)
            ax_detail.plot([8.8, 10.2], [1, 1], color=color_pro, lw=8, solid_capstyle='round', alpha=0.3)
            
            st.pyplot(fig_detail)
            
            # --- FILA 2: Fórmula y Métricas ---
            st.markdown(f"<h3 style='text-align: center;'>NPS = <span style='color:{color_pro}'>%PROMOTERS</span> - <span style='color:{color_det}'>%DETRACTORS</span></h3>", unsafe_allow_html=True)
            
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
                
                metric_box("detractors", count_det, pct_det, color_det)
                metric_box("passives", count_pas, pct_pas, color_pas)
                metric_box("promoters", count_pro, pct_pro, color_pro)
                
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

else:
    st.info("Por favor, carga los tres archivos CSV en el panel lateral para comenzar.")
    
    # Placeholder for structure info if user needs help
    with st.expander("Ver estructura esperada de archivos"):
        st.write("**Inventario:** SKU_ID, Stock_Actual, Costo_Unitario_USD, Categoria, Bodega_Origen, Lead_Time_Dias, Ultima_Revision")
        st.write("**Transacciones:** Transaccion_ID, SKU_ID, Fecha_Venta, Cantidad_Vendida, Precio_Venta_Final, Ciudad_Destino, Canal_Venta, Estado_Envio")
        st.write("**Feedback:** Feedback_ID, Transaccion_ID, Satisfaccion_NPS, Comentario_Texto, Recomienda_Marca")
