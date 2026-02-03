#Main

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import functions_eda as feda
import dictionaries as dicts
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Configuración de la página
st.set_page_config(page_title="Data Quality & EDA Dashboard", layout="wide")

# Groq for Agent (optional)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# --- ESTADO DE LA SESIÓN ---
if 'logs' not in st.session_state:
    st.session_state.logs = []

if 'fb_clean_adjusted' not in st.session_state:
    st.session_state.fb_clean_adjusted = None

if 'age_outliers_log' not in st.session_state:
    st.session_state.age_outliers_log = pd.DataFrame()

if 'user_comments' not in st.session_state:
    st.session_state.user_comments = ""

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

if 'insight_ia_result' not in st.session_state:
    st.session_state.insight_ia_result = None
if 'insight_ia_timestamp' not in st.session_state:
    st.session_state.insight_ia_timestamp = None

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

# API Key for AI Agent (Groq) - from secrets or sidebar override
def _get_groq_api_key():
    """API Key from Streamlit secrets first, then session state override."""
    try:
        from_secrets = st.secrets.get("GROQ_API_KEY", "")
    except (FileNotFoundError, KeyError, AttributeError):
        from_secrets = ""
    return (st.session_state.api_key or from_secrets or "").strip()

st.sidebar.subheader("🤖 Agente de Análisis (Opcional)")
with st.sidebar.expander("Configurar API Key (Groq)"):
    _secret_help = (
        "Configúrelo en .streamlit/secrets.toml:\n"
        "GROQ_API_KEY = \"gsk_...\"\n\n"
        "O ingréselo aquí (se mantiene en sesión):"
    )
    api_key_input = st.text_input(
        "API Key (Groq)",
        value=st.session_state.api_key,
        type="password",
        placeholder="gsk_...",
        help=_secret_help,
        key="groq_api_key_input"
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
    groq_key = _get_groq_api_key()
    if groq_key:
        st.success("✅ API Key configurada (Groq)")
    else:
        st.info("Configure GROQ_API_KEY en secrets o aquí para el chat con el agente.")

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

    # --- TABS: Auditoría, Operaciones, Cliente, Insights de IA ---
    tab_auditoria, tab_operaciones, tab_cliente, tab_insights_ia = st.tabs([
        "Auditoría", "Operaciones", "Cliente", "Insights de IA"
    ])

    with tab_auditoria:
        st.header("Auditoría")
        st.subheader("Resumen EDA por Dataset")

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

        st.divider()
        st.subheader("Salud: Inventario")
        h_score, nulls, cleaned = compute_health_score(inv_raw, inv_clean)

        c1, c2, c3 = st.columns(3)
        c1.metric("Health Score", f"{h_score:.2f}%")
        c2.metric("Nulos Totales (Raw)", nulls)
        c3.metric("Filas Removidas/Filtradas", cleaned)

        st.subheader("Reporte de Procesos de Limpieza")
        st.table(inv_report)

        st.subheader("Muestra de Datos Limpios")
        st.dataframe(inv_clean.head(20))

        st.divider()
        st.subheader("Salud: Transacciones")

        # Gráfico de cantidades negativas
        neg_qty_df = pd.DataFrame()
        if 'Cantidad_Vendida' in tx_raw.columns:
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

        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("SKUs Únicos Faltantes", ghost_skus['SKU_ID'].nunique())
        gc2.metric("Transacciones Afectadas", len(ghost_skus))
        gc3.metric("Ventas Fantasma (USD)", f"${ghost_sales:,.2f}")
        gc4.metric("% del Total de Ventas", f"{ghost_pct:.2f}%",
                   delta=None,
                   delta_color="inverse")

        # Interpretación contextual
        if ghost_pct > 10:
            st.error(f"⚠️ Las ventas fantasma representan más del 10% del total. **Acción crítica requerida.**")
        elif ghost_pct > 5:
            st.warning(f"⚠️ Las ventas fantasma superan el 5%. Se recomienda revisión urgente del catálogo.")
        else:
            st.info(f"✅ Las ventas fantasma están bajo control ({ghost_pct:.2f}%).")

        if not ghost_skus.empty:
            st.dataframe(ghost_skus[['SKU_ID', 'Transaccion_ID', 'Cantidad_Vendida', 'Precio_Venta_Final']].head(10))

        st.subheader("Reporte de Procesos de Limpieza")
        st.table(tx_report)

        st.divider()
        st.subheader("Salud: Feedback")
        h_score, nulls, cleaned = compute_health_score(fb_raw, fb_clean)
        c1, c2, c3 = st.columns(3)
        c1.metric("Health Score", f"{h_score:.2f}%")
        c2.metric("Nulos Totales (Raw)", nulls)
        c3.metric("Filas Removidas/Filtradas", cleaned)
        st.dataframe(fb_clean.head(20))

        st.divider()
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

    with tab_operaciones:
        st.header("Operaciones")
        st.divider()
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
                # Agrupación para tabla detallada (Ingreso = Cantidad × Precio)
                ghost_skus_agg = ghost_skus_all.copy()
                ghost_skus_agg['_Ingreso'] = ghost_skus_agg['Cantidad_Vendida'] * ghost_skus_agg['Precio_Venta_Final']
                ghost_details = ghost_skus_agg.groupby('SKU_ID').agg({
                    'Transaccion_ID': 'count',
                    'Cantidad_Vendida': 'sum',
                    '_Ingreso': 'sum'
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

        # SECCIÓN 5: Diagnóstico de Fidelidad (Stock Alto vs NPS Bajo)
        st.subheader("5️⃣ Diagnóstico de Fidelidad: Paradoja Stock Alto + NPS Bajo")

        # Filtrar datos válidos
        fidelidad_df = joined_df[
            (joined_df['Stock_Actual'].notna()) &
            (joined_df['Stock_Actual'] > 0) &
            (joined_df['Satisfaccion_NPS'].notna()) &
            (joined_df['Satisfaccion_NPS_Grupo'].notna())
            ].copy()

        if len(fidelidad_df) > 0:
            # Agregar por SKU
            fidelidad_agg = fidelidad_df.groupby('SKU_ID').agg({
                'Stock_Actual': 'mean',
                'Satisfaccion_NPS': 'mean',
                'Satisfaccion_NPS_Grupo': lambda x: x.mode()[0] if len(x) > 0 else 'desconocido',
                'Categoria': 'first',
                'Precio_Venta_Final': 'mean',
                'Transaccion_ID': 'count'
            }).reset_index()
            fidelidad_agg.columns = ['SKU_ID', 'Stock_Promedio', 'NPS_Promedio', 'NPS_Grupo', 'Categoria', 'Precio_Promedio', 'Num_Ventas']

            # Identificar paradoja
            stock_percentil_75 = fidelidad_agg['Stock_Promedio'].quantile(0.75)
            nps_bajo = fidelidad_agg['NPS_Promedio'] < 50

            paradoja_df = fidelidad_agg[
                (fidelidad_agg['Stock_Promedio'] >= stock_percentil_75) &
                (nps_bajo)
                ].copy()

            # Mapa de colores por grupo NPS
            color_map = {
                'muy_insatisfecho': '#E91E63',
                'neutro_o_ligeramente_satisfecho': '#FFC107',
                'satisfecho': '#8BC34A',
                'muy_satisfecho': '#4CAF50'
            }

            # Scatter plot
            fig_fidelidad = px.scatter(
                fidelidad_agg,
                x='Stock_Promedio',
                y='NPS_Promedio',
                color='NPS_Grupo',
                size='Num_Ventas',
                hover_data=['SKU_ID', 'Categoria', 'Precio_Promedio', 'Num_Ventas'],
                color_discrete_map=color_map,
                title='Relación entre Stock Disponible y Satisfacción del Cliente (NPS)',
                labels={
                    'Stock_Promedio': 'Stock Promedio (unidades)',
                    'NPS_Promedio': 'Satisfacción NPS Promedio',
                    'NPS_Grupo': 'Grupo NPS',
                    'Num_Ventas': 'Volumen de Ventas'
                },
                size_max=20
            )

            # Líneas de referencia
            fig_fidelidad.add_hline(y=50, line_dash="dash", line_color="gray",
                                    annotation_text="Umbral NPS Aceptable (50)",
                                    annotation_position="right")
            fig_fidelidad.add_vline(x=stock_percentil_75, line_dash="dash", line_color="blue",
                                    annotation_text=f"P75 Stock ({stock_percentil_75:.0f})",
                                    annotation_position="top")

            # Zona problemática
            fig_fidelidad.add_shape(
                type="rect",
                x0=stock_percentil_75, x1=fidelidad_agg['Stock_Promedio'].max() * 1.1,
                y0=0, y1=50,
                fillcolor="red", opacity=0.1, line_width=0
            )

            fig_fidelidad.update_layout(height=500)
            st.plotly_chart(fig_fidelidad, use_container_width=True)

            # Análisis de productos paradójicos
            st.write("**⚠️ Productos Paradójicos: Alto Stock + Baja Satisfacción**")

            if len(paradoja_df) > 0:
                st.write(f"Se encontraron **{len(paradoja_df)} productos** con alta disponibilidad pero baja satisfacción:")

                paradoja_display = paradoja_df[['SKU_ID', 'Categoria', 'Stock_Promedio', 'NPS_Promedio', 'Precio_Promedio', 'Num_Ventas']].sort_values('NPS_Promedio')
                st.dataframe(paradoja_display.head(10), use_container_width=True)

                # Análisis por categoría
                st.write("**Distribución por Categoría:**")
                cat_counts = paradoja_df['Categoria'].value_counts()
                fig_cat = px.bar(
                    x=cat_counts.index,
                    y=cat_counts.values,
                    labels={'x': 'Categoría', 'y': 'Cantidad de SKUs'},
                    title='Categorías más afectadas por la paradoja',
                    color=cat_counts.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_cat, use_container_width=True)

                # Interpretación
                st.markdown("""
                ### 🧐 Posibles Causas:
                
                1. **Problema de Calidad**: Los productos tienen defectos o no cumplen expectativas
                2. **Sobrecosto**: El precio es alto comparado con el valor percibido
                3. **Descatalogación inminente**: Stock alto por falta de demanda (círculo vicioso)
                4. **Problema logístico**: Entregas deficientes afectan la percepción del producto
                
                ### 💡 Recomendaciones:
                - Revisar reviews/comentarios de estos productos específicos
                - Comparar precios con competencia
                - Evaluar promociones o descuentos para reducir inventario
                - Considerar mejora de calidad o retiro del catálogo
                """)
            else:
                st.success("✅ No se detectaron productos con la paradoja de alto stock y baja satisfacción.")
        else:
            st.warning("⚠️ No hay datos suficientes para el análisis de fidelidad (requiere stock y NPS).")

        st.divider()

        # SECCIÓN 6: Crisis Logística y Correlación NPS
        st.subheader("6️⃣ Crisis Logística: Correlación Tiempo de Entrega vs NPS")

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

        st.divider()

        # SECCIÓN 7: Riesgo Operativo (Antigüedad Revisión vs Tickets)
        st.subheader("7️⃣ Riesgo Operativo: Antigüedad de Revisión vs Tickets de Soporte")

        st.markdown("""
        **Pregunta clave:** ¿Qué bodegas están operando "a ciegas" (sin revisar su inventario) 
        y cómo impacta esto en la satisfacción del cliente?
        """)

        # Verificar columnas necesarias
        columnas_necesarias = ['Ultima_Revision', 'Ticket_Soporte_Abierto_Limpio', 'Bodega_Origen']
        columnas_presentes = [col for col in columnas_necesarias if col in joined_df.columns]
        columnas_faltantes = [col for col in columnas_necesarias if col not in joined_df.columns]

        if columnas_faltantes:
            st.error(f"🚫 Columnas faltantes: {', '.join(columnas_faltantes)}")

        if len(columnas_presentes) == len(columnas_necesarias):
            riesgo_df = joined_df[
                (joined_df['Ultima_Revision'].notna()) &
                (joined_df['Ticket_Soporte_Abierto_Limpio'].notna()) &
                (joined_df['Bodega_Origen'].notna())
                ].copy()
            st.success(f"✅ Datos válidos encontrados: {len(riesgo_df)} registros")
        else:
            riesgo_df = pd.DataFrame()

        if len(riesgo_df) > 0:
            # Calcular días desde última revisión si no existe
            if 'Dias_desde_revision' not in riesgo_df.columns:
                hoy = pd.Timestamp.now()
                riesgo_df['Dias_desde_revision'] = (hoy - riesgo_df['Ultima_Revision']).dt.days

            # Convertir tickets a numérico
            riesgo_df['Ticket_Abierto_Num'] = (riesgo_df['Ticket_Soporte_Abierto_Limpio'] == 'Si').astype(int)

            # Agregar por Bodega
            riesgo_agg = riesgo_df.groupby('Bodega_Origen').agg({
                'Dias_desde_revision': 'mean',
                'Ticket_Abierto_Num': 'mean',
                'Satisfaccion_NPS': 'mean',
                'SKU_ID': 'nunique',
                'Transaccion_ID': 'count'
            }).reset_index()
            riesgo_agg.columns = ['Bodega_Origen', 'Dias_Promedio_Sin_Revision', 'Tasa_Tickets', 'NPS_Promedio', 'SKUs_Unicos', 'Num_Transacciones']
            riesgo_agg['Tasa_Tickets_Pct'] = riesgo_agg['Tasa_Tickets'] * 100

            # Scatter plot con zonas de riesgo
            fig_riesgo = px.scatter(
                riesgo_agg,
                x='Dias_Promedio_Sin_Revision',
                y='Tasa_Tickets_Pct',
                size='Num_Transacciones',
                color='NPS_Promedio',
                hover_data=['Bodega_Origen', 'SKUs_Unicos', 'Num_Transacciones'],
                text='Bodega_Origen',
                title='Relación entre Antigüedad de Revisión de Inventario y Tasa de Tickets de Soporte',
                labels={
                    'Dias_Promedio_Sin_Revision': 'Días promedio desde última revisión',
                    'Tasa_Tickets_Pct': 'Tasa de Tickets de Soporte (%)',
                    'NPS_Promedio': 'NPS Promedio',
                    'Num_Transacciones': 'Volumen de Transacciones'
                },
                color_continuous_scale='RdYlGn_r',
                size_max=30
            )

            # Umbrales
            umbral_dias = 90
            umbral_tickets = 30

            fig_riesgo.add_hline(y=umbral_tickets, line_dash="dash", line_color="red",
                                 annotation_text=f"Umbral Crítico: {umbral_tickets}% tickets",
                                 annotation_position="right")
            fig_riesgo.add_vline(x=umbral_dias, line_dash="dash", line_color="orange",
                                 annotation_text=f"Umbral: {umbral_dias} días sin revisar",
                                 annotation_position="top")

            # Zona de alto riesgo
            fig_riesgo.add_shape(
                type="rect",
                x0=umbral_dias, x1=riesgo_agg['Dias_Promedio_Sin_Revision'].max() * 1.1,
                y0=umbral_tickets, y1=riesgo_agg['Tasa_Tickets_Pct'].max() * 1.1,
                fillcolor="red", opacity=0.15, line_width=0
            )

            fig_riesgo.update_traces(textposition='top center')
            fig_riesgo.update_layout(height=500)
            st.plotly_chart(fig_riesgo, use_container_width=True)

            # Identificar bodegas críticas
            bodegas_criticas = riesgo_agg[
                (riesgo_agg['Dias_Promedio_Sin_Revision'] > umbral_dias) &
                (riesgo_agg['Tasa_Tickets_Pct'] > umbral_tickets)
                ].sort_values('Tasa_Tickets_Pct', ascending=False)

            if len(bodegas_criticas) > 0:
                st.error(f"🚨 **{len(bodegas_criticas)} bodega(s) en zona de alto riesgo operativo:**")

                for idx, row in bodegas_criticas.iterrows():
                    st.markdown(f"""
                    - **{row['Bodega_Origen']}**: 
                      - {row['Dias_Promedio_Sin_Revision']:.0f} días sin revisar inventario
                      - {row['Tasa_Tickets_Pct']:.1f}% de tasa de tickets
                      - NPS promedio: {row['NPS_Promedio']:.1f}
                      - {row['Num_Transacciones']} transacciones procesadas
                    """)

                st.markdown("""
                ### 🎯 Acciones Inmediatas Requeridas:
                1. **Auditoría física urgente** del inventario en estas bodegas
                2. **Implementar calendario de revisión periódica** (máx. cada 60 días)
                3. **Capacitación al personal** sobre impacto de inventario desactualizado
                4. **Análisis de root cause** de tickets de soporte asociados
                """)
            else:
                st.success("✅ Ninguna bodega se encuentra en la zona de alto riesgo operativo.")

            # Tabla detallada
            st.subheader("📊 Detalle por Bodega")
            st.dataframe(
                riesgo_agg.sort_values('Tasa_Tickets_Pct', ascending=False),
                use_container_width=True,
                column_config={
                    "Dias_Promedio_Sin_Revision": st.column_config.NumberColumn("Días sin revisión", format="%.0f"),
                    "Tasa_Tickets_Pct": st.column_config.NumberColumn("Tasa de tickets (%)", format="%.1f%%"),
                    "NPS_Promedio": st.column_config.NumberColumn("NPS Promedio", format="%.1f"),
                    "Num_Transacciones": st.column_config.NumberColumn("Transacciones", format="%d")
                }
            )

            # Análisis de correlación
            corr_revision_tickets = riesgo_agg[['Dias_Promedio_Sin_Revision', 'Tasa_Tickets_Pct']].corr().iloc[0, 1]
            corr_revision_nps = riesgo_agg[['Dias_Promedio_Sin_Revision', 'NPS_Promedio']].corr().iloc[0, 1]

            corr_col1, corr_col2, corr_col3 = st.columns(3)
            corr_col1.metric(
                "Correlación: Días sin revisar vs Tickets",
                f"{corr_revision_tickets:.3f}",
                help="Correlación positiva indica que más días sin revisar = más tickets"
            )
            corr_col2.metric(
                "Correlación: Días sin revisar vs NPS",
                f"{corr_revision_nps:.3f}",
                help="Correlación negativa indica que más días sin revisar = menor NPS"
            )
            corr_col3.metric(
                "Bodegas auditadas",
                len(riesgo_agg)
            )

            st.markdown(f"""
            ### 📈 Interpretación:
            - {'✅ **Correlación positiva detectada**' if corr_revision_tickets > 0.3 else '⚠️ Correlación débil'} 
              entre antigüedad de revisión y tickets de soporte ({corr_revision_tickets:.2f})
            - {'🚨 **Correlación negativa confirmada**' if corr_revision_nps < -0.2 else 'ℹ️ Impacto limitado'} 
              entre antigüedad de revisión y NPS ({corr_revision_nps:.2f})
            
            **Conclusión**: {'Las bodegas que no revisan su inventario frecuentemente generan más problemas operativos y menor satisfacción del cliente.' if abs(corr_revision_nps) > 0.2 else 'El impacto de la frecuencia de revisión en NPS es limitado. Considerar otros factores.'}
            """)
        else:
            st.warning("⚠️ No hay datos suficientes para el análisis de riesgo operativo (requiere fechas de revisión y tickets).")

    with tab_cliente:
        st.header("Cliente")
        if 'Satisfaccion_NPS' in fb_clean.columns:
            st.markdown("---")
            total_respuestas = len(fb_clean)
            detractores = fb_clean[fb_clean['Satisfaccion_NPS'] <= 0]
            pasivos = fb_clean[(fb_clean['Satisfaccion_NPS'] >= 0.1) & (fb_clean['Satisfaccion_NPS'] <= 50)]
            promotores = fb_clean[fb_clean['Satisfaccion_NPS'] >= 50.1]
            count_det, count_pas, count_pro = len(detractores), len(pasivos), len(promotores)
            pct_det = (count_det / total_respuestas * 100) if total_respuestas > 0 else 0
            pct_pas = (count_pas / total_respuestas * 100) if total_respuestas > 0 else 0
            pct_pro = (count_pro / total_respuestas * 100) if total_respuestas > 0 else 0
            nps_score = pct_pro - pct_det
            color_det, color_pas, color_pro = "#E91E63", "#FFC107", "#4CAF50"
            st.markdown(f"<h3 style='text-align: center;'>NPS = <span style='color:{color_pro}'>%PROMOTORES</span> - <span style='color:{color_det}'>%DETRACTORES</span></h3>", unsafe_allow_html=True)
            col_met1, col_met2, col_met3 = st.columns([1.5, 3, 1])
            with col_met1:
                fig_donut, ax_donut = plt.subplots(figsize=(4, 4))
                sizes = [max(0.1, pct_pro), max(0.1, pct_pas), max(0.1, pct_det)]
                ax_donut.pie(sizes, colors=[color_pro, color_pas, color_det], startangle=90, counterclock=False, wedgeprops={'width': 0.3, 'edgecolor': 'w', 'linewidth': 2})
                ax_donut.text(0, 0, f"{int(nps_score)}", fontsize=40, ha='center', va='center', fontweight='bold')
                st.pyplot(fig_donut)
            with col_met2:
                def metric_box(label, count, pct, color):
                    st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 10px;'><div style='background-color:{color}; color:white; padding:10px 20px; border-radius:20px; width:150px; text-align:center; font-weight:bold; margin-right:10px;'>{label}</div><div style='background-color:{color}; color:white; padding:10px; border-radius:50%; width:40px; height:40px; display:flex; align-items:center; justify-content:center; font-weight:bold; margin-right:10px;'>{count}</div><div style='background-color:#F0F2F6; padding:10px 20px; border-radius:20px; width:100px; text-align:center; font-weight:bold;'>{pct:.1f}%</div></div>", unsafe_allow_html=True)
                metric_box("detractores", count_det, pct_det, color_det)
                metric_box("pasivos", count_pas, pct_pas, color_pas)
                metric_box("promotores", count_pro, pct_pro, color_pro)
            with col_met3:
                st.markdown(f"<div style='text-align: center; margin-top: 50px;'><div style='background-color:#333; color:white; padding:5px 15px; border-radius:15px; display:inline-block; font-weight:bold;'>total</div><div style='background-color:#AAA; color:white; width:80px; height:80px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:24px; font-weight:bold; margin: 10px auto;'>{total_respuestas}</div></div>", unsafe_allow_html=True)
            st.markdown("---")
        else:
            st.warning("No se encontró información de NPS procesada.")
        if 'Edad_Cliente' in fb_clean.columns:
            st.subheader("Análisis de Outliers en Edad")
            if st.button("Ajustar Outliers de Edad a la Mediana"):
                median_age = fb_clean[fb_clean['Edad_Cliente'] <= 100]['Edad_Cliente'].median()
                outliers_mask = fb_clean['Edad_Cliente'] > 100
                num_outliers = outliers_mask.sum()
                if num_outliers > 0:
                    current_outliers = fb_clean[outliers_mask].copy()
                    st.session_state.age_outliers_log = pd.concat([st.session_state.age_outliers_log, current_outliers]).drop_duplicates()
                    fb_clean.loc[outliers_mask, 'Edad_Cliente'] = median_age
                    st.session_state.fb_clean_adjusted = fb_clean.copy()
                    add_log(f"Se ajustaron {num_outliers} outliers de edad a la mediana ({median_age}).")
                    st.success(f"Se han ajustado {num_outliers} registros.")
                    st.rerun()
                else:
                    st.info("No se encontraron outliers (> 100 años) para ajustar.")
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
            st.info("La columna 'Edad' no está presente en el dataset de Feedback para el análisis de outliers.")
        st.divider()
        st.subheader("📊 Comparativo: Tickets Abiertos (Si vs No)")
        if 'Ticket_Soporte_Abierto_Limpio' in joined_df.columns:
            ticket_counts = joined_df['Ticket_Soporte_Abierto_Limpio'].value_counts().reset_index()
            ticket_counts.columns = ['Ticket_Abierto', 'Cantidad']
            fig_tickets = px.bar(ticket_counts, x='Ticket_Abierto', y='Cantidad', color='Ticket_Abierto', text='Cantidad', title='Cantidad de Registros: Tickets Abiertos = Si vs No', labels={'Ticket_Abierto': 'Estado del Ticket', 'Cantidad': 'Número de Registros'}, color_discrete_map={'Si': '#E91E63', 'No': '#4CAF50'})
            fig_tickets.update_traces(textposition='outside')
            fig_tickets.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig_tickets, use_container_width=True)
            cant_si = ticket_counts[ticket_counts['Ticket_Abierto'] == 'Si']['Cantidad'].values
            cant_no = ticket_counts[ticket_counts['Ticket_Abierto'] == 'No']['Cantidad'].values
            cant_si = cant_si[0] if len(cant_si) > 0 else 0
            cant_no = cant_no[0] if len(cant_no) > 0 else 0
            total = cant_si + cant_no
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Tickets Abiertos = Si", f"{cant_si:,}")
            col_m2.metric("Tickets Abiertos = No", f"{cant_no:,}")
            col_m3.metric("% Tickets Abiertos (Sí)", f"{(cant_si/total*100):.1f}%" if total > 0 else "0%")
            st.dataframe(ticket_counts, use_container_width=True)
        else:
            st.info("ℹ️ Columna 'Ticket_Soporte_Abierto_Limpio' no encontrada.")

    with tab_insights_ia:
        st.header("Insights de IA")
        st.caption("El análisis se basa exclusivamente en los filtros aplicados en el panel lateral.")
        # Build agent context from all data
        h_inv = compute_health_score(inv_raw, inv_clean)
        h_tx = compute_health_score(tx_raw, tx_clean)
        h_fb = compute_health_score(fb_raw, fb_clean)
        health_scores = {
            "Inventario": h_inv,
            "Transacciones": h_tx,
            "Feedback": h_fb,
        }

        nps_score_val = None
        if 'Satisfaccion_NPS' in fb_clean.columns:
            total_resp = len(fb_clean)
            if total_resp > 0:
                detractores = len(fb_clean[fb_clean['Satisfaccion_NPS'] <= 0])
                pasivos = len(fb_clean[(fb_clean['Satisfaccion_NPS'] > 0) & (fb_clean['Satisfaccion_NPS'] < 50.1)])
                promotores = len(fb_clean[fb_clean['Satisfaccion_NPS'] >= 50.1])
                nps_score_val = (promotores / total_resp * 100) - (detractores / total_resp * 100)

        ghost_skus_tab6 = tx_clean[tx_clean['flag_sku_fantasma'] == True]
        total_sales_t = (tx_clean['Cantidad_Vendida'] * tx_clean['Precio_Venta_Final']).sum()
        ghost_sales_t = (ghost_skus_tab6['Cantidad_Vendida'] * ghost_skus_tab6['Precio_Venta_Final']).sum()
        ghost_pct_t = (ghost_sales_t / total_sales_t * 100) if total_sales_t > 0 else 0

        margen_neg = joined_df[joined_df['Margen_Neto_aprox'] < 0]
        margen_neg_cnt = len(margen_neg)
        margen_neg_loss = margen_neg['Margen_Neto_aprox'].sum() if margen_neg_cnt > 0 else 0

        agent_context = feda.build_agent_context(
            inv_raw=inv_raw,
            inv_clean=inv_clean,
            inv_report=inv_report,
            tx_raw=tx_raw,
            tx_clean=tx_clean,
            tx_report=tx_report,
            fb_raw=fb_raw,
            fb_clean=fb_clean,
            joined_df=joined_df,
            logs=st.session_state.logs,
            user_comments=st.session_state.user_comments,
            health_scores=health_scores,
            nps_score=nps_score_val,
            ghost_skus_count=ghost_skus_tab6['SKU_ID'].nunique() if not ghost_skus_tab6.empty else 0,
            ghost_sales_pct=ghost_pct_t,
            margen_negativo_count=margen_neg_cnt,
            margen_negativo_loss=margen_neg_loss,
        )

        groq_api_key = _get_groq_api_key()
        if not GROQ_AVAILABLE:
            st.error("⚠️ El paquete `groq` no está instalado. Ejecute: `pip install groq`")
        elif not groq_api_key:
            st.info("Configure su API Key de Groq en el panel lateral (Configurar API Key) o en .streamlit/secrets.toml (GROQ_API_KEY) para habilitar los Insights de IA.")
        else:
            INSIGHT_PROMPT = """A partir del siguiente contexto del dashboard (datos ya filtrados por el usuario), genera entre 5 y 10 insights concretos y recomendaciones accionables. Responde en español, de forma clara y numerada. El contexto es exclusivamente el que se proporciona debajo."""

            if st.button("🚀 Generar Insights con IA", type="primary", use_container_width=True):
                with st.spinner("Generando insights con Llama-3..."):
                    try:
                        client = Groq(api_key=groq_api_key)
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": INSIGHT_PROMPT + "\n\n--- CONTEXTO DEL DASHBOARD (filtros aplicados) ---\n\n" + agent_context},
                                {"role": "user", "content": "Genera los insights y recomendaciones basados en el contexto anterior."},
                            ],
                            temperature=0.4,
                        )
                        st.session_state.insight_ia_result = response.choices[0].message.content
                        st.session_state.insight_ia_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al contactar la API de Groq: {str(e)}")

            if st.session_state.insight_ia_result:
                st.subheader("📋 Insights y recomendaciones")
                if st.session_state.insight_ia_timestamp:
                    st.caption(f"Generado: {st.session_state.insight_ia_timestamp}")
                st.markdown(st.session_state.insight_ia_result)
                st.divider()

            st.subheader("Chat con el agente (seguimiento)")
            SYSTEM_PROMPT = """Eres un asistente experto en análisis de datos, calidad de datos y business intelligence.
Tienes acceso al contexto completo del análisis exploratorio de datos (EDA) y las métricas de calidad de un proyecto.
El contexto incluye: resúmenes de inventario, transacciones, feedback, NPS, SKUs fantasma, márgenes negativos, logs de limpieza y comentarios del analista.
Responde en español de forma clara y concisa. Da recomendaciones prácticas cuando sea apropiado.
Si te preguntan algo fuera del contexto, indica amablemente que solo puedes responder sobre los datos cargados en el dashboard."""

            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            if prompt := st.chat_input("Escriba su pregunta sobre los datos..."):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        try:
                            client = Groq(api_key=groq_api_key)
                            messages = [
                                {"role": "system", "content": SYSTEM_PROMPT + "\n\n--- CONTEXTO DEL DASHBOARD ---\n\n" + agent_context},
                            ]
                            for m in st.session_state.chat_messages:
                                messages.append({"role": m["role"], "content": m["content"]})

                            response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=messages,
                                temperature=0.4,
                            )
                            reply = response.choices[0].message.content
                            st.markdown(reply)
                            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            err_msg = f"Error al contactar la API de Groq: {str(e)}"
                            st.error(err_msg)
                            st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})

            if st.session_state.chat_messages and st.button("🗑️ Limpiar historial de chat"):
                st.session_state.chat_messages = []
                st.rerun()

else:
    st.info("Por favor, carga los tres archivos CSV en el panel lateral para comenzar.")

    # Placeholder for structure info if user needs help
    with st.expander("Ver estructura esperada de archivos"):
        st.write("**Inventario:** SKU_ID, Stock_Actual, Costo_Unitario_USD, Categoria, Bodega_Origen, Lead_Time_Dias, Ultima_Revision")
        st.write("**Transacciones:** Transaccion_ID, SKU_ID, Fecha_Venta, Cantidad_Vendida, Precio_Venta_Final, Ciudad_Destino, Canal_Venta, Estado_Envio")
        st.write("**Feedback:** Feedback_ID, Transaccion_ID, Satisfaccion_NPS, Comentario_Texto, Recomienda_Marca")
