import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# ===== IMPORTS LOCALES =====
from functions_eda import (
    sanitize_inventario,
    sanitize_transacciones,
    limpiar_feedback_basico,
    imputar_costo_envio_knn,
    excluir_ventas_cantidad_negativa,
    corregir_o_excluir_ventas_futuras,
    enriquecer_con_estado_envio_reglas,
    filtrar_skus_fantasma,
    excluir_feedback_duplicado,
    build_join_dataset,
    feature_engineering,
)
from health_report import compute_health_metrics
from data_analysis import (
    compute_analysis,
    show_inventory_analysis,
    show_transactions_analysis,
    show_feedback_analysis,
    show_p1_margen_negativo,
    show_p2_logistica_nps,
    show_p3_sku_fantasma,
    show_p4_stock_nps,
    show_p5_bodega_tickets,
)

# ===== CONFIGURACIÓN STREAMLIT =====
st.set_page_config(
    page_title="Challenge 02 – Data Cleaning & Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏢 Dashboard Integral: Limpieza, EDA y Análisis de Negocio")
st.markdown("**Limpieza auditable | Health Score | JOIN | Feature Engineering | Análisis P1..P5**")

# ===== SIDEBAR: CARGA DE DATOS =====
st.sidebar.header("📂 1. Cargar Datos")

inv_file = st.sidebar.file_uploader("Inventario (CSV)", type=["csv"], key="inv")
tx_file = st.sidebar.file_uploader("Transacciones (CSV)", type=["csv"], key="tx")
fb_file = st.sidebar.file_uploader("Feedback (CSV)", type=["csv"], key="fb")


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


inventario_raw = load_csv(inv_file) if inv_file is not None else pd.DataFrame()
transacciones_raw = load_csv(tx_file) if tx_file is not None else pd.DataFrame()
feedback_raw = load_csv(fb_file) if fb_file is not None else pd.DataFrame()

if inventario_raw.empty or transacciones_raw.empty or feedback_raw.empty:
    st.warning("⚠️ Por favor carga los tres archivos CSV para iniciar el análisis.")
    st.stop()

st.success("✅ Archivos cargados correctamente.")

# ===== SIDEBAR: OPCIONES DE LIMPIEZA OPCIONAL =====
st.sidebar.header("⚙️ 2. Opciones de Limpieza (opcional)")

opt_imputar_costo_envio = st.sidebar.checkbox(
    "Imputar costo de envío (KNN)", value=False
)
opt_excluir_cant_neg = st.sidebar.checkbox(
    "Excluir ventas con cantidad negativa", value=False
)
opt_modo_futuras = st.sidebar.selectbox(
    "Ventas futuras",
    ["no_tocar", "corregir", "excluir"],
    index=0,
)
opt_estado_envio_reglas = st.sidebar.checkbox(
    "Enriquecer con Estado_Envio_Reglas", value=False
)
opt_incluir_skus_fantasma = st.sidebar.checkbox(
    "Incluir SKUs no en inventario", value=True
)
opt_excluir_feedback_dup = st.sidebar.checkbox(
    "Excluir Feedback_ID duplicados", value=False
)

# ===== SECCIÓN 1: LIMPIEZA ESTÁNDAR =====
st.header("1️⃣ Limpieza Estándar (Obligatoria)")

with st.expander("📋 Expandir para ver resumen de limpieza", expanded=True):
    inv_clean, inv_report = sanitize_inventario(inventario_raw)
    tx_clean, tx_report = sanitize_transacciones(transacciones_raw)
    fb_clean = limpiar_feedback_basico(feedback_raw)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Inventario")
        st.dataframe(inv_report, use_container_width=True)
    with col2:
        st.subheader("Transacciones")
        st.dataframe(tx_report, use_container_width=True)
    with col3:
        st.subheader("Feedback")
        st.metric("Filas iniciales", len(feedback_raw))
        st.metric("Filas finales", len(fb_clean))

# ===== SECCIÓN 2: LIMPIEZA OPCIONAL =====
st.header("2️⃣ Limpieza Opcional")

tx_final = tx_clean.copy()
fb_final = fb_clean.copy()

if opt_imputar_costo_envio and not tx_final.empty:
    tx_final = imputar_costo_envio_knn(tx_final)
    st.info("✅ Costo de envío imputado con KNN")

if opt_excluir_cant_neg and not tx_final.empty:
    tx_final = excluir_ventas_cantidad_negativa(tx_final)
    st.info("✅ Ventas con cantidad negativa excluidas")

if opt_modo_futuras != "no_tocar" and not tx_final.empty:
    modo = "corregir" if opt_modo_futuras == "corregir" else "excluir"
    tx_final = corregir_o_excluir_ventas_futuras(tx_final, modo=modo)
    st.info(f"✅ Ventas futuras: {modo}")

if opt_estado_envio_reglas and not tx_final.empty and not inv_clean.empty:
    tx_final = enriquecer_con_estado_envio_reglas(tx_final, inventario=inv_clean)
    st.info("✅ Estado_Envio_Reglas derivado")

if not inv_clean.empty and not tx_final.empty:
    tx_final = filtrar_skus_fantasma(
        tx_final,
        inventario=inv_clean,
        incluir_fantasma=opt_incluir_skus_fantasma,
    )

if opt_excluir_feedback_dup and not fb_final.empty:
    fb_final = excluir_feedback_duplicado(fb_final)
    st.info("✅ Feedback_ID duplicados excluidos")

col1, col2 = st.columns(2)
with col1:
    st.metric("Transacciones finales", len(tx_final))
with col2:
    st.metric("Feedback final", len(fb_final))

# ===== SECCIÓN 3: HEALTH SCORE & OUTLIERS =====
st.header("3️⃣ Health Score y Anomalías")

flags = [
    "Costo_Envio_Imputado",
    "flag_sku_fantasma",
    "flag_sin_feedback",
    "outlier_costo",
    "outlier_precio",
]

health = compute_health_metrics(
    raw_df=transacciones_raw,
    clean_df=tx_clean,
    final_df=tx_final,
    flags=flags,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filas (raw)", health["rows_raw"])
with col2:
    st.metric("Filas (final)", health["rows_final"])
with col3:
    st.metric("Health Score (final)", f"{health['health_score_final']:.1f}%")
with col4:
    st.metric("Filas eliminadas", health["rows_removed"], delta=f"{health['pct_removed']}%", delta_color="inverse")

with st.expander("📊 Detalles completos del Health Score y Outliers"):
    st.json(health)
    
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        if "outlier_costo" in inv_clean.columns:
            n_out_inv = inv_clean["outlier_costo"].sum()
            st.warning(f"Outliers en Costo_Unitario: {int(n_out_inv)}")
            if n_out_inv > 0:
                st.dataframe(inv_clean[inv_clean["outlier_costo"]], use_container_width=True)
    with col_out2:
        if "outlier_precio" in tx_final.columns:
            n_out_tx = tx_final["outlier_precio"].sum()
            st.warning(f"Outliers en Precio_Venta: {int(n_out_tx)}")
            if n_out_tx > 0:
                st.dataframe(tx_final[tx_final["outlier_precio"]], use_container_width=True)

# ===== SECCIÓN 4: JOIN DATASET =====
st.header("4️⃣ Dataset Integrado (JOIN)")

joined = build_join_dataset(tx_final, inv_clean, fb_final)
joined = feature_engineering(joined)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Filas en JOIN", len(joined))
with col2:
    ingreso_total = joined.get("Ingreso", pd.Series([0])).sum()
    st.metric("Ingreso total", f"${ingreso_total:,.0f}")
with col3:
    margen_total = joined.get("Margen_Bruto", pd.Series([0])).sum()
    st.metric("Margen bruto", f"${margen_total:,.0f}")

with st.expander("🔍 Ver primeras filas del JOIN"):
    st.dataframe(joined.head(20), use_container_width=True)

# ===== SECCIÓN 5: ANÁLISIS P1..P5 =====
st.header("5️⃣ Análisis de Negocio (P1..P5)")

analysis_results = compute_analysis(joined)

# Tabs para cada análisis
tab_p1, tab_p2, tab_p3, tab_p4, tab_p5 = st.tabs([
    "P1 — Margen Negativo",
    "P2 — Logística vs NPS",
    "P3 — SKU Fantasma",
    "P4 — Stock vs NPS",
    "P5 — Bodega Tickets"
])

with tab_p1:
    show_p1_margen_negativo(analysis_results)

with tab_p2:
    show_p2_logistica_nps(analysis_results)

with tab_p3:
    show_p3_sku_fantasma(analysis_results)

with tab_p4:
    show_p4_stock_nps(analysis_results)

with tab_p5:
    show_p5_bodega_tickets(analysis_results)

# ===== SECCIÓN 6: EDA GENERAL =====
st.header("6️⃣ Análisis Exploratorio (EDA)")

tab_inv, tab_tx, tab_fb = st.tabs(["📦 Inventario", "🚚 Transacciones", "💬 Feedback"])

with tab_inv:
    show_inventory_analysis(inv_clean)

with tab_tx:
    show_transactions_analysis(tx_final)

with tab_fb:
    show_feedback_analysis(fb_final)

# ===== PIE DE PÁGINA =====
st.divider()
st.markdown(
    """
    ---
    **Dashboard de Limpieza y Análisis Integral**  
    Limpieza estándar + Limpieza opcional + Health Score + JOIN + Feature Engineering + P1..P5
    """
)
