import streamlit as st
import pandas as pd

from functions_eda import (
    sanitize_inventario,
    sanitize_transacciones,
    imputar_costo_envio_knn,
    enriquecer_con_estado_envio_reglas,  # si luego la quieres usar
    excluir_ventas_cantidad_negativa,
    corregir_o_excluir_ventas_futuras,
    filtrar_skus_fantasma,
    limpiar_feedback_basico,
    excluir_feedback_duplicado,
)
from health_report import compute_health_metrics
from data_analysis import (
    show_inventory_analysis,
    show_transactions_analysis,
    show_feedback_analysis,
    show_joined_analysis,
)


st.set_page_config(page_title="Challenge 02 – Data Cleaning & EDA", layout="wide")

st.title("Dashboard de Limpieza y EDA")

# -------------------------
# Carga de datos por el usuario
# -------------------------

st.sidebar.header("1. Carga de archivos CSV")

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
    st.warning("Por favor carga los tres archivos: inventario, transacciones y feedback para iniciar el análisis.")
    st.stop()

st.success("Archivos cargados correctamente. Continúa con las opciones de limpieza y análisis.")

# -------------------------
# Opciones de limpieza opcional
# -------------------------

st.sidebar.header("2. Opciones de limpieza opcional")

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
    "Derivar Estado_Envio_Reglas (lead time)", value=False
)
opt_incluir_skus_fantasma = st.sidebar.checkbox(
    "Incluir SKUs que no están en inventario", value=True
)

opt_excluir_feedback_dup = st.sidebar.checkbox(
    "Excluir Feedback_ID duplicados", value=False
)

# -------------------------
# Limpieza estándar
# -------------------------

st.header("1. Limpieza estándar")

inv_clean, inv_report = sanitize_inventario(inventario_raw)
tx_clean, tx_report = sanitize_transacciones(transacciones_raw)
fb_clean = limpiar_feedback_basico(feedback_raw)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Inventario – resumen de limpieza")
    st.dataframe(inv_report)
with col2:
    st.subheader("Transacciones – resumen de limpieza")
    st.dataframe(tx_report)

# -------------------------
# Limpieza opcional
# -------------------------

st.header("2. Limpieza opcional")

tx_final = tx_clean.copy()
fb_final = fb_clean.copy()

if opt_imputar_costo_envio and not tx_final.empty:
    tx_final = imputar_costo_envio_knn(tx_final)

if opt_excluir_cant_neg and not tx_final.empty:
    tx_final = excluir_ventas_cantidad_negativa(tx_final)

if opt_modo_futuras != "no_tocar" and not tx_final.empty:
    modo = "corregir" if opt_modo_futuras == "corregir" else "excluir"
    tx_final = corregir_o_excluir_ventas_futuras(tx_final, modo=modo)

if opt_estado_envio_reglas and not tx_final.empty and not inv_clean.empty:
    tx_final = enriquecer_con_estado_envio_reglas(tx_final, inventario=inv_clean)

if not inv_clean.empty and not tx_final.empty:
    tx_final = filtrar_skus_fantasma(
        tx_final,
        inventario=inv_clean,
        incluir_fantasma=opt_incluir_skus_fantasma,
    )

if opt_excluir_feedback_dup and not fb_final.empty:
    fb_final = excluir_feedback_duplicado(fb_final)

st.write("Filas transacciones finales:", len(tx_final))
st.write("Filas feedback finales:", len(fb_final))

# -------------------------
# Health score
# -------------------------

st.header("3. Health score")

flags = [
    "Costo_Envio_Imputado",
    "flag_sku_fantasma",
]

health = compute_health_metrics(
    raw_df=transacciones_raw,
    clean_df=tx_clean,
    final_df=tx_final,
    flags=flags,
)

st.json(health)

# -------------------------
# EDA / Visualizaciones
# -------------------------

st.header("4. Análisis exploratorio")

tab_inv, tab_tx, tab_fb, tab_join = st.tabs(
    ["Inventario", "Transacciones", "Feedback", "JOIN"]
)

with tab_inv:
    show_inventory_analysis(inv_clean)

with tab_tx:
    show_transactions_analysis(tx_final)

with tab_fb:
    show_feedback_analysis(fb_final)

with tab_join:
    joined = pd.DataFrame()  # TODO: armar JOIN más adelante
    show_joined_analysis(joined)
