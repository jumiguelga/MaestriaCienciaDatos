# main_app.py

import streamlit as st
import pandas as pd

from functions_eda import (
    sanitize_inventario,
    sanitize_transacciones,
    imputar_costo_envio_knn,
    enriquecer_con_estado_envio_reglas,
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
# Carga de datos (placeholder: aquí ajustas a tu lógica real)
# -------------------------
@st.cache_data
def load_data():
    # Sustituir por lectura real (CSV, etc.)
    inv = pd.DataFrame()
    tx = pd.DataFrame()
    fb = pd.DataFrame()
    return inv, tx, fb

inventario_raw, transacciones_raw, feedback_raw = load_data()

st.sidebar.header("Opciones de limpieza opcional")

# Flags transacciones
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

# Feedback
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

st.subheader("Inventario – resumen de limpieza")
st.dataframe(inv_report)

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
    # Aquí deberías construir el JOIN real (similar a Act2.py)
    joined = pd.DataFrame()
    show_joined_analysis(joined)
