# data_analysis.py

import pandas as pd
import altair as alt
import streamlit as st

def show_inventory_analysis(inv_final: pd.DataFrame):
    st.subheader("Inventario – análisis básico")
    st.write(inv_final.describe(include="all"))

def show_transactions_analysis(tx_final: pd.DataFrame):
    st.subheader("Transacciones – análisis básico")
    st.write(tx_final.describe(include="all"))

def show_feedback_analysis(fb_final: pd.DataFrame):
    st.subheader("Feedback – análisis básico")
    st.write(fb_final.describe(include="all"))

def show_joined_analysis(joined: pd.DataFrame):
    st.subheader("Dataset final (JOIN)")
    st.write(joined.head())

    # Ejemplo simple de visual
    if "Ingreso" in joined.columns and "Categoria" in joined.columns:
        chart = (
            alt.Chart(joined)
            .mark_bar()
            .encode(
                x="Categoria:N",
                y="sum(Ingreso):Q",
            )
        )
        st.altair_chart(chart, use_container_width=True)
