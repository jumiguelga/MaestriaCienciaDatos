import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from typing import Dict, Any, List

# -------------------------------------------------------------------
# ANÁLISIS P1..P5: FUNCIONES DE NEGOCIO
# -------------------------------------------------------------------

def compute_analysis(joined: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula análisis P1..P5 desde el dataset joined.
    Retorna dict con todas las métricas, tablas y flags para visualizaciones.
    """
    results = {}
    
    if joined.empty:
        return results
    
    # =================== P1: MARGEN NEGATIVO ===================
    df = joined.copy()
    
    if "Margen_Bruto" in df.columns and "SKU_ID" in df.columns and "Categoria" in df.columns:
        margen_por_sku = (
            df.groupby("SKU_ID")
            .agg({
                "Margen_Bruto": ["sum", "mean", "count"],
                "Ingreso": "sum",
                "Cantidad_Vendida": "sum",
            })
            .reset_index()
        )
        margen_por_sku.columns = ["SKU_ID", "Margen_Total", "Margen_Promedio", "Cantidad_Total", "Ingreso_Total", "Cantidad_Vendida"]
        margen_negativo = margen_por_sku[margen_por_sku["Margen_Total"] < 0].copy()
        results["margen_negativo"] = margen_negativo
        
        margen_por_categoria = (
            df.groupby("Categoria")
            .agg({
                "Margen_Bruto": "sum",
                "Ingreso": "sum",
                "Cantidad_Vendida": "sum",
            })
            .reset_index()
        )
        margen_por_categoria.columns = ["Categoria_clean", "Margen_Total", "Ingreso_Total", "Cantidad"]
        results["margen_por_categoria"] = margen_por_categoria
    
    # =================== P2: LOGÍSTICA VS NPS ===================
    
    if all(col in df.columns for col in ["Ciudad_Destino", "Bodega_Origen", "Tiempo_Entrega_Real", "Satisfaccion_NPS"]):
        logistica_nps = (
            df.groupby(["Ciudad_Destino", "Bodega_Origen"])
            .agg({
                "Tiempo_Entrega_Real": "mean",
                "Satisfaccion_NPS": "mean",
                "Transaccion_ID": "count",
            })
            .reset_index()
        )
        logistica_nps.columns = ["Ciudad_Destino_clean", "Bodega_Origen_clean", "Tiempo_Entrega_Prom", "NPS_Prom", "N"]
        results["logistica_vs_nps"] = logistica_nps
    
    # =================== P3: SKU FANTASMA (NO EN INVENTARIO) ===================
    
    if "flag_sku_fantasma" in df.columns and "Ingreso" in df.columns:
        fantasma_tx = df[df["flag_sku_fantasma"] == True].copy()
        
        if not fantasma_tx.empty:
            num_tx = len(fantasma_tx)
            total_ingreso_perdido = fantasma_tx["Ingreso"].sum()
            total_ingreso = df["Ingreso"].sum()
            porcentaje = total_ingreso_perdido / total_ingreso if total_ingreso > 0 else 0
            
            results["sku_fantasma"] = {
                "num_transacciones": num_tx,
                "total_perdido": total_ingreso_perdido,
                "porcentaje": porcentaje,
            }
            
            if "Categoria" in fantasma_tx.columns:
                fantasma_por_cat = (
                    fantasma_tx.groupby("Categoria")
                    .agg({"Ingreso": "sum"})
                    .reset_index()
                )
                fantasma_por_cat.columns = ["Categoria_fantasma", "Ingreso_Perdido"]
                results["sku_fantasma_por_categoria"] = fantasma_por_cat
        else:
            results["sku_fantasma"] = {
                "num_transacciones": 0,
                "total_perdido": 0,
                "porcentaje": 0,
            }
    
    # =================== P4: STOCK ALTO + NPS BAJO ===================
    
    if all(col in df.columns for col in ["Stock_Actual", "Satisfaccion_NPS", "Categoria"]):
        stock_nps = (
            df.groupby("Categoria")
            .agg({
                "Stock_Actual": "mean",
                "Satisfaccion_NPS": "mean",
                "Transaccion_ID": "count",
            })
            .reset_index()
        )
        stock_nps.columns = ["Categoria_clean", "Stock_Prom", "NPS_Prom", "N"]
        results["stock_vs_nps_scatter"] = stock_nps
        
        # Alerta: Stock alto + NPS bajo
        high_stock_low_nps = stock_nps[
            (stock_nps["Stock_Prom"] > stock_nps["Stock_Prom"].quantile(0.75)) &
            (stock_nps["NPS_Prom"] < stock_nps["NPS_Prom"].quantile(0.25))
        ].copy()
        results["stock_alto_nps_bajo_alerta"] = high_stock_low_nps
    
    # =================== P5: TICKET RATE POR BODEGA ===================
    
    if all(col in df.columns for col in ["Bodega_Origen", "flag_sin_feedback"]):
        ticket_rate = (
            df.groupby("Bodega_Origen")
            .agg({
                "flag_sin_feedback": lambda x: x.sum() / len(x) if len(x) > 0 else 0,
                "Transaccion_ID": "count",
            })
            .reset_index()
        )
        ticket_rate.columns = ["Bodega_Origen_clean", "Ticket_Rate", "N_transacciones"]
        results["riesgo_bodega_plus"] = ticket_rate
    
    # =================== DONUT: INGRESO EN RIESGO ===================
    
    if "flag_sku_fantasma" in df.columns and "Ingreso" in df.columns:
        ingreso_riesgo = df[df["flag_sku_fantasma"] == True]["Ingreso"].sum()
        ingreso_normal = df[df["flag_sku_fantasma"] == False]["Ingreso"].sum()
        
        donut_data = pd.DataFrame({
            "Tipo": ["Con SKU fantasma", "Ingreso normal"],
            "Valor": [ingreso_riesgo, ingreso_normal]
        })
        results["donut_ingreso_riesgo_fantasma"] = donut_data
    
    return results


# -------------------------------------------------------------------
# VISUALIZACIONES INDIVIDUALES (PARA TABS)
# -------------------------------------------------------------------

def show_inventory_analysis(inv_final: pd.DataFrame):
    """Resumen estadístico del inventario limpio."""
    st.subheader("📦 Inventario – Análisis")
    if inv_final is None or inv_final.empty:
        st.warning("Sin datos de inventario.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total SKUs", len(inv_final))
    with col2:
        st.metric("Stock total", int(inv_final.get("Stock_Actual", [0]).sum()))
    with col3:
        st.metric("Bodegas", inv_final.get("Bodega_Origen", pd.Series()).nunique())
    
    st.dataframe(inv_final.describe(include="all"), use_container_width=True)


def show_transactions_analysis(tx_final: pd.DataFrame):
    """Resumen estadístico de transacciones limpias."""
    st.subheader("🚚 Transacciones – Análisis")
    if tx_final is None or tx_final.empty:
        st.warning("Sin datos de transacciones.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total transacciones", len(tx_final))
    with col2:
        ingreso = tx_final.get("Ingreso", pd.Series([0])).sum()
        st.metric("Ingreso total", f"${ingreso:,.0f}")
    with col3:
        margen = tx_final.get("Margen_Bruto", pd.Series([0])).sum()
        st.metric("Margen bruto", f"${margen:,.0f}")
    
    st.dataframe(tx_final.describe(include="all"), use_container_width=True)


def show_feedback_analysis(fb_final: pd.DataFrame):
    """Resumen estadístico del feedback."""
    st.subheader("💬 Feedback – Análisis")
    if fb_final is None or fb_final.empty:
        st.warning("Sin datos de feedback.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total feedback", len(fb_final))
    with col2:
        if "Satisfaccion_NPS" in fb_final.columns:
            nps_prom = fb_final["Satisfaccion_NPS"].mean()
            st.metric("NPS promedio", f"{nps_prom:.1f}")
    with col3:
        if "Ticket_Soporte_Abierto_Limpio" in fb_final.columns:
            tickets = (fb_final["Ticket_Soporte_Abierto_Limpio"] == "Si").sum()
            st.metric("Tickets abiertos", tickets)
    
    st.dataframe(fb_final.describe(include="all"), use_container_width=True)


def show_p1_margen_negativo(analysis_results: Dict[str, Any]):
    """P1: SKUs con margen negativo."""
    st.subheader("P1 — Margen Negativo")
    
    if "margen_negativo" not in analysis_results:
        st.info("Sin datos para P1.")
        return
    
    mneg = analysis_results["margen_negativo"]
    
    if mneg.empty:
        st.success("✅ No hay SKUs con margen negativo.")
    else:
        st.warning(f"⚠️ {len(mneg)} SKUs con margen NEGATIVO")
        st.dataframe(mneg.sort_values("Margen_Total"), use_container_width=True)


def show_p2_logistica_nps(analysis_results: Dict[str, Any]):
    """P2: Logística (tiempo entrega) vs Satisfacción (NPS)."""
    st.subheader("P2 — Logística vs NPS por Ciudad+Bodega")
    
    if "logistica_vs_nps" not in analysis_results:
        st.info("Sin datos para P2.")
        return
    
    df = analysis_results["logistica_vs_nps"]
    
    if df.empty:
        st.info("Sin datos para graficar.")
        return
    
    # Scatter: Tiempo entrega vs NPS
    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.7, size=100)
        .encode(
            x=alt.X("Tiempo_Entrega_Prom:Q", title="Tiempo entrega (días)"),
            y=alt.Y("NPS_Prom:Q", title="NPS promedio"),
            color=alt.Color("Bodega_Origen_clean:N", title="Bodega"),
            size=alt.Size("N:Q", title="# transacciones"),
            tooltip=["Ciudad_Destino_clean", "Bodega_Origen_clean", "Tiempo_Entrega_Prom", "NPS_Prom", "N"]
        )
        .properties(height=350, title="Entrega vs Satisfacción por ciudad/bodega")
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df, use_container_width=True)


def show_p3_sku_fantasma(analysis_results: Dict[str, Any]):
    """P3: SKU Fantasma (en transacciones pero no en inventario)."""
    st.subheader("P3 — SKU Fantasma (ingreso en riesgo)")
    
    if "sku_fantasma" not in analysis_results:
        st.info("Sin datos para P3.")
        return
    
    fantasma = analysis_results["sku_fantasma"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Transacciones fantasma", fantasma["num_transacciones"])
    with col2:
        st.metric("Ingreso en riesgo", f"${fantasma['total_perdido']:,.0f}")
    with col3:
        st.metric("% del ingreso", f"{fantasma['porcentaje']*100:.2f}%")
    
    if "sku_fantasma_por_categoria" in analysis_results:
        st.dataframe(
            analysis_results["sku_fantasma_por_categoria"].sort_values("Ingreso_Perdido", ascending=False),
            use_container_width=True
        )


def show_p4_stock_nps(analysis_results: Dict[str, Any]):
    """P4: Stock alto vs NPS bajo por categoría."""
    st.subheader("P4 — Stock Alto + NPS Bajo (por categoría)")
    
    if "stock_vs_nps_scatter" not in analysis_results:
        st.info("Sin datos para P4.")
        return
    
    df = analysis_results["stock_vs_nps_scatter"]
    
    if df.empty:
        st.info("Sin datos.")
        return
    
    # Scatter
    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.7, size=100)
        .encode(
            x=alt.X("Stock_Prom:Q", title="Stock promedio"),
            y=alt.Y("NPS_Prom:Q", title="NPS promedio"),
            color=alt.Color("Categoria_clean:N", title="Categoría"),
            size=alt.Size("N:Q", title="# tx"),
            tooltip=["Categoria_clean", "Stock_Prom", "NPS_Prom", "N"]
        )
        .properties(height=350, title="Stock vs Satisfacción")
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df, use_container_width=True)


def show_p5_bodega_tickets(analysis_results: Dict[str, Any]):
    """P5: Ticket rate (sin feedback) por bodega."""
    st.subheader("P5 — Tasa de Tickets por Bodega")
    
    if "riesgo_bodega_plus" not in analysis_results:
        st.info("Sin datos para P5.")
        return
    
    df = analysis_results["riesgo_bodega_plus"].sort_values("Ticket_Rate", ascending=False)
    
    if df.empty:
        st.info("Sin datos.")
        return
    
    # Bar chart
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Bodega_Origen_clean:N", title="Bodega", sort="-y"),
            y=alt.Y("Ticket_Rate:Q", title="Tasa de tickets (sin feedback)"),
            color=alt.condition(
                alt.datum.Ticket_Rate > df["Ticket_Rate"].mean(),
                alt.value("red"),
                alt.value("green")
            ),
            tooltip=["Bodega_Origen_clean", "Ticket_Rate", "N_transacciones"]
        )
        .properties(height=300, title="Riesgo por bodega (% sin feedback)")
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df, use_container_width=True)
