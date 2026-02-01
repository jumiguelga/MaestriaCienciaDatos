# functions_eda.py (fragmentos clave)

import pandas as pd
from datetime import datetime
import extras.dictionaries as dicts
from sklearn.impute import KNNImputer
import re
import numpy as np

# ...

def imputar_costo_envio_knn(transacciones, n_neighbors=5):
    """
    Imputa Costo_Envio usando KNN y crea columnas de trazabilidad.
    """
    df = transacciones.copy()

    df["Costo_Envio_Original"] = df["Costo_Envio"]

    cols_base = [
        "Costo_Envio",
        "Cantidad_Vendida",
        "Precio_Venta_Final",
        "Tiempo_Entrega_Real",
    ]

    for c in cols_base:
        if c not in df.columns:
            raise ValueError(f"La columna '{c}' no existe en el DataFrame")

    df_knn = df[cols_base].copy()

    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    imputed_array = imputer.fit_transform(df_knn)
    df_imputed = pd.DataFrame(imputed_array, columns=cols_base, index=df.index)

    df["Costo_Envio_Imputado"] = df["Costo_Envio"].isna().astype(int)
    df["Costo_Envio"] = df_imputed["Costo_Envio"]

    return df


# ---------------------------
# Funciones opcionales transacciones
# ---------------------------

def excluir_ventas_cantidad_negativa(transacciones: pd.DataFrame) -> pd.DataFrame:
    """
    Excluye filas con Cantidad_Vendida < 0.
    """
    df = transacciones.copy()
    mask = df["Cantidad_Vendida"] < 0
    df = df[~mask].copy()
    return df


def corregir_o_excluir_ventas_futuras(
    transacciones: pd.DataFrame,
    modo: str = "corregir",
) -> pd.DataFrame:
    """
    Manejo de ventas futuras:
    - modo='corregir': resta 1 año a las fechas futuras.
    - modo='excluir': elimina filas con Fecha_Venta en el futuro.
    """
    df = transacciones.copy()
    df["Fecha_Venta"] = pd.to_datetime(df["Fecha_Venta"], errors="coerce")
    today = pd.Timestamp(datetime.today().date())
    future_mask = df["Fecha_Venta"] > today

    if modo == "corregir":
        df.loc[future_mask, "Fecha_Venta"] = df.loc[future_mask, "Fecha_Venta"].apply(
            lambda d: d.replace(year=d.year - 1) if pd.notnull(d) else d
        )
    elif modo == "excluir":
        df = df[~future_mask].copy()
    else:
        raise ValueError("modo debe ser 'corregir' o 'excluir'")

    return df


def filtrar_skus_fantasma(
    transacciones: pd.DataFrame,
    inventario: pd.DataFrame,
    incluir_fantasma: bool = True,
) -> pd.DataFrame:
    """
    Maneja SKUs presentes en transacciones pero no en inventario.
    - incluir_fantasma=True: devuelve todo, con flag.
    - incluir_fantasma=False: excluye filas con SKU no encontrado.
    """
    df = transacciones.copy()
    inv_skus = set(inventario["SKU_ID"].astype("string").str.strip())

    df["SKU_ID"] = df["SKU_ID"].astype("string").str.strip()
    df["flag_sku_fantasma"] = ~df["SKU_ID"].isin(inv_skus)

    if incluir_fantasma:
        return df
    else:
        return df[~df["flag_sku_fantasma"]].copy()


# ---------------------------
# Feedback: duplicados
# ---------------------------

def excluir_feedback_duplicado(feedback: pd.DataFrame) -> pd.DataFrame:
    """
    Excluye filas con Feedback_ID duplicado, conservando la primera ocurrencia.
    """
    df = feedback.copy()
    df = df[~df["Feedback_ID"].duplicated(keep="first")].copy()
    return df


# ---------------------------
# Feedback básico (ajustado)
# ---------------------------

def limpiar_feedback_basico(feedback: pd.DataFrame) -> pd.DataFrame:
    df = feedback.copy()

    if "Comentario_Texto" in df.columns:
        df["Comentario_Texto"] = df["Comentario_Texto"].fillna("No_responde")

    if "Recomienda_Marca" in df.columns:
        df["Recomienda_Marca"] = df["Recomienda_Marca"].fillna("No_responde")

    if "Satisfaccion_NPS" in df.columns:
        def agrupar_nps(x: float) -> str:
            if pd.isna(x):
                return "NPS_desconocido"
            if x < 0:
                return "muy_insatisfecho"
            if 0 <= x < 30:
                return "neutro_o_ligeramente_satisfecho"
            if 30 <= x < 70:
                return "satisfecho"
            return "muy_satisfecho"

        df["Satisfaccion_NPS_Grupo"] = df["Satisfaccion_NPS"].apply(agrupar_nps)

    if "Ticket_Soporte_Abierto" in df.columns:
        df["Ticket_Soporte_Abierto_Limpio"] = (
            df["Ticket_Soporte_Abierto"]
            .map(dicts.map_ticket_soporte)
            .fillna("Desconocido")
        )

    return df
