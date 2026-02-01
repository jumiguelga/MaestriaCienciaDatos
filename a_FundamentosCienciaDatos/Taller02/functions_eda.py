import pandas as pd
from datetime import datetime
import dictionaries as dicts
from sklearn.impute import KNNImputer
import numpy as np


# -------------------------------------------------------------------
# Limpieza estándar: INVENTARIO
# -------------------------------------------------------------------
def sanitize_inventario(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanea el DataFrame de inventario y devuelve:
    (DataFrame saneado, DataFrame resumen de filas afectadas por cada proceso).
    """
    df = dataframe.copy()
    report = {}

    # 1. Remover espacios en blanco en columnas categóricas
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    affected_spaces = 0
    for col in cat_cols:
        mask = df[col].notnull() & (df[col] != df[col].astype(str).str.strip())
        affected_spaces += mask.sum()
        df[col] = df[col].where(df[col].isnull(), df[col].astype(str).str.strip())
    report["Remover espacios en blanco"] = int(affected_spaces)

    # 2. Reemplazar cadenas vacías por NA
    before_empty = df[cat_cols].isin([""]).sum().sum()
    df[cat_cols] = df[cat_cols].replace({"": pd.NA})
    report["Reemplazar cadenas vacías por NA"] = int(before_empty)

    # 3. Conversión de fechas en 'Ultima_Revision'
    before_invalid_dates = df["Ultima_Revision"].isnull().sum()
    df["Ultima_Revision"] = pd.to_datetime(
        df["Ultima_Revision"], format="%Y-%m-%d", errors="coerce"
    ).dt.strftime("%d/%m/%Y")
    df["Ultima_Revision"] = pd.to_datetime(
        df["Ultima_Revision"], dayfirst=True, errors="coerce"
    )
    after_invalid_dates = df["Ultima_Revision"].isnull().sum()
    report["Conversión de fechas inválidas a NaT"] = int(
        after_invalid_dates - before_invalid_dates
    )

    # 4. Stock negativo a positivo
    neg_stock = (df["Stock_Actual"] < 0).sum()
    df.loc[df["Stock_Actual"] < 0, "Stock_Actual"] = (
        df.loc[df["Stock_Actual"] < 0, "Stock_Actual"].abs()
    )
    report["Stock negativo convertido a positivo"] = int(neg_stock)

    # 5. Normalización de Bodega_Origen
    before_bodega = (
        df["Bodega_Origen"].notnull()
        & df["Bodega_Origen"].isin(dicts.mapping_bodegas.keys())
    ).sum()
    df["Bodega_Origen"] = (
        df["Bodega_Origen"]
        .astype("string")
        .str.strip()
        .replace(dicts.mapping_bodegas)
    )
    after_bodega = (
        df["Bodega_Origen"].notnull()
        & df["Bodega_Origen"].isin(dicts.mapping_bodegas.values())
    ).sum()
    report["Normalización de Bodega_Origen"] = int(before_bodega)

    # 6. Normalización de Categoria
    before_cat = (
        df["Categoria"].notnull()
        & df["Categoria"].isin(dicts.mapping_categorias.keys())
    ).sum()
    df["Categoria"] = (
        df["Categoria"].astype("string").str.strip().replace(dicts.mapping_categorias)
    )
    after_cat = (
        df["Categoria"].notnull()
        & df["Categoria"].isin(dicts.mapping_categorias.values())
    ).sum()
    report["Normalización de Categoria"] = int(before_cat)

    # 7. Rellenar nulos en Lead_Time_Dias y normalizar texto
    affected_lead = 0
    if "Lead_Time_Dias" in df.columns:
        col = "Lead_Time_Dias"
        mask = df[col].isnull() | (df[col].astype(str).str.strip() == "")
        affected_lead += mask.sum()
        df[col] = df[col].astype("string").str.strip().replace({"": pd.NA})
        df[col] = df[col].fillna("sin_definir").astype("string").str.lower()
    report["Rellenar nulos y normalizar Lead_Time_Dias"] = int(affected_lead)

    # 8. Rellenar nulos en Stock_Actual con 0
    affected_stock_null = 0
    if "stock_actual" in df.columns and "Stock_Actual" not in df.columns:
        df = df.rename(columns={"stock_actual": "Stock_Actual"})
    if "Stock_Actual" in df.columns:
        mask = df["Stock_Actual"].isnull()
        affected_stock_null = mask.sum()
        df["Stock_Actual"] = pd.to_numeric(
            df["Stock_Actual"], errors="coerce"
        ).fillna(0)
    report["Rellenar nulos en Stock_Actual con 0"] = int(affected_stock_null)

    inventarios_report = pd.DataFrame(
        list(report.items()), columns=["Proceso", "Filas_afectadas"]
    )
    return df, inventarios_report


# -------------------------------------------------------------------
# Limpieza estándar: TRANSACCIONES
# -------------------------------------------------------------------
def sanitize_transacciones(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanea el DataFrame de transacciones y devuelve:
    (DataFrame saneado, DataFrame resumen de filas afectadas por cada proceso).
    """
    df = dataframe.copy()
    report = {}

    # 1. Conversión de fechas en 'Fecha_Venta'
    before_invalid_dates = df["Fecha_Venta"].isnull().sum()
    df["Fecha_Venta"] = pd.to_datetime(df["Fecha_Venta"], format="%d/%m/%Y", errors="coerce")
    after_invalid_dates = df["Fecha_Venta"].isnull().sum()
    report["Conversión de fechas inválidas a NaT"] = int(
        after_invalid_dates - before_invalid_dates
    )

    # 2. Corregir fechas futuras en 'Fecha_Venta' (ajustar el año al anterior)
    today = pd.Timestamp(datetime.today().date())
    future_mask = df["Fecha_Venta"] > today
    affected_future_dates = future_mask.sum()
    df.loc[future_mask, "Fecha_Venta"] = df.loc[future_mask, "Fecha_Venta"].apply(
        lambda d: d.replace(year=d.year - 1) if pd.notnull(d) else d
    )
    report["Fechas futuras corregidas (año -1)"] = int(affected_future_dates)

    # 3. Normalización de Ciudades Destino
    before_city = (
        df["Ciudad_Destino"].notnull()
        & df["Ciudad_Destino"].isin(dicts.mapping_ciudades_destino.keys())
    ).sum()
    df["Ciudad_Destino"] = (
        df["Ciudad_Destino"]
        .astype("string")
        .str.strip()
        .replace(dicts.mapping_ciudades_destino)
    )
    after_city = (
        df["Ciudad_Destino"].notnull()
        & df["Ciudad_Destino"].isin(dicts.mapping_ciudades_destino.values())
    ).sum()
    report["Normalización de Ciudad_Destino"] = int(before_city)

    # 4. Normalización de Canal_Venta
    before_channel = (
        df["Canal_Venta"].notnull()
        & df["Canal_Venta"].isin(dicts.mapping_canal_venta.keys())
    ).sum()
    df["Canal_Venta"] = (
        df["Canal_Venta"]
        .astype("string")
        .str.strip()
        .replace(dicts.mapping_canal_venta)
    )
    after_channel = (
        df["Canal_Venta"].notnull()
        & df["Canal_Venta"].isin(dicts.mapping_canal_venta.values())
    ).sum()
    report["Normalización de Canal_Venta"] = int(before_channel)

    # 5. Normalización de Estado_Envio
    if "Estado_Envio" in df.columns:
        before_status = (
            df["Estado_Envio"].notnull()
            & df["Estado_Envio"].isin(dicts.mapping_estado_envio.keys())
        ).sum()
        df["Estado_Envio"] = (
            df["Estado_Envio"]
            .astype("string")
            .str.strip()
            .replace(dicts.mapping_estado_envio)
        )
        after_status = (
            df["Estado_Envio"].notnull()
            & df["Estado_Envio"].isin(dicts.mapping_estado_envio.values())
        ).sum()
        report["Normalización de Estado_Envio"] = int(before_status)

    transacciones_report = pd.DataFrame(
        list(report.items()), columns=["Proceso", "Filas_afectadas"]
    )
    return df, transacciones_report


# -------------------------------------------------------------------
# Limpieza opcional: transacciones y feedback
# (las que ya habíamos armado)
# -------------------------------------------------------------------

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

    df["Costo_Envio_Imputado"] = df["Costo_Envio_Original"].isna().astype(int)
    df["Costo_Envio"] = df_imputed["Costo_Envio"]

    return df


def excluir_ventas_cantidad_negativa(transacciones: pd.DataFrame) -> pd.DataFrame:
    df = transacciones.copy()
    mask = df["Cantidad_Vendida"] < 0
    return df[~mask].copy()


def corregir_o_excluir_ventas_futuras(
    transacciones: pd.DataFrame,
    modo: str = "corregir",
) -> pd.DataFrame:
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
    df = transacciones.copy()
    inv_skus = set(inventario["SKU_ID"].astype("string").str.strip())

    df["SKU_ID"] = df["SKU_ID"].astype("string").str.strip()
    df["flag_sku_fantasma"] = ~df["SKU_ID"].isin(inv_skus)

    if incluir_fantasma:
        return df
    else:
        return df[~df["flag_sku_fantasma"]].copy()


def excluir_feedback_duplicado(feedback: pd.DataFrame) -> pd.DataFrame:
    df = feedback.copy()
    df = df[~df["Feedback_ID"].duplicated(keep="first")].copy()
    return df


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
