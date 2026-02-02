import pandas as pd
import numpy as np
from datetime import datetime
import dictionaries as dicts
from sklearn.impute import KNNImputer
import re
import unicodedata
from typing import Dict, Tuple, List, Any, Optional

# -------------------------------------------------------------------
# HELPERS: NORMALIZACIÓN Y OUTLIERS
# -------------------------------------------------------------------

def normalize_text(x: Any) -> Any:
    """Normaliza texto: quita acentos, caracteres especiales y pasa a minúsculas."""
    if pd.isna(x):
        return np.nan
    raw = str(x).strip()
    if raw == "":
        return np.nan
    # Normalización unicode y remover acentos
    x2 = unicodedata.normalize("NFKD", raw.lower()).encode("ascii", "ignore").decode("utf-8")
    # Quitar caracteres especiales excepto espacios y números
    x2 = re.sub(r"[^a-z0-9\s]", " ", x2)
    x2 = re.sub(r"\s+", " ", x2).strip()
    return x2 or np.nan

def iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    """Calcula límites IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def outlier_flag_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.Series:
    """Retorna máscara booleana de outliers para una columna."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    base = s.dropna()
    if len(base) < 5: # Mínimo para que tenga sentido
        return pd.Series(False, index=df.index)
    low, high = iqr_bounds(base, k)
    return s.notna() & ((s < low) | (s > high))

# -------------------------------------------------------------------
# LIMPIEZA ESTÁNDAR: INVENTARIO
# -------------------------------------------------------------------

def sanitize_inventario(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanea el DataFrame de inventario y devuelve:
    (DataFrame saneado, DataFrame resumen de filas afectadas por cada proceso).
    """
    df = dataframe.copy()
    report = {}

# 1. Remover espacios en blanco y normalizar texto
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    affected_norm = 0
    id_cols = ["SKU_ID", "Transaccion_ID", "Feedback_ID"]
    for col in cat_cols:
        if col in id_cols:
            continue
        original = df[col].copy()
        df[col] = df[col].apply(normalize_text)
        affected_norm += (df[col] != original).sum()
    report["Normalización de texto (minúsculas, acentos, etc.)"] = int(affected_norm)

    # 2. Reemplazar nulos/vacíos en categóricas (ya manejado por normalize_text en parte)
    report["Reemplazar cadenas vacías por NA"] = int(df[cat_cols].isna().sum().sum())

    # 2.1 Renombrar columnas si es necesario para consistencia
    if "stock_actual" in df.columns and "Stock_Actual" not in df.columns:
        df = df.rename(columns={"stock_actual": "Stock_Actual"})
    if "lead_time_dias" in df.columns and "Lead_Time_Dias" not in df.columns:
        df = df.rename(columns={"lead_time_dias": "Lead_Time_Dias"})

    # 3. Conversión de fechas en 'Ultima_Revision'
    if "Ultima_Revision" in df.columns:
        before_invalid_dates = df["Ultima_Revision"].isnull().sum()
        df["Ultima_Revision"] = pd.to_datetime(
            df["Ultima_Revision"], format="%Y-%m-%d", errors="coerce"
        )
        after_invalid_dates = df["Ultima_Revision"].isnull().sum()
        report["Conversión de fechas inválidas a NaT"] = int(
            after_invalid_dates - before_invalid_dates
        )

    # 4. Stock negativo a positivo
    if "Stock_Actual" in df.columns:
        neg_stock = (df["Stock_Actual"] < 0).sum()
        df.loc[df["Stock_Actual"] < 0, "Stock_Actual"] = (
            df.loc[df["Stock_Actual"] < 0, "Stock_Actual"].abs()
        )
        report["Stock negativo convertido a positivo"] = int(neg_stock)

    # 5. Normalización de Bodega_Origen
    if "Bodega_Origen" in df.columns:
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
        report["Normalización de Bodega_Origen"] = int(before_bodega)

    # 6. Normalización de Categoria
    if "Categoria" in df.columns:
        df["Categoria"] = (
            df["Categoria"]
            .astype("string")
            .str.strip()
            .replace(dicts.mapping_categorias)
        )
        report["Normalización de Categoria"] = int(df["Categoria"].notnull().sum())

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
    if "Stock_Actual" in df.columns:
        mask = df["Stock_Actual"].isnull()
        affected_stock_null = mask.sum()
        df["Stock_Actual"] = pd.to_numeric(
            df["Stock_Actual"], errors="coerce"
        ).fillna(0)
    report["Rellenar nulos en Stock_Actual con 0"] = int(affected_stock_null)
    # 9. Detección de outliers (IQR)
    if "Costo_Unitario_USD" in df.columns:
        df["outlier_costo"] = outlier_flag_iqr(df, "Costo_Unitario_USD")
        report["Outliers detectados en Costo_Unitario_USD"] = int(df["outlier_costo"].sum())

    inventarios_report = pd.DataFrame(
        list(report.items()), columns=["Proceso", "Filas_afectadas"]
    )
    return df, inventarios_report


# -------------------------------------------------------------------
# LIMPIEZA ESTÁNDAR: TRANSACCIONES
# -------------------------------------------------------------------

def sanitize_transacciones(dataframe: pd.DataFrame, normalize_status: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanea el DataFrame de transacciones y devuelve:
    (DataFrame saneado, DataFrame resumen de filas afectadas por cada proceso).
    """
    df = dataframe.copy()
    report = {}

    # 0. Normalización de texto básica
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    id_cols = ["SKU_ID", "Transaccion_ID", "Feedback_ID"]
    for col in cat_cols:
        if col in id_cols:
            continue
        df[col] = df[col].apply(normalize_text)

    # 1. Conversión de fechas en 'Fecha_Venta'
    if "Fecha_Venta" in df.columns:
        before_invalid_dates = df["Fecha_Venta"].isnull().sum()
        df["Fecha_Venta"] = pd.to_datetime(df["Fecha_Venta"], format="%d/%m/%Y", errors="coerce")
        after_invalid_dates = df["Fecha_Venta"].isnull().sum()
        report["Conversión de fechas inválidas a NaT"] = int(
            after_invalid_dates - before_invalid_dates
        )

    # 2. Corregir fechas futuras en 'Fecha_Venta' (ajustar el año al anterior)
    if "Fecha_Venta" in df.columns:
        today = pd.Timestamp(datetime.today().date())
        future_mask = df["Fecha_Venta"] > today
        affected_future_dates = future_mask.sum()
        df.loc[future_mask, "Fecha_Venta"] = df.loc[future_mask, "Fecha_Venta"].apply(
            lambda d: d.replace(year=d.year - 1) if pd.notnull(d) else d
        )
        report["Fechas futuras corregidas (año -1)"] = int(affected_future_dates)

    # 3. Normalización de Ciudades Destino
    if "Ciudad_Destino" in df.columns:
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
        report["Normalización de Ciudad_Destino"] = int(before_city)

    # 4. Normalización de Canal_Venta
    if "Canal_Venta" in df.columns:
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
        report["Normalización de Canal_Venta"] = int(before_channel)

    # 5. Normalización de Estado_Envio
    if "Estado_Envio" in df.columns:
        if normalize_status:
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
            report["Normalización de Estado_Envio"] = int(before_status)
        else:
            report["Normalización de Estado_Envio"] = 0

    # 6. Detección de outliers (IQR)
    if "Precio_Venta_Final" in df.columns:
        df["outlier_precio"] = outlier_flag_iqr(df, "Precio_Venta_Final")
        report["Outliers detectados en Precio_Venta_Final"] = int(df["outlier_precio"].sum())

    transacciones_report = pd.DataFrame(
        list(report.items()), columns=["Proceso", "Filas_afectadas"]
    )
    return df, transacciones_report


# -------------------------------------------------------------------
# LIMPIEZA OPCIONAL: TRANSACCIONES
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
    """Excluye filas con Cantidad_Vendida < 0."""
    df = transacciones.copy()
    mask = df["Cantidad_Vendida"] < 0
    return df[~mask].copy()


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


# -------------------------------------------------------------------
# LIMPIEZA OPCIONAL: FEEDBACK
# -------------------------------------------------------------------

def excluir_feedback_duplicado(feedback: pd.DataFrame) -> pd.DataFrame:
    """Excluye filas con Feedback_ID duplicado, conservando la primera ocurrencia."""
    df = feedback.copy()
    df = df[~df["Feedback_ID"].duplicated(keep="first")].copy()
    return df


def limpiar_feedback_basico(feedback: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica del dataset de feedback:
    - Comentario_Texto y Recomienda_Marca: nulos -> 'No_responde'
    - Agrupa Satisfaccion_NPS en categorías
    - Normaliza Ticket_Soporte_Abierto con diccionario
    """
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


# -------------------------------------------------------------------
# HELPER: ENRIQUECIMIENTO CON ESTADO_ENVIO_REGLAS
# -------------------------------------------------------------------

def parse_lead_time_to_days(value):
    """Convierte '25', '25-30 dias', '25-30 días' -> número (promedio del rango).
    Si no puede parsear, devuelve NaN.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    nums = re.findall(r"\d+", s)
    if not nums:
        return np.nan
    nums = list(map(int, nums))
    if len(nums) == 1:
        return float(nums[0])
    return float((min(nums) + max(nums)) / 2.0)


def enriquecer_con_estado_envio_reglas(
    transacciones: pd.DataFrame,
    inventario: pd.DataFrame,
    hoy: str = "2026-02-01",
    margen_dias: int = 2,
) -> pd.DataFrame:
    """Enriquece transacciones con Lead_Time_Dias numérico y Estado_Envio_Reglas."""
    df = transacciones.copy()

    if "SKU_ID" not in df.columns:
        raise ValueError("Se requiere columna 'SKU_ID' en transacciones")
    if "SKU_ID" not in inventario.columns:
        raise ValueError("Se requiere columna 'SKU_ID' en inventario")
    if "Lead_Time_Dias" not in inventario.columns:
        raise ValueError("Se requiere columna 'Lead_Time_Dias' en inventario")

    inv = inventario.copy()
    inv["Lead_Time_Dias_num"] = inv["Lead_Time_Dias"].apply(parse_lead_time_to_days)

    df = df.merge(
        inv[["SKU_ID", "Lead_Time_Dias_num"]],
        on="SKU_ID",
        how="left",
    )

    df["Fecha_Venta"] = pd.to_datetime(df["Fecha_Venta"], errors="coerce")
    hoy_dt = pd.to_datetime(hoy)
    df["dias_desde_venta"] = (hoy_dt - df["Fecha_Venta"]).dt.days

    def clasificar_estado(row):
        estado = row.get("Estado_Envio")
        lead_time = row.get("Lead_Time_Dias_num")
        dias = row.get("dias_desde_venta")

        if pd.notna(estado):
            return estado
        if pd.isna(lead_time) or pd.isna(dias):
            return "estado_desconocido"
        if dias <= lead_time + margen_dias:
            return "aun_en_tiempo_o_en_camino"
        return "posible_retraso"

    df["Estado_Envio_Reglas"] = df.apply(clasificar_estado, axis=1)

    return df


# -------------------------------------------------------------------
# JOIN: TRANSACCIONES ↔ INVENTARIO ↔ FEEDBACK
# -------------------------------------------------------------------

def build_join_dataset(
    tx_final: pd.DataFrame,
    inv_clean: pd.DataFrame,
    fb_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el JOIN final: Transacciones ↔ Inventario ↔ Feedback
    Incluye flags de SKU fantasma y sin feedback.
    """
    # Preparar datos
    invj = inv_clean.copy()
    invj["SKU_ID"] = invj["SKU_ID"].astype("string").str.strip()
    
    txj = tx_final.copy()
    txj["SKU_ID"] = txj["SKU_ID"].astype("string").str.strip()
    if "Transaccion_ID" in txj.columns:
        txj["Transaccion_ID"] = txj["Transaccion_ID"].astype("string").str.strip()
    
    fbj = fb_clean.copy()
    if "Transaccion_ID" in fbj.columns:
        fbj["Transaccion_ID"] = fbj["Transaccion_ID"].astype("string").str.strip()
    
    # JOIN: Tx ↔ Inv
    join_tx_inv = txj.merge(
        invj, on="SKU_ID", how="left", suffixes=("", "_inv"), indicator="merge_tx_inv"
    )
    join_tx_inv["flag_sku_fantasma"] = (join_tx_inv["merge_tx_inv"] == "left_only")
    
    # Resolver colisiones de nombres si las hay (excepto SKU_ID que es la llave)
    # Si hay columnas con el mismo nombre en tx e inv, la de inv tendrá sufijo _inv
    # Pero queremos mantener las de tx como principales.
    
    # JOIN: (Tx+Inv) ↔ Feedback
    joined = join_tx_inv.merge(
        fbj, on="Transaccion_ID", how="left", indicator="merge_tx_fb"
    )
    joined["flag_sin_feedback"] = (joined["merge_tx_fb"] == "left_only")
    
    # Limpiar columnas de merge
    joined = joined.drop(columns=["merge_tx_inv", "merge_tx_fb"], errors="ignore")
    
    return joined


# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------

def feature_engineering(joined: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features: Ingreso, Costo_producto, Margen_Bruto, Margen_Neto_aprox, Dias_desde_revision
    """
    df = joined.copy()
    
    # Conversión a numérico
    df["Cantidad_Vendida"] = pd.to_numeric(df.get("Cantidad_Vendida", 0), errors="coerce").fillna(0)
    df["Precio_Venta_Final"] = pd.to_numeric(df.get("Precio_Venta_Final", 0), errors="coerce").fillna(0)
    df["Costo_Unitario_USD"] = pd.to_numeric(df.get("Costo_Unitario_USD", 0), errors="coerce").fillna(0)
    df["Costo_Envio"] = pd.to_numeric(df.get("Costo_Envio", 0), errors="coerce").fillna(0)
    df["Stock_Actual"] = pd.to_numeric(df.get("Stock_Actual", 0), errors="coerce").fillna(0)
    
    # Ingreso, Costos, Márgenes
    # Si Costo_Unitario_USD vino de Inventario, ahora se llama Costo_Unitario_USD (o Costo_Unitario_USD_inv si hubo colisión)
    costo_col = "Costo_Unitario_USD" if "Costo_Unitario_USD" in df.columns else "Costo_Unitario_USD_inv"
    
    df["Ingreso"] = df["Cantidad_Vendida"] * df["Precio_Venta_Final"]
    df["Costo_producto"] = df["Cantidad_Vendida"] * pd.to_numeric(df.get(costo_col, 0), errors="coerce").fillna(0)
    df["Margen_Bruto"] = df["Ingreso"] - df["Costo_producto"]
    df["Margen_Neto_aprox"] = df["Margen_Bruto"] - df["Costo_Envio"]
    
    # Días desde última revisión
    if "Ultima_Revision" in df.columns:
        today_dt = pd.Timestamp(datetime.now().date())
        df["Ultima_Revision"] = pd.to_datetime(df["Ultima_Revision"], errors="coerce")
        df["Dias_desde_revision"] = (today_dt - df["Ultima_Revision"].dt.floor("D")).dt.days
    
    return df


def aplicar_exclusion_global(
    inv_df: pd.DataFrame, 
    tx_df: pd.DataFrame, 
    fb_df: pd.DataFrame, 
    exclude_outliers: bool = False, 
    exclude_nulls: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aplica exclusión de outliers y nulos de forma global a los tres datasets.
    """
    inv_c = inv_df.copy()
    tx_c = tx_df.copy()
    fb_c = fb_df.copy()

    if exclude_outliers:
        if "outlier_costo" in inv_c.columns:
            inv_c = inv_c[inv_c["outlier_costo"] == False]
        if "outlier_precio" in tx_c.columns:
            tx_c = tx_c[tx_c["outlier_precio"] == False]
        if "Edad_Cliente" in fb_c.columns:
            age_outliers = outlier_flag_iqr(fb_c, "Edad_Cliente")
            fb_c = fb_c[~age_outliers]

    if exclude_nulls:
        inv_c = inv_c.dropna()
        tx_c = tx_c.dropna()
        fb_c = fb_c.dropna()

    return inv_c, tx_c, fb_c
