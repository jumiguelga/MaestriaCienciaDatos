import pandas as pd
from datetime import datetime
import extras.dictionaries as dicts


# Funciones útiles para el análisis de datos
def print_table(title: str, dataframe: pd.DataFrame, rows: int = 10, show_index: bool = False):
    """
    Muestra un DataFrame en forma de tabla simple.
    Parámetros:
      - title: Título que se mostrará arriba de la tabla.
      - dataframe: pd.DataFrame a mostrar.
      - rows: Número de filas a mostrar (por defecto 10).
      - show_index: Mostrar índice (por defecto False).
    """
    print(f"\n{title}")
    print(f"{dataframe.shape[0]} filas x {dataframe.shape[1]} columnas\n")

    if dataframe.empty:
        print("DataFrame vacío")
        return

    df_show = dataframe.head(rows)
    if not show_index:
        df_show = df_show.reset_index(drop=True)

    display(df_show)


# Función de saneamiento de la información.
def sanitize_inventario(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanea el DataFrame de inventario y devuelve una dupla:
    (DataFrame saneado, DataFrame resumen de filas afectadas por cada proceso).
    """
    dataframe = dataframe.copy()
    report = {}

    # 1. Remover espacios en blanco en columnas categóricas
    cat_cols = dataframe.select_dtypes(include=['object', 'string']).columns
    affected_spaces = 0
    for col in cat_cols:
        mask = dataframe[col].notnull() & (dataframe[col] != dataframe[col].astype(str).str.strip())
        affected_spaces += mask.sum()
        dataframe[col] = dataframe[col].where(dataframe[col].isnull(),
                                              dataframe[col].astype(str).str.strip())
    report["Remover espacios en blanco"] = int(affected_spaces)

    # 2. Reemplazar cadenas vacías por NA
    before_empty = dataframe[cat_cols].isin(['']).sum().sum()
    dataframe[cat_cols] = dataframe[cat_cols].replace({'': pd.NA})
    report["Reemplazar cadenas vacías por NA"] = int(before_empty)

    # 3. Conversión de fechas en 'Ultima_Revision'
    before_invalid_dates = dataframe['Ultima_Revision'].isnull().sum()
    dataframe['Ultima_Revision'] = pd.to_datetime(dataframe['Ultima_Revision'], format='%Y-%m-%d', errors='coerce').dt.strftime('%d/%m/%Y')
    dataframe['Ultima_Revision'] = pd.to_datetime(dataframe['Ultima_Revision'], dayfirst=True, errors='coerce')
    after_invalid_dates = dataframe['Ultima_Revision'].isnull().sum()
    report["Conversión de fechas inválidas a NaT"] = int(after_invalid_dates - before_invalid_dates)

    # 4. Stock negativo a positivo
    neg_stock = (dataframe['Stock_Actual'] < 0).sum()
    dataframe.loc[dataframe['Stock_Actual'] < 0, 'Stock_Actual'] = dataframe.loc[dataframe['Stock_Actual'] < 0, 'Stock_Actual'].abs()
    report["Stock negativo convertido a positivo"] = int(neg_stock)

    # 5. Normalización de Bodega_Origen
    
    before_bodega = (dataframe['Bodega_Origen'].notnull() & dataframe['Bodega_Origen'].isin(dicts.mapping_bodegas.keys())).sum()
    dataframe['Bodega_Origen'] = dataframe['Bodega_Origen'].astype("string").str.strip().replace(dicts.mapping_bodegas)
    after_bodega = (dataframe['Bodega_Origen'].notnull() & dataframe['Bodega_Origen'].isin(dicts.mapping_bodegas.values())).sum()
    report["Normalización de Bodega_Origen"] = int(before_bodega)

    # 6. Normalización de Categoria
    before_cat = (dataframe['Categoria'].notnull() & dataframe['Categoria'].isin(dicts.mapping_categorias.keys())).sum()
    dataframe['Categoria'] = dataframe['Categoria'].astype("string").str.strip().replace(dicts.mapping_categorias)
    after_cat = (dataframe['Categoria'].notnull() & dataframe['Categoria'].isin(dicts.mapping_categorias.values())).sum()
    report["Normalización de Categoria"] = int(before_cat)

    # 7. Rellenar nulos en Lead_Time_Dias y normalizar texto
    affected_lead = 0
    for col in ('Lead_Time_Dias', 'leads_time_dias'):
        if col in dataframe.columns:
            mask = dataframe[col].isnull() | (dataframe[col].astype(str).str.strip() == '')
            affected_lead += mask.sum()
            dataframe[col] = dataframe[col].astype("string").str.strip().replace({'': pd.NA})
            dataframe[col] = dataframe[col].fillna('sin_definir').astype("string").str.lower()
    report["Rellenar nulos y normalizar Lead_Time_Dias"] = int(affected_lead)

    # 8. Renombrar y rellenar nulos en Stock_Actual
    affected_stock_null = 0
    if 'stock_actual' in dataframe.columns and 'Stock_Actual' not in dataframe.columns:
        dataframe = dataframe.rename(columns={'stock_actual': 'Stock_Actual'})
    if 'Stock_Actual' in dataframe.columns:
        mask = dataframe['Stock_Actual'].isnull()
        affected_stock_null = mask.sum()
        dataframe['Stock_Actual'] = pd.to_numeric(dataframe['Stock_Actual'], errors='coerce').fillna(0)
    report["Rellenar nulos en Stock_Actual con 0"] = int(affected_stock_null)

    inventarios_report = pd.DataFrame(list(report.items()), columns=["Proceso", "Filas_afectadas"])
    return dataframe, inventarios_report

def sanitize_transacciones(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanea el DataFrame de transacciones y devuelve una dupla:
    (DataFrame saneado, DataFrame resumen de filas afectadas por cada proceso).
    """
    dataframe = dataframe.copy()
    report = {}

    # 1. Conversión de fechas en 'Fecha_Venta'
    before_invalid_dates = dataframe['Fecha_Venta'].isnull().sum()
    dataframe['Fecha_Venta'] = pd.to_datetime(dataframe['Fecha_Venta'], format='%d/%m/%Y', errors='coerce')
    after_invalid_dates = dataframe['Fecha_Venta'].isnull().sum()
    report["Conversión de fechas inválidas a NaT"] = int(after_invalid_dates - before_invalid_dates)

    # 2. Corregir fechas futuras en 'Fecha_Venta' (solo ajustar el año al anterior)

    today = pd.Timestamp(datetime.today().date())
    future_mask = dataframe['Fecha_Venta'] > today
    affected_future_dates = future_mask.sum()
    dataframe.loc[future_mask, 'Fecha_Venta'] = dataframe.loc[future_mask, 'Fecha_Venta'].apply(
        lambda d: d.replace(year=d.year - 1) if pd.notnull(d) else d
    )
    report["Fechas futuras corregidas (año -1)"] = int(affected_future_dates)

    # 3. Normalización de Ciudades Destino
    before_city = (dataframe['Ciudad_Destino'].notnull() & dataframe['Ciudad_Destino'].isin(dicts.mapping_ciudades.keys())).sum()
    dataframe['Ciudad_Destino'] = dataframe['Ciudad_Destino'].astype("string").str.strip().replace(dicts.mapping_ciudades)
    after_city = (dataframe['Ciudad_Destino'].notnull() & dataframe['Ciudad_Destino'].isin(dicts.mapping_ciudades.values())).sum()
    report["Normalización de Ciudad_Destino"] = int(before_city)

    # 4. Normalización de Canal_Venta
    before_channel = (dataframe['Canal_Venta'].notnull() & dataframe['Canal_Venta'].isin(dicts.mapping_canal_venta.keys())).sum()
    dataframe['Canal_Venta'] = dataframe['Canal_Venta'].astype("string").str.strip().replace(dicts.mapping_canal_venta)
    after_channel = (dataframe['Canal_Venta'].notnull() & dataframe['Canal_Venta'].isin(dicts.mapping_canal_venta.values())).sum()
    report["Normalización de Canal_Venta"] = int(before_channel)

    # 4. Normalización de Estado_Envio
    before_status = (dataframe['Estado_Envio'].notnull() & dataframe['Estado_Envio'].isin(dicts.mapping_estado_envio.keys())).sum()
    dataframe['Estado_Envio'] = dataframe['Estado_Envio'].astype("string").str.strip().replace(dicts.mapping_estado_envio)
    after_status = (dataframe['Estado_Envio'].notnull() & dataframe['Estado_Envio'].isin(dicts.mapping_estado_envio.values())).sum()
    report["Normalización de Estado_Envio"] = int(before_status)


    transacciones_report = pd.DataFrame(list(report.items()), columns=["Proceso", "Filas_afectadas"])
    return dataframe, transacciones_report