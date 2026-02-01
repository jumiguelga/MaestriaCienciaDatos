# Dashboard con Streamlit - Integración de Limpieza y Análisis Exploratorio de Datos (EDA)

Generar un nuevo archivo main_app.py que contenga la lógica principal de la aplicación, integrando las funciones de limpieza y análisis exploratorio de datos. Debe de ajustarse a Stramlit para la interfaz de usuario. Tener en cuenta el archivo de referencia de Act2.py, y documentar muy bien el código para saber por secciones está compuesto. De ser necesario, se sugiere crear varios archivos adicionales para organizar mejor el código.

## Archivos sugeridos: 
main_app.py : Archivo principal que ejecuta la aplicación Streamlit.
Limpieza de datos: donde se realiza el proceso actual de limpieza
Analisis exploratorio de datos (EDA): donde se realizan las funciones de análisis exploratorio de datos (Cualitativos, Cuantitativos y Visualizaciones)


## Limpieza de archivos estándar considerando el primer acercamiento con los datos.

Los siguientes pasos se dejan en las funciones de limpieza de los archivos:

funciones: sanitize_inventario, sanitize_transacciones
1. Campos de fechas se formatean en Datetime.
2. Diccionarios de algunos campos como Ciudades, Estados de Envío, Categorías de Productos, etc.
3. Inventario: Leads_Time_Days sin información se deja como "no definido" ya que los productos son llave para otros archivos, pero no cuentan con otros campos que permitan determinar un posible valor.
4. Inventario: El stock negativo es considerado como "error de digitación" por lo que se convierte a positivo.
5. Inventario: producto con stock nulo se determina como stock agotado = 0.
6. Transacciones: las ciudades destino que no están en el diccionario se asignan como "desconocido".

Los siguientes procesos de limpieza serán opcionales con opción desde el UI y definidas como funciones:
1. transacciones: función imputar_costo_envio_knn. Será utilizada para imputar costos de envío faltantes en el archivo de transacciones, usando KNN basado en otras columnas numéricas relevantes.
2. transacciones: Ventas con Cantidad_Vendida < 0 (negativa) podrá ser excluida del análisis si el usuario lo desea. (Pendiente por crear función)
3. transacciones: Ventas futuras (comparando con la fecha de hoy) podrán ser o exluidas, o cambiar el año al año pasado si el usuario lo desea. (Pendiente por crear función)
4. transacciones: función enriquecer_con_estado_envio_reglas. Se crea una nueva columna "Estado_Envio_Reglas" que categoriza el estado del envío basado en reglas lógicas considerando las fechas de envío, entrega y la fecha actual. (función ya creada)
5. transacciones: Por último, para terminar con las transacciones. Hay muchos SKU_ID que solo existen en transacciones pero no existen en inventario. Se nos ha pedido tomar una decisión. Nosotros definimos no "eliminarlas" por lo que el ingreso bruto es reprsentativo, pero tampoco lo podemos considerar como fraude, pues no hay un patrón en el nombramiento de los SKU para identificar alguno errado. Por lo tanto, la idea es dejar la opción al usuario de si desea incluir estos productos en el análisis o no. (Pendiente por crear función)
6. feedback: permitir la exlución de filas con Feedback_ID duplicados si el usuario lo desea. (Pendiente por crear función)


## Reporte de Health Score: 
Antes de la limpieza de los dats, se debe de realizar una tabla con filas con anomalías, como datos nulos, outliers
Después de la limpieza, se debe de generar un reporte con las filas que fueron limpiadas o eliminadas y el porcentaje que representan del total de filas.

## Archivos: 
Act2.py -> Referencia actual
dictionaries.py : Continene los diccionarios para limpieza de datos.
functions_eda.py : Funciones de limpieza y análisis exploratorio de datos (EDA).

requirements.txt : Librerías necesarias para correr el código. (se debe de actualizar)


## instrucciones finales
Del archivo de referencia Act2.py tener presente todos los análisis estadísticos y visuales que se realizan, para incluirlos en la aplicación Streamlit, ajustados a las nuevas funciones de limpieza y EDA. Documentarlos muy bien para luego hacer ajustes de forma más natural.