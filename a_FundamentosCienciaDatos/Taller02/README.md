# 📊 Data Quality & EDA Dashboard

Este dashboard es una herramienta integral diseñada para el análisis de calidad de datos (Data Quality), análisis exploratorio de datos (EDA) y monitoreo de métricas de negocio basadas en tres fuentes principales: Inventario, Transacciones y Feedback (NPS).

---

## 🛠️ Guía de Uso

### 1. Carga de Datos y Configuración (Sidebar)
Para iniciar el análisis, debe cargar los tres archivos CSV requeridos en el panel lateral:
- **Cargar Inventario**: Datos sobre SKUs, stock y costos.
- **Cargar Transacciones**: Historial de ventas y entregas.
- **Cargar Feedback**: Encuestas de satisfacción y NPS.

**Opciones de Limpieza Global:**
- **Excluir Outliers**: Detecta y elimina automáticamente filas con valores atípicos en cualquier columna numérica mediante el método IQR.
- **Excluir Filas con Nulos**: Elimina registros que contengan valores faltantes en cualquiera de los datasets.
- **Ventas Futuras**: Permite elegir entre mantener, corregir (restar 1 año) o excluir registros con fechas posteriores a hoy.
- **Normalizar Estado_Envio**: Estandariza los estados de envío según el diccionario del sistema.

---

### 2. Exploración de Pestañas

#### 📈 Pestaña 1: EDA General
Muestra un análisis detallado de cada dataset (Inventario, Transacciones, Feedback) dividido en:
- **Estadísticas Descriptivas**: Resumen cuantitativo y cualitativo.
- **Visualizaciones de Distribución**: Boxplots individuales para cada variable numérica para identificar la dispersión y outliers.
- **Gráficos de Frecuencia**: Distribución de categorías principales (Categorías, Canales de Venta, Grupos NPS).

#### 📦 Pestaña 2: Salud Inventario
- **Health Score**: Métrica porcentual de la integridad de los datos de inventario.
- **Procesos de Limpieza**: Detalle de cuántas filas fueron afectadas por normalización de texto, corrección de stock negativo y mapeo de bodegas.

#### 💸 Pestaña 3: Salud Transacciones
- **Análisis de Cantidades Negativas**: Identificación y visualización de registros con ventas negativas.
- **SKUs Fantasma**: Métricas sobre productos vendidos que no existen en el inventario cargado, incluyendo el impacto económico.
- **Imputación KNN**: Opción en el sidebar para completar costos de envío faltantes mediante el algoritmo K-Nearest Neighbors.

#### 😊 Pestaña 4: Salud NPS
- **NPS Score Profesional**: Visualización avanzada que incluye:
  - **Donut Chart**: Con el puntaje NPS final.
  - **Distribución 0-10**: Gráfico con emojis y colores (Rojo: Detractores, Amarillo: Pasivos, Verde: Promotores).
  - **Métricas Detalladas**: Porcentajes y conteos exactos por grupo.
- **Ajuste de Outliers de Edad**: Botón para imputar edades > 100 años con la mediana de los datos válidos.

#### 📊 Pestaña 5: Reporte (Dashboard)
Consolida los hallazgos más críticos del análisis:
1. **Métricas de Calidad**: Comparativa Registros Raw vs Clean y pérdida de datos.
2. **Decisiones Éticas**: Log de auditoría de todas las acciones realizadas y sección para **comentarios del analista** (estos comentarios persisten durante la sesión).
3. **Dilema del SKU Fantasma**: Análisis de impacto en ventas de productos no inventariados.
4. **Fuga de Capital**: Identificación de SKUs con margen neto negativo (pérdidas).
5. **Crisis Logística**: Heatmap de correlación entre tiempo de entrega y satisfacción NPS, identificando rutas críticas que requieren atención inmediata.

#### 🤖 Pestaña 6: Chat con Agente
Pestaña de chat con un agente de IA (Groq) que tiene acceso al contexto completo del dashboard: resúmenes de datos, métricas, NPS, SKUs fantasma, márgenes, logs y comentarios del analista.
- Configure su **API Key de Groq** en el panel lateral o en `.streamlit/secrets.toml` (variable `GROQ_API_KEY`).
- Instale el paquete: `pip install groq`.

---

### 📄 Exportación de Resultados
Al final del panel lateral (Sidebar), encontrará el botón **"Generar Log PDF"**. Esto descargará un reporte formal que incluye:
- El historial cronológico de todas las limpiezas realizadas.
- Los comentarios y justificaciones éticas ingresados por el analista en la pestaña de Reporte.

---

## 👥 Participantes
- **Andrés Felipe Velasco Hernández**
- **Juan Miguel Gómez Alzate**

**Materia:** Fundamentos De la Ciencia de Datos - Universidad EAFIT