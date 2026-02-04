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

## 2. Exploración de pestañas

### Pestaña Auditoría

En esta pestaña se realiza el análisis exploratorio clásico de cada dataset:

- Inventario: estadísticas descriptivas, boxplots por variable numérica, distribución por categoría.
- Transacciones: estadísticas descriptivas, boxplots, distribución por canal de venta.
- Feedback / NPS: estadísticas descriptivas, boxplots, distribución por grupos NPS.
- Health Score por dataset (Inventario, Transacciones, Feedback) y métricas de nulos y filas filtradas.
- Reporte de procesos de limpieza por dataset.
- Comparativo “Registros originales vs limpios vs excluidos” con gráfico de barras apiladas.
- Sección de “Decisiones éticas de limpieza” con:
  - Log de acciones realizadas.
  - Textarea para comentarios del analista (persisten durante la sesión).
  - Tabla de resumen de decisiones de imputación y limpieza.
  - Posible análisis antes/después para EdadCliente si se ajustan outliers.

### Pestaña Operaciones

Enfocada en riesgos operativos y “dolores” del negocio:

- Análisis de SKUs fantasma:
  - Métricas agregadas (SKUs únicos fantasma, % de ventas fantasma, transacciones afectadas).
  - Top 10 SKUs fantasma por frecuencia y gráfico de barras.
  - Tabla detallada por SKU (transacciones, cantidades, ingreso total).
- Análisis de cantidades negativas:
  - Tabla de ejemplos de transacciones con cantidad negativa.
  - Histograma de distribución de cantidades negativas.
- Análisis por bodega:
  - Tabla con días promedio desde última revisión, tasa de tickets, NPS promedio, número de transacciones.
  - Scatter de riesgo: días desde última revisión vs tasa de tickets, tamaño por transacciones, color por NPS, con zonas de riesgo y umbrales.
  - Listado de bodegas críticas en “zona de alto riesgo”.


### Pestaña Cliente

Foco en experiencia de cliente y NPS:

- Cálculo del NPS (detractores, pasivos, promotores) a partir de SatisfacciónNPS.
- Gráfico tipo “donut” con el score NPS general.
- Métricas de conteo y porcentaje para cada grupo.
- Visualizaciones adicionales relacionadas con feedback, tickets de soporte y su impacto en cliente:
  - Comparativo “Tickets Abiertos: Sí vs No” con gráfico de barras y métricas.
  - Tabla con los conteos de tickets si la columna está disponible.
- (Opcional según datos cargados) análisis de outliers en EdadCliente y su relación con NPS.


### Pestaña Insights de IA

Pestaña de interacción con el agente de IA (Groq):

- Requiere configurar `GROQAPIKEY` en `.streamlit/secrets.toml` o en el panel lateral, además de tener instalado `groq`.
- El agente tiene acceso a:
  - Resúmenes de inventario, transacciones, feedback.
  - Health scores, NPS, métricas de SKUs fantasma, margen negativo.
  - Logs de limpieza y comentarios del analista.
- Permite:
  - Chatear sobre los datos cargados en el dashboard (las preguntas se restringen al contexto del proyecto).
  - Generar entre 5 y 10 insights y recomendaciones accionables con el botón **“Generar Insights con IA”**.
- El resultado de insights se guarda en sesión y se muestra con marca de tiempo.


## Requisitos

- Python 3.10+ (recomendado).
- Paquetes (ver `requirements.txt`):
  - streamlit
  - pandas, numpy
  - matplotlib, seaborn, plotly
  - scikit-learn
  - reportlab
  - groq (opcional, solo para la pestaña “Insights de IA”)

Instalación rápida:

```bash
pip install -r requirements.txt
```


```markdown
### Estructura esperada de datos

El dashboard espera tres archivos CSV:

- **Inventario**:
  - `SKUID`, `StockActual`, `CostoUnitarioUSD`, `Categoria`, `BodegaOrigen`, `LeadTimeDias`, `UltimaRevision`
- **Transacciones**:
  - `TransaccionID`, `SKUID`, `FechaVenta`, `CantidadVendida`, `PrecioVentaFinal`, `CiudadDestino`, `CanalVenta`, `EstadoEnvio`
- **Feedback**:
  - `FeedbackID`, `TransaccionID`, `SatisfaccionNPS`, `ComentarioTexto`, `RecomiendaMarca`, `TicketSoporteAbierto`
```

## Reporte de Hallazgos
Para ver el reporte de hallazgos puedes consultar el siguiente documento: [Reporte de Hallazgos](./ReporteDeHallazgos.md)!

---

## 👥 Participantes
- **Andrés Felipe Velasco Hernández**
- **Juan Miguel Gómez Alzate**

**Materia:** Fundamentos De la Ciencia de Datos - Universidad EAFIT
