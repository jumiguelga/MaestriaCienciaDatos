# Análisis DSS para TechLogistics S.A.S
## Comparativo de sistemas ECR y CRM
**Fecha:** 2026-02-02  
**Autor(es):** 
- Andrés Felipe Velasco Hernández
- Juan Miguel Gómez Alzate  

**Versión:** 1.0

---

## 1. Resumen ejecutivo

Luego de evaluar la información presentada por TechLogistics S.A.S, que fue extraída de sus sistemas de información, y de compartir sus preocupaciones, se efectuó un análisis exploratorio que permitió conocer parte de la problemática, entre ello como lo mencionó la junta, acerca de la falta de comunicación entre sus sistemas de gestión interna.

Dentro de los aspectos relevantes tenemos fugas de ventas que no se encuentran relacionadas con los inventarios de la compañía, así como productos sin clasificar en sus respectivas categorías, lo cual impide realizar un seguimiento adecuado al stock que es administrado en cada una de las bodegas, que por cierto, presentan un nivel crítico de abandono al seguimiento del mismo stock.

Este análisis nos ha permitido otorgar algunas recomendaciones bases con las que la compañía puede iniciar un plan de mejora continua en pro de mejorar y optimizar sus sistemas de información, lo que les permitirá "recobrar la vista" respecto a como se encuentra el negocio. 

---

## 2. Contexto y alcance

### 2.1 Contexto

- TechLogistics S.A.S, empresa de retail tecnológico.
- Se detectaron inconsistencias entre sus sistemas de información.
   - Erosión en margen de beneficios.
   - Caída drástica en la lealtad de los clientes.
   - Sistemas que no se conversan entre sí.


### 2.3 Objetivos de la auditoría / investigación

- Fuga de Capital y Rentabilidad: Localizar los SKUs que se están vendiendo con
margen negativo y determinar si corresponden a una pérdida aceptable o si son interpretados como fallas del sistema.
- Crisis Logística y Cuellos de Botella: Determinar correlación etre ciudades de destino y bbodegas de origen, y los tiempos de entrega.
- Análisis de la Venta Invisible: Cuantificar transacciones a productos inexistentes y su impacto en el margen de ventas.
- Diagnóstico de Fidelidad: Evaluar el impacto del sentimiento del cliente con relación a los altos volúmenes de stock.
- Storytelling de Riesgo Operativo: Efectúe una relación entre la última revisión de stock y los tickets abiertos por transacciones.

---

## 3. Criterios y estándares
### 3.1 Limpieza base de los archivos
- Campos de fecha se ajustaron a formatos de fecha para calcularlos de forma correcta.
- Definición de diccionarios para unificar datos relevantes tales como:
   - Inventario: Bodega
   - Inventario: Categorías
   - Transacciones: Ciudades destino
   - Transacciones: Estados de envío
   - Transacciones: Canales de venta
- Se determina el campo "Lead_time_dias" que se encontraban NULL como "No definidos", ya que no es posible imputarlo por calculo estadístico y no se puede omitir del cálculo por la relación con las tablas de transacciones y feedback.
- El stock que se encontraba null o vacío se imputó como "sin existencias" (0) ya que no es posible determinar la posible existencia de inventario.
- El stock que se encontraba negativo (<0) fue convertido a positivo (>0) al determinarse como error de digitación.
- La calificación NPS queda agrupada como
   - Muy insatisfecho
   - Neutro o parcialmente satisfecho
   - Satisfecho
   - Muy satisfecho
  
- Definición de niveles de severidad (Crítico, Alto, Medio, Bajo).

| Severidad | Descripción breve                                     |
|----------:|--------------------------------------------------------|
| Crítico   | Impacto muy alto, requiere acción inmediata.          |
| Alto      | Impacto alto, acción prioritaria en el corto plazo.   |
| Medio     | Impacto moderado, plan de acción en el mediano plazo. |
| Bajo      | Impacto menor, mejora recomendable.                    |

---

## 4. Metodología

### 4.1 Enfoque general

- Auditoría de Datos Externa.
- Enfoque mixto de calidad de los datos.

### 4.2 Fuentes de información

- Datasets exportados de los tres sistemas de información.
   - Inventario: contiene la información de los productos, su stock, fecha de actualización, costos unitarios, y bodegas.
   - Transacciones: acá se encuentran los detalles de ventas como canales comerciales y ciudades de destino, estados de envíos, fechas de transacción, entre otros.
   - Feedback: se ha recopilado el sentimiento del cliente. 
- Sistemas o bases de datos analizados.

### 4.3 Técnicas y herramientas

- Técnicas usadas: análisis estadístico y minería de datos
- Herramientas: Python y Streamlit.

---

## 5. Hallazgos

### 5.1 Hallazgo 1 – Ventas con cantidades en negativo

**Descripción:**  
El dataset de _Transacciones_ permite ver un patrón no descrito en la definición de los datasets, y es que se cuentan con al menos 100 transacciones que tiene como "Cantidad Vendida" el total de -5.

**Evidencia:**
- Evidencia 1: Captura de pantalla con gráfico de conteo
  <img width="1427" height="476" alt="image" src="https://github.com/user-attachments/assets/f3228cc5-155f-4475-adcf-914e5c6fd2f6" />

**Riesgo / Impacto:**
- Se evaluó el comportamiento de las demás columnas, pero solo esta cantidad tenía el patrón de contar con el -5 como valor repetido; las demás columnas tenían valores sin patrones claros, por lo que se puede inferir que este campo está siendo utilizado para reportar alguna novedad en la plataforma.
- Al ser un valor con patrón de repetición, no es posible realizar alguna imputación, por lo que estas 100 transacciones deben de ser excluidas el análisis final.

**Causa raíz:**
- Pendiente por investigación con el cliente. 

**Recomendación:**
| Severidad | Descripción                                     |
|----------:|--------------------------------------------------------|
| Alto      | De ser una novedad con la transacción, debería agregarse un campo que permita e análisis de dicha novedad, como por ejemplo "transacción eliminada" u ptro, para así evitar datos corruptos en la columna de cantidad vendida |

---

### 5.2 Hallazgo 2 – Ventas fantasmas

**Descripción:**  
La relación entre los datasets de _Transacciones_ e _Inventario_ muestra que hay un total de 1.728 transacciones que corresponden a 474 SKU_ID únicos que no se encuentran registrados en inventario. 

**Evidencia:**
- Evidencia 1: Captura de pantalla con gráfico de conteo

<img width="1454" height="617" alt="image" src="https://github.com/user-attachments/assets/8bce17e9-1f7a-4c41-a39a-845d9ffba313" />

**Riesgo / Impacto:**
- Las ventas efectuadas a los productos inexistentes en la base de inventario reprsenta más del 16% de los ingresos, para un total de US $12'976,848.54. Estas ventas no tienen una bodega origen asociada, por lo que no es posible llevarles un control claro de sus movimientos.
- Debido al monto y porcentaje de participación, estos valores no son excluídos por completos del análisis, ya que representan una parte importante de los ingresos y su existencia se debe de evaluar.

**Causa raíz:**
- Mala gestión de inventarios y falta de relación en la base de datos. No debería de ser posible insertar transacciones a productos inexistentes.

**Recomendación:**
| Severidad | Descripción                                     |
|----------:|--------------------------------------------------------|
| Crítico | La relación del stock se debe de resolver cuanto antes para determinar si los productos deben de ser registrados  sanear la falta de información |

### 5.3 Hallazgo 3 – Demoras en tiempos de entrega vs NPS

**Descripción:**  
NPS afectado por baja percepción debido a tiempos de entrega.

**Evidencia:**
- Evidencia 1: Captura de pantalla con gráfico

<img width="1460" height="1271" alt="image" src="https://github.com/user-attachments/assets/d6f7acff-a7bf-48f6-b3b6-20ec454bf530" />


**Riesgo / Impacto:**
- Las ciudades de destino que más afectación presentan en tiempos de entrega son Bogotá, Cali, y Bucaramanga

**Causa raíz:**
- Por investigar

**Recomendación:**
| Severidad | Descripción                                     |
|----------:|--------------------------------------------------------|
| Alto | Se debe de iniciar un plan de acción y seguimiento para las entregas que se realizan en estas ciudades desde las diferentes bodegas |

---

## 6. Resultados analíticos (si aplica a datos)

### 6.1 Descripción de los datos

- Origen de los datos: los datos son exportes recibidos de 3 fuentes de información diferentes, entre los cuales tenemos un sistema de gestión de inventario, uno de gestión de transacciones y uno de recibimiento de feedback.
- Rango de fechas: 2024 - 2025
- Número de registros, variables, principales transformaciones.

Tabla de resumen de calidad de datos:

<img width="1483" height="212" alt="image" src="https://github.com/user-attachments/assets/6139504e-13f2-424e-bbf6-b9ff197d87bd" />


### 6.2 Análisis exploratorio

- Principales patrones observados:
  - Los canales de gestión digital no están alineados, siendo el principal impacto de fuga de ingresos.
  - Las ciudades de Cali y Bucaramanga son las principales afectadas cuando el despacho se realiza desde las bodegas de norte, occidente y sur.
  - No se evidencia un control de stock regular en ninguna de las bodegas

---

## 7. Conclusiones

- Resumen de los hallazgos más relevantes.
- Evaluación global del riesgo.

---

## 8. Recomendaciones

| Recomendación # | Severidad | Descripción breve                                                                                                                                                           |
|----------------:|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1               | Crítico   | Preparar una revisión de inventario para actualizar el estado del stock y los productos faltantes.                                                                          |
| 2               | Crítico   | Ajustar los sistemas de información para reducir el error del factor humano al momento de insertar la data, limitando los formularios con valores por defecto.            |
| 3               | Alto      | Efectuar un análisis de la calidad de la data con el fin de ajustar e imputar de forma correcta los valores faltantes o desconocidos, como categorías, ciudades de destino, estados de entrega, entre otros. |
| 4               | Alto      | Ajustar los lead times de acuerdo con la bodega de origen y la ciudad destino.                                                                                              |
| 5               | Alto      | Re-distribuir las bodegas para que se focalicen en aquellas ciudades donde sus operaciones tienen una correlación positiva.                                                 |
| 6               | Alto      | Priorizar la gestión de incidentes (tickets) abiertos y efectuar una campaña de fidelización con los clientes que vienen siendo detractores del servicio.                  |
