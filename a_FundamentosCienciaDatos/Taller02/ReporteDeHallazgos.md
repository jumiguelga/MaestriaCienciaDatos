# Título del Informe de Auditoría / Investigación
## Subtítulo opcional
**Fecha:** AAAA-MM-DD  
**Autor(es):** Nombre Apellido, Rol  
**Versión:** 1.0

---

## 1. Resumen ejecutivo

- Objetivo principal de la auditoría/investigación.
- Alcance y periodo cubierto.
- Hallazgos clave (en bullets).
- Conclusión general y recomendaciones de alto nivel.

---

## 2. Contexto y alcance

### 2.1 Contexto

- Descripción general de la organización/proceso/sistema auditado.
- Antecedentes relevantes (auditorías previas, incidentes, cambios normativos, etc.).

### 2.2 Alcance

- Áreas, procesos, sistemas o datasets incluidos.
- Exclusiones explícitas.
- Periodo de tiempo considerado.

### 2.3 Objetivos de la auditoría / investigación

- Objetivo 1
- Objetivo 2
- Objetivo 3

---

## 3. Criterios, estándares y normativa

- Normas o marcos de referencia (por ejemplo: ISO, NIST, políticas internas).
- Criterios de evaluación utilizados.
- Definición de niveles de severidad (Crítico, Alto, Medio, Bajo).

Ejemplo de tabla de severidad:

| Severidad | Descripción breve                                     |
|----------:|--------------------------------------------------------|
| Crítico   | Impacto muy alto, requiere acción inmediata.          |
| Alto      | Impacto alto, acción prioritaria en el corto plazo.   |
| Medio     | Impacto moderado, plan de acción en el mediano plazo. |
| Bajo      | Impacto menor, mejora recomendable.                    |

---

## 4. Metodología

### 4.1 Enfoque general

- Tipo de auditoría/investigación (interna, externa, forense, cumplimiento, etc.).
- Enfoque cuantitativo, cualitativo o mixto.

### 4.2 Fuentes de información

- Entrevistas (roles, áreas).
- Documentación revisada (manuales, políticas, procedimientos).
- Sistemas o bases de datos analizados.

### 4.3 Técnicas y herramientas

- Técnicas usadas (muestreo, análisis estadístico, minería de datos, revisión de logs, etc.).
- Herramientas (Python, R, SQL, BI, otras).

---

## 5. Hallazgos

> Cada hallazgo debería ser independiente, numerado y trazable a un objetivo o criterio.

### 5.1 Hallazgo 1 – Título descriptivo

**Descripción:**  
Texto claro describiendo qué se encontró, dónde y cómo se detectó.

**Evidencia:**
- Evidencia 1 (por ejemplo, query, log, captura de pantalla, referencia de documento).
- Evidencia 2.

**Riesgo / Impacto:**
- Impacto en el negocio, cumplimiento, seguridad, calidad de datos, etc.
- Breve análisis de probabilidad y severidad.

**Causa raíz:**
- Explicación sintética de la causa principal.

**Recomendación:**
- Acción propuesta, responsable sugerido y prioridad.

---

### 5.2 Hallazgo 2 – Título descriptivo

*(Repetir estructura de la sección 5.1 para cada hallazgo)*

---

## 6. Resultados analíticos (si aplica a datos)

### 6.1 Descripción de los datos

- Origen de los datos.
- Rango de fechas.
- Número de registros, variables, principales transformaciones.

Tabla de resumen de calidad de datos:

| Variable       | Tipo      | % Nulos | Notas de calidad                     |
|----------------|----------|--------:|--------------------------------------|
| id_cliente     | entero   | 0.0     | Único por registro.                  |
| fecha_evento   | fecha    | 0.2     | Algunas fechas faltantes.            |
| monto          | numérico | 1.3     | Outliers detectados en el percentil 99. |

### 6.2 Análisis exploratorio

- Principales patrones observados.
- Anomalías o desviaciones relevantes.

### 6.3 Modelos o pruebas aplicadas

- Tipo de modelo o prueba (por ejemplo, regresión, test estadístico).
- Métricas clave (accuracy, precision, recall, AUC, etc.).

Ejemplo de tabla de métricas:

| Modelo         | Métrica | Valor |
|----------------|--------:|------:|
| Regresión logística | AUC     | 0.87 |
| Árbol de decisión   | AUC     | 0.81 |

---

## 7. Conclusiones

- Resumen de los hallazgos más relevantes.
- Evaluación global del riesgo.
- Grado de cumplimiento con políticas/normas.

---

## 8. Recomendaciones

> Prioriza (por ejemplo, usando la misma escala de severidad definida antes).

- Recomendación 1 (Severidad: Alta, Responsable sugerido, Plazo).
- Recomendación 2 (Severidad: Media, Responsable sugerido, Plazo).
- Recomendación 3 (Severidad: Baja, Responsable sugerido, Plazo).

---

## 9. Plan de acción (opcional)

| ID | Hallazgo relacionado | Acción propuesta                         | Responsable | Fecha objetivo | Estado   |
|----|----------------------|-------------------------------------------|-------------|----------------|----------|
| 1  | 5.1                  | Implementar controles adicionales de X.  | Área A      | AAAA-MM-DD     | Pendiente|
| 2  | 5.2                  | Actualizar política Y.                   | Área B      | AAAA-MM-DD     | En curso |

---

## 10. Limitaciones del trabajo

- Limitaciones de datos (calidad, disponibilidad, acceso).
- Limitaciones metodológicas.
- Supuestos adoptados.

---

## 11. Anexos

### 11.1 Evidencias adicionales

Describe aquí referencias a:
- Capturas de pantalla.
- Consultas SQL.
- Fragmentos de código relevantes.

### 11.2 Código o notebooks

- Enlace al repositorio (GitHub, GitLab, etc.).
- Rutas de notebooks o scripts clave.

---

## Formato para imágenes

### Ejemplo 1: Imagen simple en línea

Texto de contexto previo.

![Descripción de la imagen](ruta/o/url/de/la_imagen.png "Título opcional de la imagen")

Texto de contexto posterior.

### Ejemplo 2: Sección de visualizaciones

## Visualizaciones clave

1. Distribución de variable X  
   ![Distribución de X](imagenes/distribucion_x.png "Distribución de X")

2. Matriz de correlación  
   ![Matriz de correlación](imagenes/matriz_correlacion.png "Matriz de correlación")

3. Flujo del proceso auditado  
   ![Flujo del proceso](imagenes/flujo_proceso.png "Flujo del proceso")
