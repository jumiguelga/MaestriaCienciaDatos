# Taller 01 – Análisis Exploratorio de Datos (EDA)
## Gaming Hours vs Academic and Work Performance

Este repositorio presenta el desarrollo del Taller 01 de Análisis Exploratorio de Datos (EDA), realizado en el marco de la Maestría en Ciencias de los Datos y Analítica de la Universidad EAFIT. El trabajo se estructura a partir de la metodología CRISP-DM y tiene como propósito analizar la relación entre el uso de videojuegos y el desempeño académico o laboral, desde una perspectiva orientada a la toma de decisiones institucionales.

---

## Objetivo del proyecto

Analizar la relación entre los hábitos de juego (tipo de videojuego, horas y momentos de uso), variables asociadas al bienestar (sueño, estrés y nivel de concentración) y el desempeño académico o laboral, con el fin de generar recomendaciones informadas para una institución educativa sobre el uso responsable de videojuegos dentro de sus instalaciones.

---

## Metodología: CRISP-DM

El proyecto sigue las fases de la metodología CRISP-DM, aplicadas de la siguiente manera:

1. **Business Understanding**  
   Se formulan tres preguntas de negocio desde la perspectiva de una universidad:
   - ¿Qué tipo de videojuego se asocia con un mejor desempeño académico o laboral?
   - ¿Qué recomendaciones pueden formularse respecto al uso de videojuegos dentro de las instalaciones para no afectar el rendimiento?
   - ¿Qué variables del perfil de estudiantes o trabajadores podrían asociarse con un impacto negativo en el desempeño?

2. **Data Understanding**  
   Exploración inicial del conjunto de datos, incluyendo la descripción de las variables, el análisis de distribuciones, la identificación de valores atípicos y la exploración de posibles relaciones entre variables.

3. **Data Preparation**  
   Procesos de limpieza, selección de variables relevantes, generación de segmentaciones por niveles de desempeño e impacto, y preparación de subconjuntos para el análisis exploratorio.

4. **Modeling**  
   En este taller no se desarrollan modelos predictivos. El análisis se centra en técnicas exploratorias (visualización, segmentación y análisis de correlaciones) con el objetivo de identificar patrones relevantes.

5. **Evaluation**  
   Interpretación de los resultados del EDA para responder de manera explícita a las preguntas de negocio y traducir los hallazgos en recomendaciones prácticas para la universidad.

6. **Deployment**  
   Discusión sobre el uso institucional de los resultados como insumo para la formulación de políticas de bienestar, uso responsable de videojuegos y estrategias de acompañamiento preventivo.

---

## Dataset

- **Fuente:** Kaggle – Gaming Hours vs Academic and Work Performance
- **Número de observaciones:** 1000 registros  
- **Variables principales:**
  - Demográficas: `Age`, `Gender`, `Occupation`
  - Hábitos de juego: `Game_Type`, `Daily_Gaming_Hours`, `Weekly_Gaming_Hours`, `Primary_Gaming_Time`
  - Bienestar: `Sleep_Hours`, `Stress_Level`, `Focus_Level`
  - Desempeño: `Academic_or_Work_Score`, `Productivity_Level`, `Performance_Impact`

---

## Estructura del repositorio

```text
├── Taller01_EDA.ipynb        # Notebook principal con el análisis completo
├── Gaming_Hours_vs_Performance_1000_Rows.csv # Base de datos usada para el análisis
├── README.md                # Descripción del proyecto
