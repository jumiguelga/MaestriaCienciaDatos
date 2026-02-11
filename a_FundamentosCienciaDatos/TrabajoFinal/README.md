## Descripción del Problema


## Instalación
Si bien la aplicación cuenta con un demo desplegado en Streamlit ambiente nube, también cuenta con la opción de ser instalada localmente.

Estos son los pasos sugeridos:

1. Clona o descarga el proyecto y entra en la carpeta del TrabajoFinal.
2. (Opcional) Crea un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # En Windows: .venv\Scripts\activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

## Link al Despliegue
Para encontrar el demo de la aplicación puede utilizar el siguiente enlace de Streamlit el cual ha sido previamente configurado para su revisión: 

https://eda-general-eafit.streamlit.app/

## Créditos:

### Autores: 
- Andrés Felipe Velasco Hernández
- Juan Miguel Gómez Alzate

### Institución:
Universidad EAFIT, sede Medellín. 

### Fuentes de datos:
1. Dataset obtenido de: https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download
2. Guia de marca Eafit obtenida de: https://www.slideshare.net/slideshow/1-manual-identidad-visual-eafit-ejemplo/53089709#1

### Dataset ejemplos
Hemos preparado un dataset específico para validar que el dinamismo de la aplicación se cumpliera desde la carga de archivos, hasta el análisis del agente Groq.

El dataset *_clean es el original descargado desde Kaggle, y el *_dirty es el cual fue previamente "dañado" a propósito para cumplir los fines de la limpieza.

```txt
/data/
- WA_Fn-UseC_-Telco-Customer-Churn_clean (json y csv)
- WA_Fn-UseC_-Telco-Customer-Churn_dirty (json y csv)
```