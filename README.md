# TubeCast
Proyecto de predicción para youtube

# TubePulse: Predicción de Viralidad y Análisis de Engagement en YouTube

## 1. Descripción del Proyecto
Este proyecto busca modelar y predecir la trayectoria de engagement y probabilidad de viralidad de videos en YouTube. El sistema combina métricas de rendimiento temprano (vistas, likes, comentarios) con procesamiento de lenguaje natural (NLP) aplicado a los comentarios de la audiencia para capturar señales cualitativas de recepción y aceleración de contenido.

## 2. Objetivos Principales
* **Extracción de Datos:** Recolección dinámica de metadatos de videos y muestras de comentarios mediante la YouTube Data API v3.
* **Análisis de Sentimiento:** Procesamiento de texto en comentarios para cuantificar la polaridad y el tono de la respuesta del público frente al contenido.
* **Pronóstico de Métricas:** Modelado predictivo de series de tiempo para estimar la acumulación de visualizaciones en las primeras horas/días posteriores a la publicación.
* **Clasificación de Viralidad:** Definición y predicción de umbrales de aceleración de vistas para catalogar contenido con alto potencial viral.

## 3. Fuentes de Datos y Variables
* **Fuente Primaria:** YouTube Data API v3.
* **Variables Clave Consideradas:**
  * Metadatos: Título, etiquetas, categoría, duración, fecha y hora de publicación.
  * Métricas de Desempeño: Conteo de vistas (`viewCount`), likes (`likeCount`), total de comentarios (`commentCount`).
  * Interacciones Textuales: Hilos de comentarios y respuestas para análisis semántico y cálculo de índices de sentimiento.

## 4. Estructura del Repositorio
```text
├── data/              # Almacenamiento local de datos (ignorado en Git)
├── notebooks/         # Exploración preliminar y prototipos
├── src/               # Scripts modulares de extracción y preprocesamiento
├── .gitignore         # Exclusión de archivos temporales y secretos
└── README.md          # Documentación del proyecto
