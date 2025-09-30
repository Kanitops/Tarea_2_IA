# Tarea 2 - Inteligencia Artificial
Autor: Aníbal Rivas Henríquez (Kanitops)

Este repositorio contiene el desarrollo de la Tarea 2 del curso de Inteligencia Artificial (segundo semestre 2025). El objetivo principal es comparar el rendimiento de dos modelos de clasificación supervisada aplicados a un conjunto de datos de productos.

## Modelos Implementados
- Árbol de Decisión (`DecisionTreeClassifier`)
- Perceptrón Multicapa (`MLPClassifier`)

## Estructura del Repositorio

### Notebooks de Análisis
- `Tarea2_modelos.ipynb`: Notebook principal con el análisis completo y resultados finales.
- `Alternative_Version_Tarea_2.ipynb`: Versión alternativa del análisis con algunas variaciones en el preprocesamiento.
- `Tarea2_modelos _Mala_Idea.ipynb`: Versión experimental (no recomendada para uso).

### Datos
- `productos.csv`: Dataset original proporcionado para la tarea.

### Documentación
- `Reporte_Tarea2_IA.md`: Informe detallado del análisis en formato Markdown.
- `Reporte_Tarea2_IA.pdf`: Versión PDF del informe.
- `comparacion_notebooks.md`: Análisis comparativo de las diferentes versiones de notebooks.
- `comparacion_notebooks.pdf`: Versión PDF de la comparación de notebooks.

## Guía de Navegación

1. **Para Revisión Rápida**:
   - Revisar `Reporte_Tarea2_IA.pdf` para un resumen completo del análisis y resultados.
   - Consultar `comparacion_notebooks.pdf` para entender las diferencias entre las versiones de los notebooks.

2. **Para Análisis Detallado**:
   - Utilizar `Tarea2_modelos.ipynb` que contiene el análisis completo y funcional.
   - Los otros notebooks (`Alternative_Version` y `Mala_Idea`) están disponibles para referencia pero no se recomienda su uso directo.

3. **Para Replicar el Análisis**:
   ```bash
   # 1. Clonar el repositorio
   git clone https://github.com/Kanitops/Tarea_2_IA.git
   cd Tarea_2_IA

   # 2. Crear un ambiente virtual (recomendado)
   python -m venv venv
   source venv/bin/activate  # En Unix/macOS
   # o
   .\venv\Scripts\activate  # En Windows

   # 3. Instalar dependencias
   pip install jupyter pandas numpy scikit-learn matplotlib seaborn
   ```

## Dependencias Principales
- `pandas`: Manipulación y análisis de datos
- `scikit-learn`: Implementación de modelos de Machine Learning
- `matplotlib` y `seaborn`: Visualización de datos y resultados
- `numpy`: Operaciones numéricas y manipulación de arrays

## Métricas de Evaluación Implementadas
- Exactitud (Accuracy)
- F1-score (macro y weighted)
- Precisión y Recall por clase
- Matrices de confusión
- Curvas ROC y AUC
- Tiempos de entrenamiento e inferencia
- Curvas de aprendizaje

## Resultados y Conclusiones
Los resultados detallados y las conclusiones se pueden encontrar en:
1. El informe principal (`Reporte_Tarea2_IA.pdf`)
2. El notebook principal (`Tarea2_modelos.ipynb`)
3. El análisis comparativo (`comparacion_notebooks.pdf`)

## Notas Adicionales
- Los notebooks están diseñados para ejecutarse secuencialmente.
- Se recomienda usar el ambiente virtual sugerido para evitar conflictos de dependencias.
- Todos los gráficos y visualizaciones se generan automáticamente durante la ejecución.

## Contacto
Para cualquier duda o consulta sobre el análisis, contactar a través de:
- GitHub: [@Kanitops](https://github.com/Kanitops)
