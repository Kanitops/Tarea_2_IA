# Comparación de Notebooks

Este documento compara dos notebooks de Jupyter:
1. `Tarea2_modelos.ipynb` (Notebook Original)
2. `Tarea2_modelos _Mala_Idea.ipynb` (Notebook Alternativo)

## 1. Estructura General

Ambos notebooks tienen una estructura idéntica en términos de número de celdas y tipos:
- Total de celdas: 41 celdas
- Celdas de código: 24 celdas
- Celdas de markdown: 17 celdas

## 2. Estado de Ejecución

### Notebook Original (`Tarea2_modelos.ipynb`)
- **Estado**: Completamente ejecutado
- **Rango de ejecución**: Celdas ejecutadas del conteo 23 al 44
- Todas las celdas de código han sido ejecutadas exitosamente

### Notebook Alternativo (`Tarea2_modelos _Mala_Idea.ipynb`)
- **Estado**: No ejecutado
- Ninguna celda ha sido ejecutada, aunque mantiene los outputs de una ejecución anterior

## 3. Diferencias Principales

### 3.1 Outputs y Resultados
- El notebook original muestra resultados de ejecución recientes y consistentes
- El notebook alternativo mantiene outputs antiguos pero no tiene estado de ejecución actual

### 3.2 Variables en Memoria
El notebook original tiene un estado completo de variables en memoria, incluyendo:
- Modelos entrenados (DecisionTreeClassifier, MLPClassifier)
- Datos procesados (X, y, X_train, X_test, etc.)
- Métricas calculadas (accuracy, f1_score, etc.)
- Visualizaciones generadas

### 3.3 Error Notable
En el notebook alternativo, la celda 33 (líneas 382-398) muestra un error en los outputs (stderr), lo que sugiere que hubo problemas en la ejecución anterior.

## 4. Similitudes

1. **Estructura de Importaciones**: Ambos notebooks utilizan las mismas bibliotecas y dependencias
2. **Flujo de Trabajo**: El flujo de procesamiento y análisis es idéntico
3. **Visualizaciones**: Los tipos de gráficos y análisis visuales son los mismos

## 5. Recomendaciones

1. Se recomienda usar el notebook original (`Tarea2_modelos.ipynb`) ya que:
   - Está completamente ejecutado y funcional
   - No muestra errores en la ejecución
   - Mantiene un estado consistente de variables y resultados

2. El notebook alternativo podría servir como respaldo, pero necesitaría:
   - Una ejecución completa para verificar su funcionamiento
   - Corrección del error en la celda 33
   - Validación de los resultados

## 6. Conclusión

El notebook `Tarea2_modelos.ipynb` representa la versión más confiable y funcional del análisis. Muestra una ejecución completa y exitosa de todo el flujo de trabajo, mientras que la versión alternativa parece ser un intento anterior o una variación que no se completó exitosamente.