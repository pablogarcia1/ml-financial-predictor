# Financial Machine Learning Predictor  
### Desarrollo, Validación y Hallazgos Estructurales

---

## 1. Propósito del Sistema

Este proyecto tiene como objetivo construir un pipeline algorítmico capaz de predecir movimientos en series de tiempo financieras, priorizando la **gestión asimétrica del riesgo** por encima de la precisión estadística tradicional.

A lo largo de múltiples iteraciones, el sistema evolucionó desde un clasificador binario teórico hacia una **arquitectura defensiva con enfoque institucional**, donde la robustez operativa supera la optimización superficial de métricas.

---

## 2. Arquitectura de Datos y Entrenamiento

Uno de los hallazgos más relevantes surgió en la etapa de diseño de datos.

Para mitigar:
- Overfitting  
- Data Leakage  

se implementó un enfoque de **Panel Data** utilizando activos de alta capitalización:

- AAPL  
- MSFT  
- GOOGL  
- JPM  

El modelo (**XGBoost**) fue entrenado exclusivamente con estos activos, mientras que el **SPY** se reservó como conjunto de prueba.

### Implicación clave

Este diseño fuerza al modelo a aprender la **estructura subyacente del mercado (market physics)** en lugar de memorizar patrones específicos de un solo activo.

El resultado es una validación real de **capacidad de generalización macroeconómica**.

---

## 3. Probabilidades Financieras y el Problema del Cash Drag

Durante el backtesting, se identificó una limitación estructural del dominio financiero:

> La relación señal-ruido es extremadamente baja.

Esto provoca que las predicciones del modelo converjan naturalmente hacia **0.50**.

### Problema inicial

El uso de umbrales tradicionales (ej. 0.60):
- Inhabilitaba la toma de decisiones
- Generaba un exceso de capital inactivo (**Cash Drag**)

### Insight crítico

En finanzas cuantitativas:

> Probabilidades entre **0.51 y 0.53** representan una ventaja estadística real.

### Solución aplicada

Se redefinió el criterio de ejecución:

```python
buy_threshold = 0.50
```


## 4. Resultados Empíricos

Al habilitar la ejecución bajo este nuevo marco probabilístico, el sistema mostró resultados sólidos:

### Métricas clave

- **Win Rate:** 64.47% (76 operaciones)  
- **Retorno Acumulado:** 70.54%  
- **Benchmark (SPY):** 72.99%  

### Métricas de eficiencia

- **Sharpe Ratio:** 1.4188  
- **Sharpe Pasivo:** 0.1437  
- **Profit Factor:** 2.0458  

### Riesgo

- **Maximum Drawdown:** -12.17%  

### Tabla resumen consolidada

| Categoría   | Métrica                | Valor    | Benchmark |
|-------------|------------------------|----------|-----------|
| Operación   | Win Rate               | 64.47%   | —         |
| Operación   | Número de Operaciones  | 76       | —         |
| Rendimiento | Retorno Acumulado      | 70.54%   | 72.99%    |
| Eficiencia  | Sharpe Ratio           | 1.4188   | 0.1437    |
| Eficiencia  | Profit Factor          | 2.0458   | —         |
| Riesgo      | Maximum Drawdown       | -12.17%  | —         |