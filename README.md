# ML Financial Predictor v3.0  
### Estudio crítico de Machine Learning en mercados financieros

Este proyecto implementa un pipeline completo de Machine Learning para predecir la dirección del ETF **SPY (S&P 500)** a un horizonte de 10 días, utilizando datos tipo **Panel (AAPL, MSFT, GOOGL, JPM, SPY)**. 

> **Definición del Target ($y_t$):** > El problema se formula como una clasificación binaria, donde la variable objetivo evalúa si el precio de cierre ($P$) será mayor en 10 días hábiles respecto al día actual ($t$):
> $$y_t = \begin{cases} 1 & \text{si } P_{t+10} > P_t \text{ (Señal de Compra)} \\ 0 & \text{si } P_{t+10} \le P_t \text{ (Señal de Espera/Venta)} \end{cases}$$

A diferencia de versiones anteriores, este repositorio no presenta una estrategia “ganadora”, sino un **análisis honesto y reproducible de por qué modelos aparentemente prometedores no logran generar edge estadístico real**.

---

## ⚠️ Resultado Principal

> Un modelo con **ROC-AUC ≈ 0.49** puede generar curvas de capital atractivas…  
> pero **no necesariamente tiene poder predictivo real**.

---

## 📊 Evidencia Empírica

### 📈 Equity Curve — Bull Market (2023–2026)

![RF Bull](monitoring/img/medium_03_bull_equity_RF.png)
![XGB Bull](monitoring/img/medium_03_bull_equity_XG.png)

- Crecimiento consistente  
- Drawdowns controlados  
- Aparente robustez  






---

### 🎲 Monte Carlo — Test de Significancia

![RF Monte Carlo](monitoring/img/monte_carlo_RandomForest.png)

> ❌ **Resultado clave:**  
> El rendimiento del modelo **no es estadísticamente significativo** frente a estrategias aleatorias.

---


## 📉 El Caso del Bear Market (2022): Un Falso Positivo
> **Lección aprendida:** Cómo el Data Leakage puede disfrazarse de "estrategia defensiva".

Inicialmente, el modelo mostraba una protección de capital sólida durante 2022. Sin embargo, tras una auditoría interna, **este resultado fue invalidado**.

<details>
<summary>🔍 Ver análisis del error estructural (Haz clic para expandir)</summary>

### ❌ El problema: Data Leakage a nivel de Régimen
Aunque no había *lookahead bias* en las variables, el periodo de validación incluía datos de 2022 que el modelo ya había "visto" indirectamente, resultando en:
- **Evaluación no Out-of-Sample:** Contaminación indirecta del conjunto de prueba.
- **Sesgo de Supervivencia:** Ajuste involuntario a un régimen de mercado ya conocido.

### Conclusión 
El desempeño defensivo **NO es válido**. Este hallazgo fue el motor para implementar el *Purging + Embargo* y las simulaciones de *Monte Carlo* que ahora definen esta versión.
</details>

## 📂 Dataset y Partición de Datos

Para garantizar la integridad del estudio, se utilizó una separación cronológica estricta:

* **Training & Validation (In-Sample):** Datos históricos hasta finales de 2022. Incluye el periodo del "Bear Market", el cual se utilizó para el ajuste de parámetros (lo que explica el sesgo detectado anteriormente).
* **Test Set (Out-of-Sample):** Desde **enero de 2023 hasta la actualidad (2026)**. 
    * Son datos que el modelo **nunca vio** durante el entrenamiento.
    * Este periodo corresponde al **Bull Market** reciente.
    * **Propósito:** Evaluar si el modelo realmente aprendió patrones predictivos o si solo estaba "memorizando" la volatilidad pasada.

> 💡 **Nota Crítica:** Aunque las gráficas de 2023+ muestran retornos positivos, el test de Monte Carlo demuestra que esos retornos son indistinguibles de la suerte, confirmando que el modelo no tiene *edge* real en datos fuera de muestra.
---

## 🏗️ Arquitectura del Sistema

```text
yfinance (OHLCV)
   └─► Feature Engineering    # 26 indicadores técnicos + shift(1)
         └─► División Temporal  # Purging + Embargo (10 días)
               └─► Modelos      # Random Forest + XGBoost
                     └─► Señales # Threshold 0.55 → BUY / HOLD
                           └─► Backtest # Sin solapamiento (horizonte fijo)
                                 └─► Evaluación estadística (Monte Carlo)


```
## Hallazgos Clave

### 1. La Paradoja del ROC-AUC
* Un **ROC-AUC** bajo no implica automáticamente inutilidad, pero tampoco puede ignorarse.
* En este caso, el modelo no logra superar al azar, incluso en la cola de alta probabilidad.

### 2. Filtrar ≠ Predecir
El modelo opera poco, reduce la exposición y produce métricas atractivas, pero:
* **No genera señal predictiva real.**

### 3. Inestabilidad Temporal
Ejemplo de precisión por año:
* **2023:** ~61%
* **2024:** ~81%
* **2025:** ~69%
* **2026:** ~28%
* ⚠️ **Alta varianza:** El comportamiento no es robusto.

### 4. Overfitting estructural en series financieras
Se implementaron técnicas de *Advances in Financial Machine Learning*:
* Subsampling agresivo ($\alpha = 0.1$)
* Control de leakage (purging + embargo)
* Horizonte fijo sin solapamiento

**Resultado:** El modelo no logra capturar una señal estable.

---

## 🔬 Validación Rigurosa
* **Sin lookahead bias:** Uso de `shift(1)`.
* **Sin leakage:** Control estricto entre splits.
* **Backtest:** Sin overlapping trades.
* **Baseline:** Evaluación contra un modelo aleatorio.
* **Monte Carlo:** 1,000 simulaciones.

---

## ❗ Conclusión
Este proyecto demuestra que **una equity curve atractiva no implica edge**. En ML financiero es extremadamente fácil:
1. Sobreestimar resultados.
2. Malinterpretar métricas.
3. Encontrar patrones donde no existen.

---

## 🚀 Roadmap
* **Diferenciación fraccionaria:** Estacionariedad sin perder memoria.
* **Feature clustering:** Reducción de colinealidad.
* **Meta-labeling + bet sizing:** Separar dirección de tamaño.
* **Regime detection:** Switching basado en VIX o volatilidad.
* **Métricas de trading:** Evaluación más allá del ROC-AUC.

---

## 📚 Referencias
* **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley.
* **Davis, J., & Goadrich, M. (2006).** *The Relationship Between Precision-Recall and ROC Curves*.
* **Chan, E. P. (2008).** *Quantitative Trading*.

> **Nota Final:** Este repositorio no busca demostrar que el ML funciona en trading, sino qué tan fácil es creer que funciona cuando no lo hace.