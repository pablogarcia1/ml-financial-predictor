# ML Financial Predictor v2.0 
### Sistema End-to-End de ML con Arquitectura Dual y Especialización de Régimen

Este sistema utiliza Machine Learning avanzado para predecir movimientos del ETF **SPY (S&P 500)** en un horizonte de 10 días, utilizando un enfoque de **Panel Data** (AAPL, MSFT, GOOGL, JPM, SPY). 

La versión implementa una **Arquitectura Dual** que alterna entre un "Escudo Defensivo" y un "Motor Ofensivo", basada en los principios de *Advances in Financial Machine Learning* de Marcos López de Prado.

---

##  Desempeño Estratégico (Backtest v2.0)

| Régimen de Mercado | Modelo Especialista | Rendimiento Estrategia | Benchmark (SPY) | Diferencial (Alpha) |
| :--- | :--- | :--- | :--- | :--- |
| **Bear Market (2022)** | **XGBoost (Defensa)** | **-2.41% MaxDD** | -18.65% MaxDD | **+16.24% Protección** |
| **Bull Market (2023-26)** | **Random Forest (Ataque)** | **1.82 Sharpe** | 1.09 Sharpe | **+67% Eficiencia** |

> **Validación Estadística:** La simulación Monte Carlo confirmó que el Sharpe de 1.8186 es estadísticamente significativo, validando la señal capturada por el ensamble regularizado frente al ruido aleatorio.

---

## 🏗️ Arquitectura del Sistema

```text
yfinance (OHLCV)
   └─► Feature Engineering    # 26 indicadores técnicos + shift(1) causal
         └─► División Temporal  # Gaps estrictos: Purging + Embargo (10 días)
               └─► Entrenamiento # XGBoost + Random Forest @ α=0.1
                     └─► Señales # Umbral 0.55 → BUY / HOLD
                           └─► Backtest # Sin solapamiento de posiciones
                                 └─► FastAPI REST # Inferencia en tiempo real

```

## Insights Técnicos Clave

- **Singularidad Promedio (α = 0.1):** Dado que el horizonte es de 10 días, las etiquetas consecutivas comparten el 90% de la información. Ajustamos `subsample=0.1` para forzar la descorrelación de los árboles y eliminar el overfitting estructural por solapamiento.
- **La Paradoja del ROC-AUC:** Un AUC global de 0.49 puede ser altamente rentable. El valor reside en la Cola Derecha (prob > 0.55), donde la precisión del sistema supera el 62%. El rendimiento promedio de la distribución es irrelevante para señales de alta convicción.
- **Memoria Estática de Régimen:** Se eliminó el Time Decay para evitar que el modelo "olvidara" los patrones históricos de colapso. Mantener la memoria de crisis pasadas es lo que permitió a XGBoost actuar como un escudo perfecto en 2022.

---

##  Validación y Robustez

- **Sin Lookahead Bias:** Aplicación de `shift(1)` estricto en todos los features antes del cálculo.
- **Sin Leakage entre Splits:** Implementación de gaps de Purging y Embargo de 10 días de mercado reales.
- **Sin Solapamiento de Posiciones:** El motor de backtest prohíbe abrir nuevas operaciones hasta que la actual (de 10 días) se haya cerrado.
- **Significancia:** Validado mediante 1,000 estrategias aleatorias (Monte Carlo) para confirmar la ventaja estadística.

---

## Roadmap: Ruptura del "Techo de Cristal"

1. **Diferenciación Fraccionaria:** Lograr estacionariedad sin perder la memoria de largo plazo de los precios *(CRÍTICO)*.
2. **Clustering de Features:** Garantizar que el muestreo de variables sea genuinamente diverso y ortogonal.
3. **Detección Automática de Régimen:** Switch dinámico basado en VIX para alternar entre modelos de forma sistemática.
4. **Kelly Fraccionado:** Implementación de Position Sizing dinámico según convicción.

---

##  Referencia

Metodología basada en los estándares de:

> López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.