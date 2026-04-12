---

## Instalación

```bash
git clone https://github.com/TU_USUARIO/ml-financial-predictor.git
cd ml-financial-predictor
pip install -r requirements.txt
```

---

## Uso rápido

```bash
# Pipeline completo
python src/run_pipeline.py

# Solo la API
python api/main.py
# Abre http://localhost:8000/docs
```

---

## Resultados

### Bull Market (2023 – 2026)
| Métrica | Estrategia | Benchmark (SPY) |
|---|---|---|
| Retorno Acumulado | 70.54% | 72.99% |
| Sharpe Ratio | 1.42 | 1.15 |
| Máx Drawdown | -8.33% | — |
| Win Rate | 75% | — |
| Profit Factor | 2.05 | — |

### Bear Market Stress Test (2022)
| Métrica | Estrategia | Benchmark (SPY) |
|---|---|---|
| Retorno Acumulado | -1.77% | -18.65% |
| Sharpe Ratio | -0.15 | -0.86 |

**Conclusión:** El modelo no supera al benchmark en retorno absoluto en mercados alcistas, pero preserva capital en mercados bajistas con una diferencia de +16.88 puntos porcentuales.

---

## Decisiones de Diseño y Aprendizajes

### 1. Separación entre ingesta e ingeniería de features
El target de 10 días requiere `shift(-10)`, lo que elimina las últimas 10 filas del dataset. Si se construye el target en la ingesta, los datos guardados en `raw/` ya llegan incompletos y la API predice con un retraso de 10 días.

**Solución:** `download_data.py` guarda OHLCV puro. `engineering.py` construye el target solo para entrenamiento. Para inferencia se usa `build_features_for_inference()` que no elimina filas.

### 2. Data leakage por solapamiento entre splits
El target `log(P(t+10)/P(t))` usa precios futuros. Si train termina el 31 de dic y validación empieza el 1 de enero, las últimas 10 filas de train usan precios de enero — que ya son datos de validación.

**Solución:** Gap de 10 días de mercado reales entre cada split usando fechas únicas del índice, no filas.

Se aplicó un shift(1) estricto en el motor de cálculo de las métricas financieras (backtester). Esto asegura que cualquier decisión de trading o cálculo de retorno asuma la ejecución al cierre real de t-1, garantizando cero Look-Ahead Bias.

### 3. Overfitting severo en los primeros experimentos
Train AUC: 0.99  |  Val AUC: 0.49  ← peor que azar

Los mercados financieros tienen muy poca señal. Modelos complejos memorizan ruido.

**Solución:** Regularización agresiva — `max_depth=2`, `min_child_weight=30`, `learning_rate=0.01`.

### 4. El Sharpe del backtest estaba inflado por solapamiento de posiciones
Con señales BUY en días consecutivos, múltiples posiciones se abrían simultáneamente. El retorno acumulado sumaba operaciones paralelas como si fueran secuenciales.

**Solución:** `next_entry_allowed = exit_date` — solo se abre una posición nueva cuando la anterior cierra.

### 5. El Monte Carlo reveló que la señal no es estadísticamente significativa
Con threshold bajo (0.50), el modelo opera 87% del tiempo siguiendo al mercado. Los monos aleatorios con el mismo threshold obtienen el mismo Sharpe.

**Aprendizaje:** En un bull market 2023-2026 cualquier estrategia que compre frecuentemente gana. El valor real del modelo está en su selectividad, no en su frecuencia.

### 6. El valor real está en la preservación de capital
El modelo con threshold 0.55 operó solo 17.9% del tiempo en 2022. Resultado: -1.77% vs -18.65% del benchmark. La señal genuina no es "cuándo subir" sino "cuándo no estar expuesto".

### 7. Métricas financieras vs métricas de ML
Un ROC-AUC de 0.51 parece inútil como métrica de clasificación. Pero traducido a operaciones reales con gestión de posiciones produce un Sharpe de 1.42 y un drawdown de solo -8.33%. Las métricas de ML y las métricas financieras miden cosas distintas.

---

## API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| POST | `/train` | Reentrena el modelo con datos actuales |
| GET | `/predict/spy` | Predicción para SPY hoy |
| POST | `/predict` | Predicción para cualquier ticker |
| GET | `/backtest/spy` | Métricas históricas del modelo |

### Ejemplo de respuesta `/predict/spy`
```json
{
  "ticker": "SPY",
  "fecha_prediccion": "2026-03-30",
  "fecha_objetivo": "2026-04-13",
  "probabilidad_suba": 0.5129,
  "signal": "HOLD",
  "buy_threshold": 0.55
}
```

---

## Limitaciones y Trabajo Futuro

- **Sin gestión de riesgo:** No hay position sizing ni stop loss automático
- **Features solo técnicos:** Agregar VIX, tasas de interés y datos macro mejoraría la señal
- **Un solo activo principal:** El modelo fue diseñado y validado principalmente para SPY
- **Sin reentrenamiento automático:** El modelo necesita reentrenarse manualmente cada mes
- **P-Value no significativo:** La señal estadística es débil con features técnicos solos

---

## Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Datos | `yfinance`, `pandas`, `pyarrow` |
| Features | `ta` (Technical Analysis library) |
| Modelos | `xgboost`, `scikit-learn` |
| Tracking | `MLflow` |
| API | `FastAPI`, `uvicorn`, `pydantic` |
| Estadística | `scipy`, `numpy` |
| Despliegue | `Docker` |