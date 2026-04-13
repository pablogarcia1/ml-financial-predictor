import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def generate_signals(model, df: pd.DataFrame,
                     buy_threshold: float = 0.60,
                     sell_threshold: float = 0.40) -> pd.DataFrame:
    """
    Genera señales de trading a partir de las predicciones del modelo.

    Args:
        model:          modelo entrenado con predict_proba
        df:             DataFrame del test set con features y target
        buy_threshold:  probabilidad mínima para señal BUY
        sell_threshold: probabilidad máxima para señal SELL

    Returns:
        DataFrame con columnas: proba, signal, target, close
    """
    from src.models.train import NON_FEATURE_COLS

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]

    proba = model.predict_proba(X)[:, 1]

    signals = pd.DataFrame(index=df.index)
    signals["proba"]  = proba
    signals["target"] = df["target"].values
    signals["close"]  = df["Close"].values

    signals["signal"] = "HOLD"
    signals.loc[signals["proba"] >= buy_threshold,  "signal"] = "BUY"
    signals.loc[signals["proba"] <= sell_threshold, "signal"] = "SELL"

    n_buy  = (signals["signal"] == "BUY").sum()
    n_sell = (signals["signal"] == "SELL").sum()
    n_hold = (signals["signal"] == "HOLD").sum()

    print(f"  Señales generadas:")
    print(f"    BUY:  {n_buy}  ({n_buy/len(signals):.1%})")
    print(f"    SELL: {n_sell}  ({n_sell/len(signals):.1%})")
    print(f"    HOLD: {n_hold}  ({n_hold/len(signals):.1%})")

    return signals


def compute_returns(signals: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """
    Calcula el retorno real de cada operación BUY.
    Solo entra en una nueva operación cuando la anterior ya cerró.
    Evita el solapamiento de posiciones.
    """
    trades = []
    next_entry_allowed = signals.index[0]  # primera fecha disponible

    buy_signals = signals[signals["signal"] == "BUY"].copy()

    for entry_date, row in buy_signals.iterrows():

        # Si todavía hay una posición abierta, saltar
        if entry_date < next_entry_allowed:
            continue

        # Buscar precio de salida 'horizon' días después
        future_prices = signals.loc[entry_date:]["close"].iloc[1:horizon + 1]

        if len(future_prices) < horizon:
            continue

        exit_date   = future_prices.index[-1]
        entry_price = row["close"]
        exit_price  = future_prices.iloc[-1]

        log_return = np.log(exit_price / entry_price)
        return_pct = (exit_price / entry_price - 1) * 100

        trades.append({
            "entry_date":  entry_date,
            "exit_date":   exit_date,
            "proba":       row["proba"],
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "log_return":  log_return,
            "return_pct":  return_pct,
            "result":      "WIN" if log_return > 0 else "LOSS"
        })

        # La siguiente entrada solo es válida después del cierre
        next_entry_allowed = exit_date

    if not trades:
        print("  Sin operaciones ejecutadas.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades).set_index("entry_date")

    return trades_df

def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """
    Calcula métricas financieras del backtest.

    Args:
        trades_df: DataFrame con columna log_return por operación

    Returns:
        dict con todas las métricas
    """
    rets = trades_df["log_return"].values
    wins = trades_df["result"] == "WIN"

    # Retorno acumulado
    cumulative_return = np.expm1(np.sum(rets)) * 100

    # Sharpe Ratio anualizado (asumiendo ~25 operaciones por año)
    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252 / 10)
    else:
        sharpe = 0.0

    # Máximo Drawdown
    cum_log    = np.cumsum(rets)
    running_max = np.maximum.accumulate(cum_log)
    drawdown    = cum_log - running_max
    max_drawdown = np.expm1(np.min(drawdown)) * 100

    # Win Rate
    win_rate = wins.mean() * 100

    # Profit Factor
    gross_profit = rets[rets > 0].sum()
    gross_loss   = abs(rets[rets < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Retorno promedio por operación
    avg_return = np.expm1(rets.mean()) * 100

    metrics = {
        "n_trades":          len(trades_df),
        "win_rate":          round(win_rate, 2),
        "cumulative_return": round(cumulative_return, 2),
        "avg_return_pct":    round(avg_return, 4),
        "sharpe_ratio":      round(sharpe, 4),
        "max_drawdown_pct":  round(max_drawdown, 2),
        "profit_factor":     round(profit_factor, 4),
    }

    return metrics


def benchmark_returns(signals: pd.DataFrame, horizon: int = 10) -> dict:
    close = signals["close"]

    # Retorno acumulado
    log_return = np.log(close.iloc[-1] / close.iloc[0])

    # Retornos diarios para el Sharpe
    daily_returns = np.log(close / close.shift(1)).dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    return {
        "cumulative_return": round(np.expm1(log_return) * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
    }


def print_results(metrics: dict, benchmark: dict):
    """Imprime comparación entre estrategia y benchmark."""
    print(f"\n{'═'*50}")
    print(f"  RESULTADOS DEL BACKTEST")
    print(f"{'═'*50}")
    print(f"  {'Métrica':<25} {'Estrategia':>12} {'Benchmark':>12}")
    print(f"  {'─'*49}")
    print(f"  {'Operaciones':<25} {metrics['n_trades']:>12}")
    print(f"  {'Win Rate %':<25} {metrics['win_rate']:>11.2f}%")
    print(f"  {'Retorno Acumulado %':<25} {metrics['cumulative_return']:>11.2f}%  {benchmark['cumulative_return']:>11.2f}%")
    print(f"  {'Sharpe Ratio':<25} {metrics['sharpe_ratio']:>12.4f}  {benchmark['sharpe_ratio']:>12.4f}")
    print(f"  {'Máx Drawdown %':<25} {metrics['max_drawdown_pct']:>11.2f}%")
    print(f"  {'Profit Factor':<25} {metrics['profit_factor']:>12.4f}")
    print(f"{'═'*50}")


def run_backtest(model, test_df: pd.DataFrame,
                 buy_threshold: float = 0.60,
                 sell_threshold: float = 0.40,
                 horizon: int = 10) -> dict:
    """
    Función principal del backtest.

    Args:
        model:          modelo entrenado
        test_df:        DataFrame del período de test
        buy_threshold:  umbral para señal BUY
        sell_threshold: umbral para señal SELL
        horizon:        días de holding por operación

    Returns:
        dict con métricas del backtest
    """
    print(f"\nGenerando señales...")
    signals  = generate_signals(model, test_df, buy_threshold, sell_threshold)

    print(f"\nCalculando retornos de operaciones...")
    trades   = compute_returns(signals, horizon)

    if len(trades) == 0:
        print("  Sin operaciones — prueba bajando el buy_threshold")
        return {}

    metrics   = compute_metrics(trades)
    benchmark = benchmark_returns(signals, horizon)

    print_results(metrics, benchmark)

    return {"metrics": metrics, "benchmark": benchmark, "trades": trades}

def monte_carlo_significance(model,
                              test_df: pd.DataFrame,
                              real_sharpe: float,
                              n_simulations: int = 1000,
                              buy_threshold: float = 0.52,
                              horizon: int = 10,
                              save_plot: bool = True) -> dict:
    """
    Prueba de significancia estadística via Simulación Monte Carlo.

    Genera N distribuciones de Sharpe aleatorias y compara contra
    el Sharpe real del modelo para calcular Z-Score y P-Value.

    Args:
        model:          modelo entrenado (no se usa para las simulaciones)
        test_df:        DataFrame del período de test con columna 'Close'
        real_sharpe:    Sharpe ratio real obtenido por el modelo
        n_simulations:  número de simulaciones aleatorias (default: 1000)
        buy_threshold:  umbral de activación de señal (default: 0.52)
        horizon:        días de holding por operación (default: 10)
        save_plot:      guardar gráfico en disco (default: True)

    Returns:
        dict con z_score, p_value, mean_random, std_random
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    print(f"\nEjecutando Monte Carlo ({n_simulations} simulaciones)...")

    # ── Retornos logarítmicos reales del mercado ───────────────────────────────
    close       = test_df["Close"].values
    # Usamos saltos continuos (::) para que los retornos no se solapen
    log_returns = np.log(close[horizon::horizon] / close[::horizon][:-1])
    n           = len(log_returns)

    # ── Vectorización matricial: N simulaciones x T señales ───────────────────
    # Genera matriz de señales aleatorias U(0,1) → shape (n_simulations, n)
    random_probas = np.random.uniform(0, 1, size=(n_simulations, n))

    # Convierte a señales binarias usando el umbral
    # 1 = BUY, 0 = HOLD  →  shape (n_simulations, n)
    random_signals = (random_probas >= buy_threshold).astype(float)

    # Cruza señales con retornos reales
    # Señal en t-1 con retorno en t (causalidad temporal)
    # shape: (n_simulations, n-1)
    signals_t     = random_signals[:, :-1]   # señales en t
    returns_t1    = log_returns[1:]           # retornos en t+1

    # Retornos de cada simulación: solo donde hay señal BUY
    # shape: (n_simulations, n-1)
    strategy_returns = signals_t * returns_t1
    sharpe_distribution = np.zeros(n_simulations)

    active_fraction = signals_t.mean(axis=1)  # % días con BUY por simulación
    means = (strategy_returns.sum(axis=1) /
             signals_t.sum(axis=1).clip(min=1))  # media solo sobre activos

    # Varianza solo sobre días activos usando fórmula matricial
    squared = strategy_returns ** 2
    variance = (squared.sum(axis=1) / signals_t.sum(axis=1).clip(min=1)) - means ** 2
    stds_adj = np.sqrt(variance.clip(min=0))

    valid = stds_adj > 0
    sharpe_distribution[valid] = (means[valid] / stds_adj[valid]) * np.sqrt(252 / horizon)

    # ── Z-Score y P-Value (cola derecha) ──────────────────────────────────────
    mu_rand    = sharpe_distribution.mean()
    std_rand   = sharpe_distribution.std()
    z_score    = (real_sharpe - mu_rand) / std_rand
    p_value    = 1 - stats.norm.cdf(z_score)  # cola derecha

    # ── Resultados ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  PRUEBA DE SIGNIFICANCIA ESTADÍSTICA (Monte Carlo)")
    print(f"{'═'*55}")
    print(f"  Simulaciones:          {n_simulations}")
    print(f"  Sharpe real del modelo: {real_sharpe:.4f}")
    print(f"  Sharpe medio aleatorio: {mu_rand:.4f}")
    print(f"  Desv. estándar:         {std_rand:.4f}")
    print(f"  {'─'*53}")
    print(f"  Z-Score:                {z_score:.4f}")
    print(f"  P-Value (cola derecha): {p_value:.4f}")

    if p_value < 0.01:
        verdict = "MUY SIGNIFICATIVO  (p < 0.01) — señal real"
    elif p_value < 0.05:
        verdict = "SIGNIFICATIVO      (p < 0.05) — señal real"
    elif p_value < 0.10:
        verdict = "MARGINAL          (p < 0.10) — señal débil"
    else:
        verdict = "NO SIGNIFICATIVO   (p >= 0.10) — posible ruido"

    print(f"  Veredicto:              {verdict}")
    print(f"{'═'*55}")

    # ── Gráfico ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(sharpe_distribution, bins=50, color="#2E75B6", alpha=0.7,
            edgecolor="white", label=f"Distribución aleatoria\n"
                                     f"μ={mu_rand:.3f}, σ={std_rand:.3f}")

    ax.axvline(real_sharpe, color="#E74C3C", linewidth=2.5, linestyle="--",
               label=f"Modelo real: {real_sharpe:.3f}\n"
                     f"Z={z_score:.3f} | P={p_value:.4f}\n"
                     f"{verdict}")

    # Área de la cola derecha (Corregida)
    ax.axvspan(real_sharpe, sharpe_distribution.max() + 1, alpha=0.15, color="#E74C3C")

    ax.set_xlabel("Sharpe Ratio", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title(f"Monte Carlo — Distribución de Sharpe Aleatorio\n"
                 f"N={n_simulations} simulaciones | Umbral BUY > {buy_threshold}",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_plot:
        plot_path = ROOT_DIR / "monitoring" / "monte_carlo.png"
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"\n  Gráfico guardado en {plot_path}")

    plt.close()

    return {
        "real_sharpe":   real_sharpe,
        "mean_random":   round(mu_rand, 4),
        "std_random":    round(std_rand, 4),
        "z_score":       round(z_score, 4),
        "p_value":       round(p_value, 4),
        "n_simulations": n_simulations,
        "verdict":       verdict
    }


#----------------------------
if __name__ == "__main__":
    from src.data.download_data import ingest
    from src.features.engineering import build_features
    from src.models.train import (load_multiple, split_data,
                                  get_features, train_xgboost, train_random_forest)
    import pandas as pd

    # 1. Carga de datos
    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]
    from src.features.engineering import load_processed

    dfs = []
    for ticker in tickers:
        df = load_processed(ticker)
        df["ticker"] = ticker
        dfs.append(df)

    df_combined = pd.concat(dfs).sort_index()

    # 2. Split
    train_df, val_df, test_df = split_data(df_combined)
    X_train, y_train = get_features(train_df)
    X_val, y_val = get_features(val_df)

    # 3. Entrenar AMBOS Modelos
    print("\n" + " " * 20)
    print(" FASE 1: ENTRENAMIENTO DE MODELOS")
    print("- " * 20)

    print("\nEntrenando Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    print("\nEntrenando XGBoost...")
    # XGBoost requiere validation set para el early stopping y regularización
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    # Umbral Institucional Defensivo
    THRESHOLD = 0.55

    # =================================================================
    # ESCENARIO 1: EL ESCUDO (Bear Market 2022 - Set de Validación)
    # =================================================================
    val_spy_2022 = val_df[(val_df["ticker"] == "SPY") &
                          (val_df.index >= "2022-01-01") &
                          (val_df.index <= "2022-12-31")]

    print("\n\n" + " " * 20)
    print(f" ESCENARIO 1: BEAR MARKET ({val_spy_2022.index[0].date()} → {val_spy_2022.index[-1].date()})")
    print(" -" * 20)

    print("\n▶ EVALUANDO: RANDOM FOREST (Bear Market)")
    res_bear_rf = run_backtest(rf_model, val_spy_2022, buy_threshold=THRESHOLD, sell_threshold=0.45)
    if res_bear_rf:
        monte_carlo_significance(rf_model, val_spy_2022, res_bear_rf["metrics"]["sharpe_ratio"],
                                 n_simulations=1000, buy_threshold=THRESHOLD, save_plot=False)

    print("\n" + "-" * 60)

    print("\n EVALUANDO: XGBOOST (Bear Market)")
    res_bear_xgb = run_backtest(xgb_model, val_spy_2022, buy_threshold=THRESHOLD, sell_threshold=0.45)
    if res_bear_xgb:
        monte_carlo_significance(xgb_model, val_spy_2022, res_bear_xgb["metrics"]["sharpe_ratio"],
                                 n_simulations=1000, buy_threshold=THRESHOLD, save_plot=False)

    # =================================================================
    # ESCENARIO 2: EL MOTOR (Bull Market 2023+ - Set de Test)
    # =================================================================
    test_spy_bull = test_df[test_df["ticker"] == "SPY"]

    print("\n\n" + "- " * 20)
    print(f" ESCENARIO 2: BULL MARKET ({test_spy_bull.index[0].date()} → {test_spy_bull.index[-1].date()})")
    print("- " * 20)

    print("\n▶️ EVALUANDO: RANDOM FOREST (Bull Market)")
    res_bull_rf = run_backtest(rf_model, test_spy_bull, buy_threshold=THRESHOLD, sell_threshold=0.45)
    if res_bull_rf:
        monte_carlo_significance(rf_model, test_spy_bull, res_bull_rf["metrics"]["sharpe_ratio"],
                                 n_simulations=1000, buy_threshold=THRESHOLD, save_plot=False)

    print("\n" + "-" * 60)

    print("\n EVALUANDO: XGBOOST (Bull Market)")
    res_bull_xgb = run_backtest(xgb_model, test_spy_bull, buy_threshold=THRESHOLD, sell_threshold=0.45)
    if res_bull_xgb:
        monte_carlo_significance(xgb_model, test_spy_bull, res_bull_xgb["metrics"]["sharpe_ratio"],
                                 n_simulations=1000, buy_threshold=THRESHOLD, save_plot=False)