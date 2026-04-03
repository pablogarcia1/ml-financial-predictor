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
    signals.loc[signals["proba"] > buy_threshold,  "signal"] = "BUY"
    signals.loc[signals["proba"] < sell_threshold, "signal"] = "SELL"

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
    """
    Calcula el retorno de la estrategia Buy & Hold como benchmark.
    Compra al inicio del test y vende al final.
    """
    start_price = signals["close"].iloc[0]
    end_price   = signals["close"].iloc[-1]
    log_return  = np.log(end_price / start_price)

    # Número de períodos de 10 días en el test
    n_periods = len(signals) / horizon
    sharpe    = (log_return / n_periods) / (signals["close"].pct_change().std() * np.sqrt(252 / 10))

    return {
        "cumulative_return": round(np.expm1(log_return) * 100, 2),
        "sharpe_ratio":      round(sharpe, 4),
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
    if len(trades) == 0:
        print("  Sin operaciones — prueba bajando el buy_threshold")
        return {}

    return {"metrics": metrics, "benchmark": benchmark, "trades": trades}


# ── Prueba rápida ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.download_data import ingest
    from src.features.engineering import build_features
    from src.models.train import (load_multiple, split_data,
                                   get_features, train_xgboost)

    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]

    # Cargar datos ya procesados
    from src.features.engineering import load_processed
    dfs = []
    for ticker in tickers:
        df = load_processed(ticker)
        df["ticker"] = ticker
        dfs.append(df)

    df_combined = pd.concat(dfs).sort_index()

    # Split
    train_df, val_df, test_df = split_data(df_combined)
    X_train, y_train = get_features(train_df)
    X_val,   y_val   = get_features(val_df)

    # Entrenar XGBoost
    print("Entrenando modelo para backtest...")
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Usar solo SPY para el backtest (un ticker limpio)
    test_spy = test_df[test_df["ticker"] == "SPY"]

    print(f"\nBacktest sobre SPY ({test_spy.index[0].date()} → {test_spy.index[-1].date()})")
    results = run_backtest(model, test_spy, buy_threshold=0.50, sell_threshold=0.45)