import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import ta
import os

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data/raw")
PROCESSED_DIR = ROOT_DIR / os.getenv("PROCESSED_DIR", "data/processed")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los features técnicos a partir de datos OHLCV.

    IMPORTANTE: se aplica shift(1) al final para evitar lookahead bias.
    En el día t, el modelo solo ve información disponible hasta t-1.

    Args:
        df: DataFrame con columnas Open, High, Low, Close, Volume

    Returns:
        DataFrame con features adicionales, sin filas con NaN
    """
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── 1. Momentum: log-returns ───────────────────────────────────────────────
    for n in [5, 10, 20, 60]:
        df[f"ret_{n}d"] = np.log(close / close.shift(n))

    # ── 2. Volatilidad móvil (anualizada) ─────────────────────────────────────
    daily_ret = np.log(close / close.shift(1))
    for n in [10, 20, 60]:
        df[f"vol_{n}d"] = daily_ret.rolling(n).std() * np.sqrt(252)

    # ── 3. Medias móviles simples ──────────────────────────────────────────────
    for n in [20, 50, 200]:
        df[f"sma_{n}"] = close.rolling(n).mean()

    # ── 4. Distancia del precio a las medias (señal de tendencia) ─────────────
    df["price_vs_sma20"]  = close / df["sma_20"]  - 1
    df["price_vs_sma50"]  = close / df["sma_50"]  - 1
    df["price_vs_sma200"] = close / df["sma_200"] - 1
    df["sma20_vs_sma50"]  = df["sma_20"] / df["sma_50"] - 1

    # ── 5. RSI ─────────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_7"]  = ta.momentum.RSIIndicator(close, window=7).rsi()

    # ── 6. MACD ────────────────────────────────────────────────────────────────
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()

    # ── 7. Bandas de Bollinger ─────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"]   = bb.bollinger_pband()   # posición dentro de las bandas (0-1)
    df["bb_width"] = bb.bollinger_wband()   # amplitud de las bandas

    # ── 8. ATR (Average True Range) ───────────────────────────────────────────
    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # ── 9. Volumen relativo ────────────────────────────────────────────────────
    df["vol_ratio_20d"] = volume / volume.rolling(20).mean()
    df["vol_change_1d"] = volume / volume.shift(1) - 1

    # ── 10. Distancia a máximos y mínimos de 52 semanas ───────────────────────
    df["dist_52w_high"] = close / close.rolling(252).max() - 1
    df["dist_52w_low"]  = close / close.rolling(252).min() - 1

    # ── CRÍTICO: shift(1) en todos los features ────────────────────────────────
    # Las columnas originales OHLCV y el target NO se shiftean
    cols_to_shift = [c for c in df.columns
                     if c not in ["Open", "High", "Low", "Close", "Volume",
                                  "target", "log_return_10d"]]
    df[cols_to_shift] = df[cols_to_shift].shift(1)

    # Eliminar filas con NaN (las primeras ~200 filas por sma_200)
    df = df.dropna()

    print(f"  Features calculados: {len(cols_to_shift)} columnas")
    print(f"  Filas resultantes:   {len(df)}")
    return df


def save_processed(df: pd.DataFrame, ticker: str) -> Path:
    """Guarda el DataFrame procesado en data/processed/"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{ticker}.parquet"
    df.to_parquet(path)
    print(f"  Guardado en {path}")
    return path


def load_processed(ticker: str) -> pd.DataFrame:
    """Carga datos procesados desde data/processed/"""
    path = PROCESSED_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta build_features primero.")
    return pd.read_parquet(path)


def build_features(ticker: str) -> pd.DataFrame:
    """
    Para entrenamiento: construye target y calcula features.
    Elimina las últimas 10 filas (sin target futuro).
    """
    from src.data.download_data import load_raw, build_target

    print(f"\nCalculando features para {ticker}...")
    df = load_raw(ticker)
    df = build_target(df, horizon=10)
    df = compute_features(df)
    save_processed(df, ticker)
    return df

def build_features_for_inference(ticker: str) -> pd.DataFrame:
    """
    Para predicción: NO construye target.
    Usa todos los datos disponibles hasta hoy.
    """
    from src.data.download_data import load_raw

    print(f"\nCalculando features de inferencia para {ticker}...")
    df = load_raw(ticker)

    cols_to_drop = [c for c in ["target", "log_return_10d"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df = compute_features(df)
    print(f"  Última fecha disponible: {df.index[-1].date()}")
    return df

# ── Prueba ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_features("AAPL")

    feature_cols = [c for c in df.columns
                    if c not in ["Open", "High", "Low", "Close", "Volume",
                                 "target", "log_return_10d"]]

    print(f"\nFeatures generados ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  {col}")

    print(f"\nPrimeras filas (columnas clave):")
    print(df[["Close", "ret_10d", "rsi_14", "macd_diff", "bb_pct", "target"]].head())