import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import date

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data/raw")

def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Descarga datos OHLCV de Yahoo Finance.

    Args:
        ticker: símbolo del activo, ej: 'AAPL'
        start:  fecha inicio 'YYYY-MM-DD'
        end:    fecha fin    'YYYY-MM-DD'

    Returns:
        DataFrame con columnas: Open, High, Low, Close, Volume
    """
    print(f"Descargando {ticker} desde {start} hasta {end}...")

    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No se encontraron datos para {ticker}")

    # Aplanar columnas si yfinance devuelve MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = ["Close", "High", "Low", "Open", "Volume"]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    df = df.dropna()

    print(f"  {len(df)} filas descargadas ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def build_target(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """
    Construye la variable objetivo binaria usando log-returns.

    y = 1  si  log(P(t+horizon) / P(t)) > 0   → el precio sube
    y = 0  si  log(P(t+horizon) / P(t)) <= 0   → el precio no sube

    Args:
        df:      DataFrame con columna 'Close'
        horizon: número de días de mercado hacia adelante (default: 10)

    Returns:
        DataFrame con columnas adicionales: log_return_Nd, target
    """
    df = df.copy()

    df[f"log_return_{horizon}d"] = np.log(
        df["Close"].shift(-horizon) / df["Close"]
    )
    df["target"] = (df[f"log_return_{horizon}d"] > 0).astype(int)

    # Eliminar filas sin target (las últimas 'horizon' filas)
    df = df.dropna(subset=["target"])

    pos = df["target"].mean()
    print(f"  Target construido: {pos:.1%} positivos (sube), {1-pos:.1%} negativos")
    return df


def save_raw(df: pd.DataFrame, ticker: str) -> Path:
    """Guarda el DataFrame en data/raw como Parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{ticker}.parquet"
    df.to_parquet(path)
    print(f"  Guardado en {path}")
    return path


def load_raw(ticker: str) -> pd.DataFrame:
    """Carga datos crudos desde data/raw."""
    path = DATA_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta download_ohlcv primero.")
    return pd.read_parquet(path)


def ingest(ticker: str, start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """
    Guarda OHLCV puro en raw (sin target, sin eliminar filas).
    El target se construye después en el pipeline de entrenamiento.
    """
    if end is None:
        end = str(date.today())

    df = download_ohlcv(ticker, start, end)
    save_raw(df, ticker)           # guarda OHLCV puro hasta hoy ✅
    return df


if __name__ == "__main__":
    df = ingest("AAPL")
    print("\nPrimeras filas:")
    print(df[["Open", "Close", "Volume", "log_return_10d", "target"]].head())
    print(f"\nShape: {df.shape}")
    print(f"Rango: {df.index[0].date()} → {df.index[-1].date()}")