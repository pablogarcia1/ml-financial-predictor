import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime
from pathlib import Path
import joblib
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.download_data import ingest
from src.features.engineering import build_features, load_processed
from src.models.train import (load_multiple, split_data, get_features,
                              train_xgboost, NON_FEATURE_COLS)
from src.backtest.engine import run_backtest

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Financial Predictor",
    description="Predice la probabilidad de que SPY suba en 10 días de mercado.",
    version="1.0.0"
)

# ── Estado global del modelo ──────────────────────────────────────────────────
STATE = {
    "model": None,
    "trained_at": None,
    "test_df": None,
    "tickers": ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"],
    "buy_threshold": 0.55,
}


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    ticker: str = "SPY"


class PredictResponse(BaseModel):
    ticker: str
    fecha_prediccion: str
    fecha_objetivo: str
    probabilidad_suba: float
    signal: str
    buy_threshold: float


class TrainResponse(BaseModel):
    status: str
    trained_at: str
    tickers_used: list
    message: str


class BacktestResponse(BaseModel):
    ticker: str
    periodo_inicio: str
    periodo_fin: str
    n_operaciones: int
    win_rate: float
    retorno_acumulado: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    benchmark_retorno: float
    benchmark_sharpe: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    trained_at: Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_next_trading_date(from_date: date, days: int = 10) -> str:
    """Estima la fecha objetivo sumando días calendario (~14 días = 10 días de mercado)."""
    import datetime as dt
    calendar_days = int(days * 1.4)
    target = from_date + dt.timedelta(days=calendar_days)
    return str(target)


def load_or_train_model():
    """Carga el modelo si ya está entrenado, si no lo entrena."""
    if STATE["model"] is not None:
        return STATE["model"]

    print("Modelo no encontrado en memoria — entrenando...")
    _train_model()
    return STATE["model"]


def _train_model():
    """Descarga datos, calcula features y entrena el modelo."""
    tickers = STATE["tickers"]

    # Descargar y procesar
    for ticker in tickers:
        ingest(ticker)
        build_features(ticker)

    # Combinar
    dfs = []
    for ticker in tickers:
        df = load_processed(ticker)
        df["ticker"] = ticker
        dfs.append(df)
    df_combined = pd.concat(dfs).sort_index()

    # Split y entrenamiento
    train_df, val_df, test_df = split_data(df_combined)
    X_train, y_train = get_features(train_df)
    X_val, y_val = get_features(val_df)

    model = train_xgboost(X_train, y_train, X_val, y_val)

    STATE["model"] = model
    STATE["trained_at"] = datetime.now().isoformat()
    STATE["test_df"] = test_df

    print(f"Modelo entrenado correctamente a las {STATE['trained_at']}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """Verifica el estado del servicio."""
    return HealthResponse(
        status="ok",
        model_loaded=STATE["model"] is not None,
        trained_at=STATE["trained_at"]
    )


@app.post("/train", response_model=TrainResponse)
def train():
    """
    Descarga los datos más recientes y reentrena el modelo.
    Llama este endpoint una vez al inicio o cuando quieras actualizar el modelo.
    """
    try:
        _train_model()
        return TrainResponse(
            status="ok",
            trained_at=STATE["trained_at"],
            tickers_used=STATE["tickers"],
            message="Modelo entrenado correctamente con datos actualizados."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Genera una predicción para el ticker indicado.
    Descarga los últimos 300 días, calcula features y predice.

    - signal BUY  → probabilidad > buy_threshold
    - signal HOLD → en cualquier otro caso
    """
    try:
        model = load_or_train_model()
        ticker = request.ticker.upper()

        # Descargar datos recientes del ticker
        from src.features.engineering import build_features_for_inference
        ingest(ticker)
        df = build_features_for_inference(ticker)

        # Tomar la última fila disponible
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        ultima_fila = df[feature_cols].iloc[[-1]]
        ultima_fecha = df.index[-1].date()

        # Predicción
        proba = float(model.predict_proba(ultima_fila)[:, 1][0])
        signal = "BUY" if proba > STATE["buy_threshold"] else "HOLD"

        return PredictResponse(
            ticker=ticker,
            fecha_prediccion=str(ultima_fecha),
            fecha_objetivo=get_next_trading_date(ultima_fecha, days=10),
            probabilidad_suba=round(proba, 4),
            signal=signal,
            buy_threshold=STATE["buy_threshold"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/spy", response_model=PredictResponse)
def predict_spy():
    """
    Shortcut: predicción directa para SPY sin body en el request.
    El endpoint más útil del sistema.
    """
    return predict(PredictRequest(ticker="SPY"))


@app.get("/backtest/spy", response_model=BacktestResponse)
def backtest_spy():
    """
    Ejecuta el backtest sobre SPY usando el test set del modelo actual.
    """
    try:
        model = load_or_train_model()

        if STATE["test_df"] is None:
            raise HTTPException(status_code=400,
                                detail="Entrena el modelo primero con POST /train")

        test_spy = STATE["test_df"][STATE["test_df"]["ticker"] == "SPY"]

        results = run_backtest(
            model, test_spy,
            buy_threshold=STATE["buy_threshold"],
            sell_threshold=0.45
        )

        m = results["metrics"]
        b = results["benchmark"]

        return BacktestResponse(
            ticker="SPY",
            periodo_inicio=str(test_spy.index[0].date()),
            periodo_fin=str(test_spy.index[-1].date()),
            n_operaciones=m["n_trades"],
            win_rate=m["win_rate"],
            retorno_acumulado=m["cumulative_return"],
            sharpe_ratio=m["sharpe_ratio"],
            max_drawdown=m["max_drawdown_pct"],
            profit_factor=m["profit_factor"],
            benchmark_retorno=b["cumulative_return"],
            benchmark_sharpe=b["sharpe_ratio"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)