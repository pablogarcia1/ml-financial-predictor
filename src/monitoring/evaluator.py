import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT_DIR / "monitoring"

NON_FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume",
                    "target", "log_return_10d", "ticker"]


def load_metrics_history() -> list:
    """Carga el historial de métricas guardadas."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "metrics_history.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_metrics(metrics: dict):
    """Agrega nuevas métricas al historial."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history = load_metrics_history()
    history.append(metrics)
    path = METRICS_DIR / "metrics_history.json"
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Métricas guardadas en {path}")


def evaluate_recent_predictions(model,
                                 ticker: str = "SPY",
                                 lookback_days: int = 60,
                                 horizon: int = 10) -> dict:
    """
    Evalúa el rendimiento del modelo sobre las últimas semanas.

    Lógica:
    - Toma las últimas 'lookback_days' filas del processed
    - Las últimas 'horizon' filas no tienen resultado aún (futuro)
    - Evalúa las anteriores donde ya conocemos el resultado real
    """
    from src.features.engineering import load_processed

    print(f"\nEvaluando modelo sobre {ticker} (últimos {lookback_days} días)...")

    df = load_processed(ticker)

    # Tomar el período reciente con resultado conocido
    # Las últimas 'horizon' filas aún no tienen resultado real
    df_eval = df.iloc[-(lookback_days + horizon):-horizon]

    if len(df_eval) < 10:
        print("  Insuficientes datos para evaluar.")
        return {}

    feature_cols = [c for c in df_eval.columns if c not in NON_FEATURE_COLS]
    X = df_eval[feature_cols]
    y = df_eval["target"]

    # Predicciones
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    # Métricas
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

    metrics = {
        "date":           str(date.today()),
        "ticker":         ticker,
        "periodo_inicio": str(df_eval.index[0].date()),
        "periodo_fin":    str(df_eval.index[-1].date()),
        "n_predicciones": len(y),
        "roc_auc":        round(roc_auc_score(y, proba), 4),
        "accuracy":       round(accuracy_score(y, pred), 4),
        "precision":      round(precision_score(y, pred, zero_division=0), 4),
        "recall":         round(recall_score(y, pred, zero_division=0), 4),
        "pct_positivos":  round(float(y.mean()), 4),
    }

    return metrics


def check_degradation(metrics: dict,
                       auc_threshold: float = 0.50,
                       consecutive_days: int = 3) -> bool:
    """
    Verifica si el modelo se está degradando.
    Alerta si el ROC-AUC cae por debajo del umbral
    en los últimos 'consecutive_days' registros.
    """
    history = load_metrics_history()

    if len(history) < consecutive_days:
        return False

    recent = history[-consecutive_days:]
    aucs   = [r["roc_auc"] for r in recent if r.get("ticker") == metrics.get("ticker")]

    if len(aucs) < consecutive_days:
        return False

    degraded = all(auc < auc_threshold for auc in aucs)

    if degraded:
        print(f"\nALERTA: Modelo degradado en {metrics['ticker']}")
        print(f"   ROC-AUC de los últimos {consecutive_days} días: {aucs}")
        print(f"   Considera reentrenar el modelo.")

    return degraded


def print_metrics(metrics: dict):
    """Imprime las métricas de evaluación."""
    print(f"\n{'═'*50}")
    print(f"  EVALUACIÓN DEL MODELO — {metrics['ticker']}")
    print(f"{'═'*50}")
    print(f"  Período:        {metrics['periodo_inicio']} → {metrics['periodo_fin']}")
    print(f"  Predicciones:   {metrics['n_predicciones']}")
    print(f"  {'─'*48}")
    print(f"  ROC-AUC:        {metrics['roc_auc']}")
    print(f"  Accuracy:       {metrics['accuracy']}")
    print(f"  Precision:      {metrics['precision']}")
    print(f"  Recall:         {metrics['recall']}")
    print(f"  % Positivos:    {metrics['pct_positivos']:.1%}")
    print(f"{'═'*50}")


def run_evaluation(model, tickers: list = ["SPY"]) -> list:
    """
    Función principal de evaluación.
    Evalúa el modelo sobre cada ticker y guarda las métricas.
    """
    all_metrics = []

    for ticker in tickers:
        metrics = evaluate_recent_predictions(model, ticker)

        if not metrics:
            continue

        print_metrics(metrics)
        save_metrics(metrics)
        check_degradation(metrics)
        all_metrics.append(metrics)

    return all_metrics


# ── Prueba rápida ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.features.engineering import load_processed, build_features
    from src.models.train import (load_multiple, split_data,
                                   get_features, train_xgboost)

    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]

    # Cargar datos procesados
    dfs = []
    for ticker in tickers:
        df = load_processed(ticker)
        df["ticker"] = ticker
        dfs.append(df)
    df_combined = pd.concat(dfs).sort_index()

    # Split y entrenamiento
    train_df, val_df, test_df = split_data(df_combined)
    X_train, y_train = get_features(train_df)
    X_val,   y_val   = get_features(val_df)

    print("Entrenando modelo para evaluación...")
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Evaluar
    run_evaluation(model, tickers=["SPY"])

    # Ver historial
    print("\nHistorial de métricas guardado:")
    history = load_metrics_history()
    for entry in history:
        print(f"  {entry['date']}  {entry['ticker']}  AUC={entry['roc_auc']}")