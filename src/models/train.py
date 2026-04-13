import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data/processed"
MLFLOW_DIR = ROOT_DIR / "mlruns"
# Agrega esta
MLFLOW_TRACKING_URI = f"file:///{ROOT_DIR.as_posix()}/mlruns"
# Columnas que NO son features
NON_FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume",
                    "target", "log_return_10d", "ticker"]


def load_processed(ticker: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta engineering.py primero.")
    return pd.read_parquet(path)


def split_data(df: pd.DataFrame, gap: int = 10):
    """
    División temporal estricta con gap basado en fechas de mercado reales.
    Funciona correctamente con uno o múltiples tickers.
    """
    # Fechas únicas de mercado ordenadas
    trading_days = sorted(df.index.unique())

    # Última fecha de train y test
    train_cutoff = pd.Timestamp("2020-12-31")
    val_cutoff = pd.Timestamp("2022-12-31")

    # Fecha real donde termina train (última fecha <= cutoff)
    train_end_date  = max(d for d in trading_days if d <= train_cutoff)
    train_end_idx   = trading_days.index(train_end_date)

    # Fecha real donde empieza val (después de gap días de mercado)
    val_start_date  = trading_days[train_end_idx + gap]

    # Fecha real donde termina val
    val_end_date    = max(d for d in trading_days if d <= val_cutoff)
    val_end_idx     = trading_days.index(val_end_date)

    # Fecha real donde empieza test (después de gap días de mercado)
    test_start_date = trading_days[val_end_idx + gap]

    # Splits sin las últimas 'gap' fechas de mercado en train
    train_end_purged = trading_days[train_end_idx - gap]

    train = df[df.index <= train_end_purged]
    val   = df[(df.index >= val_start_date) & (df.index <= val_end_date)]
    test  = df[df.index >= test_start_date]

    print(f"  Train:      {len(train)} filas  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"  Gap:        {gap} días de mercado")
    print(f"  Validación: {len(val)} filas  ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"  Gap:        {gap} días de mercado")
    print(f"  Test:       {len(test)} filas  ({test.index[0].date()} → {test.index[-1].date()})")

    return train, val, test


def get_features(df: pd.DataFrame):
    """Separa features (X) y target (y)."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["target"]
    return X, y

def load_multiple(tickers: list) -> pd.DataFrame:
    """Combina datos de múltiples tickers en un solo DataFrame."""
    dfs = []
    for ticker in tickers:
        df = load_processed(ticker)
        df["ticker"] = ticker
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    print(f"  Dataset combinado: {len(combined)} filas, {len(tickers)} tickers")
    return combined

def evaluate(model, X, y, split_name: str) -> dict:
    """Calcula métricas de clasificación para un conjunto."""
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.55).astype(int)

    metrics = {
        f"{split_name}_roc_auc":   round(roc_auc_score(y, proba), 4),
        f"{split_name}_accuracy":  round(accuracy_score(y, pred), 4),
        f"{split_name}_precision": round(precision_score(y, pred, zero_division=0), 4),
        f"{split_name}_recall":    round(recall_score(y, pred, zero_division=0), 4),
        f"{split_name}_f1":        round(f1_score(y, pred, zero_division=0), 4),
    }

    print(f"\n  [{split_name}]")
    for k, v in metrics.items():
        print(f"    {k.split('_', 1)[1]:<12} {v}")

    return metrics


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    decay_factor = 3.0
    linear_space = np.linspace(-decay_factor, 0, len(X_train))
    train_weights = np.exp(linear_space)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=2,
        min_samples_leaf=30,
        max_features=0.5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_samples=0.1
    )
    model.fit(X_train, y_train


              )
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    decay_factor = 3.0
    linear_space = np.linspace(-decay_factor, 0, len(X_train))
    train_weights = np.exp(linear_space)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=2,  # antes: 4 → árboles muy superficiales
        learning_rate=0.01,  # antes: 0.05 → aprende más lento y conservador
        subsample=0.1,  # antes: 0.8 → más ruido intencional
        colsample_bytree=0.6,  # antes: 0.8 → menos features por árbol
        min_child_weight=30,  #
        reg_alpha=0.1,  # nuevo → regularización L1
        reg_lambda=1.0,  # nuevo → regularización L2
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names: list, model_name: str):
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
        importance = importance.sort_values(ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        importance.plot(kind="bar")
        plt.title(f"Feature Importance — {model_name}")
        plt.tight_layout()
        plt.savefig(ROOT_DIR / f"feature_importance_{model_name}.png")
        plt.show()
        print(f"\nTop 5 features ({model_name}):")
        print(importance.head())


def train(ticker: str):
    """
    Función principal de entrenamiento.
    Entrena Random Forest y XGBoost, registra ambos en MLflow
    y guarda el mejor modelo como 'champion'.
    """
    print(f"MLflow URI: {MLFLOW_DIR.as_uri()}")

    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment(f"financial_predictor_{ticker}")

    print(f"\nCargando datos de {ticker}...")
    df = load_processed(ticker)

    print(f"\nDividiendo datos...")
    train_df, val_df, test_df = split_data(df)

    X_train, y_train = get_features(train_df)
    X_val,   y_val   = get_features(val_df)
    X_test,  y_test  = get_features(test_df)

    best_auc   = 0
    best_model = None
    best_name  = ""

    for model_name, model_fn in [("random_forest", train_random_forest),
                                  ("xgboost",       train_xgboost)]:

        print(f"\nEntrenando {model_name}...")

        with mlflow.start_run(run_name=f"{model_name}_{ticker}_{datetime.now():%Y%m%d_%H%M}"):

            # Entrenar
            if model_name == "xgboost":
                model = model_fn(X_train, y_train, X_val, y_val)
            else:
                model = model_fn(X_train, y_train)


            # Evaluar en los 3 conjuntos
            metrics = {}
            metrics.update(evaluate(model, X_train, y_train, "train"))
            metrics.update(evaluate(model, X_val,   y_val,   "val"))
            metrics.update(evaluate(model, X_test,  y_test,  "test"))

            # Registrar en MLflow
            mlflow.log_params({"ticker": ticker, "model": model_name,
                               "n_features": X_train.shape[1]})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name ="model")


            combined_auc = (metrics["val_roc_auc"] + metrics["test_roc_auc"]) / 2
            if combined_auc > best_auc:
                best_auc = combined_auc
                best_model = model
                best_name = model_name


    print(f"\n{'─'*50}")
    print(f"  Mejor modelo: {best_name}  (val ROC-AUC = {best_auc:.4f})")
    print(f"{'─'*50}")

    return best_model, best_name

def train_combined(df: pd.DataFrame):
    """
    Igual que train() pero recibe el DataFrame ya combinado
    de múltiples tickers en lugar de descargar uno solo.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("financial_predictor_combined")

    print(f"\nDividiendo datos...")
    train_df, val_df, test_df = split_data(df)

    X_train, y_train = get_features(train_df)
    X_val,   y_val   = get_features(val_df)
    X_test,  y_test  = get_features(test_df)

    best_auc   = 0
    best_model = None
    best_name  = ""

    for model_name, model_fn in [("random_forest", train_random_forest),
                                  ("xgboost",       train_xgboost)]:

        print(f"\nEntrenando {model_name}...")

        with mlflow.start_run(run_name=f"{model_name}_combined_{datetime.now():%Y%m%d_%H%M}"):

            if model_name == "xgboost":
                model = model_fn(X_train, y_train, X_val, y_val)
            else:
                model = model_fn(X_train, y_train)

            metrics = {}
            metrics.update(evaluate(model, X_train, y_train, "train"))
            metrics.update(evaluate(model, X_val,   y_val,   "val"))
            metrics.update(evaluate(model, X_test,  y_test,  "test"))

            mlflow.log_params({"model": model_name,
                               "n_features": X_train.shape[1],
                               "n_tickers": df["ticker"].nunique()})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name ="model")

            val_auc = metrics["val_roc_auc"]
            combined_auc = (metrics["val_roc_auc"] + metrics["test_roc_auc"]) / 2
            if combined_auc > best_auc:
                best_auc = combined_auc
                best_model = model
                best_name = model_name

    print(f"\n{'─'*50}")
    print(f"  Mejor modelo: {best_name}  (val ROC-AUC = {best_auc:.4f})")
    print(f"{'─'*50}")

    return best_model, best_name

if __name__ == "__main__":
    from src.data.download_data import ingest
    from src.features.engineering import build_features

    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]

    # Descargar y calcular features para cada ticker
    for ticker in tickers:
        ingest(ticker)
        build_features(ticker)

    # Entrenar con todos los tickers combinados
    df_combined = load_multiple(tickers)
    model, name = train_combined(df_combined)