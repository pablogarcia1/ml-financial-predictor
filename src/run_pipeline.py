import pandas as pd
from pathlib import Path

# Importamos tus módulos personalizados
from src.data.download_data import ingest
from src.features.engineering import build_features, load_processed
from src.models.train import split_data, get_features, train_xgboost
from src.backtest.engine import run_backtest  # Asumiendo que guardaste tu código anterior aquí


def run_full_pipeline():
    print("Iniciando Pipeline de Machine Learning Financiero...")
    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]

    # ---------------------------------------------------------
    # ETAPA 1: Ingesta de Datos (Data Engineering)
    # ---------------------------------------------------------
    print("\n[1/4] Descargando datos más recientes...")
    for ticker in tickers:
        # Asumiendo que 'ingest' baja de Yahoo Finance y guarda en data/raw
        ingest(ticker)
        print(f"  ✓ Datos crudos de {ticker} actualizados.")

    # ---------------------------------------------------------
    # ETAPA 2: Feature Engineering (Procesamiento)
    # ---------------------------------------------------------
    print("\n[2/4] Calculando indicadores y transformando datos...")
    dfs = []
    for ticker in tickers:
        # Asumiendo que 'build_features' lee de raw, procesa y guarda en data/processed
        build_features(ticker)

        # Cargamos el archivo .parquet recién procesado
        df = load_processed(ticker)
        df["ticker"] = ticker
        dfs.append(df)
        print(f"  ✓ Features generados para {ticker}.")

    # Unimos todo
    df_combined = pd.concat(dfs).sort_index()

    # ---------------------------------------------------------
    # ETAPA 3: Entrenamiento del Modelo (Machine Learning)
    # ---------------------------------------------------------
    print("\n[3/4] Entrenando modelo XGBoost...")
    train_df, val_df, test_df = split_data(df_combined)

    X_train, y_train = get_features(train_df)
    X_val, y_val = get_features(val_df)

    # Entrenamos con la data fresca
    model = train_xgboost(X_train, y_train, X_val, y_val)
    print("  ✓ Modelo re-entrenado exitosamente con datos actuales.")

    # ---------------------------------------------------------
    # ETAPA 4: Backtesting y Evaluación (Business Intelligence)
    # ---------------------------------------------------------
    print("\n[4/4] Ejecutando Backtest sobre SPY...")
    test_spy = test_df[test_df["ticker"] == "SPY"]

    print(f"  Periodo de evaluación: {test_spy.index[0].date()} → {test_spy.index[-1].date()}")

    # Corremos el motor de backtest (ya con la corrección de superposición)
    results = run_backtest(model, test_spy, buy_threshold=0.52, sell_threshold=0.45, horizon=10)

    print("\nPipeline ejecutado correctamente al 100%.")


if __name__ == "__main__":
    run_full_pipeline()