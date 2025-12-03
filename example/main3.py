import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def load_bitcoin_data(start="2022-12-01", end="2025-12-01"):
    btc = yf.download("BTC-USD", start=start, end=end, auto_adjust=False)

    if btc.empty:
        raise ValueError("No data fetched from Yahoo Finance. Check ticker or date range.")

    df = btc.reset_index()

    # Flatten MultiIndex columns if present
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    print("Flattened columns:", df.columns.tolist())

    # Ensure required columns exist
    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError(f"Expected 'Date' and 'Close' columns, got: {df.columns.tolist()}")

    # Rename for Prophet
    df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Ensure numeric target
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y'])

    return df



def objective(trial, df):
    """Objective function for Optuna hyperparameter optimization."""
    changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
    seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True)
    holidays_prior_scale = trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True)

    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        yearly_seasonality=True,
        weekly_seasonality=True
    )

    model.fit(df)

    # Cross-validation
    df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='90 days')
    df_p = performance_metrics(df_cv, rolling_window=1)

    # Use mean RMSE as the metric to minimize
    return df_p['rmse'].mean()


def optimize_hyperparameters(df, n_trials=50):
    """Run Optuna optimization to find best hyperparameters."""
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)
    return study.best_params


def train_final_model(df, best_params):
    """Train the final Prophet model with optimized hyperparameters."""
    model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        holidays_prior_scale=best_params['holidays_prior_scale'],
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(df)
    return model


def forecast(model, periods=90):
    """Generate forecast for the next specified periods."""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

import numpy as np

def evaluate_baseline(df):
    """Evaluate a simple baseline model (naive forecast)."""
    df = df.copy()
    df['yhat_naive'] = df['y'].shift(1)
    df.dropna(inplace=True)

    # Compute RMSE manually for compatibility
    mse = mean_squared_error(df['y'], df['yhat_naive'])
    rmse = np.sqrt(mse)

    # MAPE works fine
    mape = mean_absolute_percentage_error(df['y'], df['yhat_naive'])

    return {'rmse': rmse, 'mape': mape}

# def evaluate_baseline(df):
#     """Evaluate a simple baseline model (naive forecast)."""
#     df = df.copy()
#     df['yhat_naive'] = df['y'].shift(1)
#     df.dropna(inplace=True)
#     rmse = mean_squared_error(df['y'], df['yhat_naive'], squared=False)
#     mape = mean_absolute_percentage_error(df['y'], df['yhat_naive'])
#     return {'rmse': rmse, 'mape': mape}


if __name__ == '__main__':
    # Load Bitcoin data
    df = load_bitcoin_data()

    # Optimize hyperparameters
    best_params = optimize_hyperparameters(df, n_trials=50)
    print('Best hyperparameters:', best_params)

    # Train final model
    final_model = train_final_model(df, best_params)

    # Generate forecast
    forecast_df = forecast(final_model, periods=90)
    print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Evaluate baseline
    baseline_metrics = evaluate_baseline(df)
    print('Baseline RMSE:', baseline_metrics['rmse'])
    print('Baseline MAPE:', baseline_metrics['mape'])
