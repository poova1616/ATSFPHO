import yfinance as yf
import pandas as pd
import numpy as np
import logging
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import optuna

logging.basicConfig(level=logging.INFO)

def load_bitcoin_data(start="2022-12-01", end="2025-12-01"):
    """
    Load Bitcoin price data from Yahoo Finance and prepare it for Prophet.
    """
    btc = yf.download("BTC-USD", start=start, end=end, auto_adjust=False)
    if btc.empty:
        raise ValueError("No data fetched from Yahoo Finance. Check ticker or date range.")

    df = btc.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    logging.info("Flattened columns: %s", df.columns.tolist())

    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError(f"Expected 'Date' and 'Close' columns, got: {df.columns.tolist()}")

    df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y'])

    return df


def objective(trial, df):
    """
    Objective function for Optuna hyperparameter optimization.
    """
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
    df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='90 days')
    df_p = performance_metrics(df_cv, rolling_window=1)

    return df_p['rmse'].mean()


def optimize_hyperparameters(df, n_trials=50):
    """
    Run Optuna optimization to find best hyperparameters.
    """
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)
    return study.best_params


def train_final_model(df, best_params):
    """
    Train the final Prophet model with optimized hyperparameters.
    """
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
    """
    Generate forecast for the next specified periods.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


def evaluate_baseline(df):
    """
    Evaluate a simple baseline model (naive forecast).
    """
    df = df.copy()
    df['yhat_naive'] = df['y'].shift(1)
    df.dropna(inplace=True)

    mse = mean_squared_error(df['y'], df['yhat_naive'])
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(df['y'], df['yhat_naive'])

    return {'rmse': rmse, 'mape': mape}


def interpret_forecast(forecast_df):
    """
    Interpret the forecast intervals and reliability.
    """
    latest = forecast_df.tail(1)
    ds = latest['ds'].values[0]
    yhat = latest['yhat'].values[0]
    lower = latest['yhat_lower'].values[0]
    upper = latest['yhat_upper'].values[0]

    logging.info(f"Forecast for {ds}: {yhat:.2f} USD")
    logging.info(f"Confidence interval: [{lower:.2f}, {upper:.2f}] USD")
    logging.info("Interpretation: Wider intervals indicate higher uncertainty. Use with caution in business decisions.")


def compare_models(df, forecast_df):
    """
    Compare Prophet forecast with naive baseline.
    """
    actual = df.set_index('ds')['y']
    predicted = forecast_df.set_index('ds')['yhat']
    common_dates = actual.index.intersection(predicted.index)

    rmse = np.sqrt(mean_squared_error(actual.loc[common_dates], predicted.loc[common_dates]))
    mape = mean_absolute_percentage_error(actual.loc[common_dates], predicted.loc[common_dates])

    logging.info(f"Prophet RMSE: {rmse:.2f}")
    logging.info(f"Prophet MAPE: {mape:.2f}")


if __name__ == '__main__':
    df = load_bitcoin_data()
    best_params = optimize_hyperparameters(df, n_trials=50)
    logging.info('Best hyperparameters: %s', best_params)

    final_model = train_final_model(df, best_params)
    forecast_df = forecast(final_model, periods=90)

    logging.info("Final 5-day forecast:")
    logging.info("\n%s", forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    interpret_forecast(forecast_df)

    baseline_metrics = evaluate_baseline(df)
    logging.info('Baseline RMSE: %.2f', baseline_metrics['rmse'])
    logging.info('Baseline MAPE: %.2f', baseline_metrics['mape'])

    compare_models(df, forecast_df)
