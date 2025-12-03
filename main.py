"""
Project: Advanced Time Series Forecasting with Prophet
Author: POOVARASAN
Version: 1.1
Description: Forecasting Bitcoin prices using Prophet with Optuna-based hyperparameter tuning,
             plus visualization of forecasts, components, and model comparison.
"""

import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly  # optional if you want interactive plots
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import optuna

# -------- Logging setup --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# -------- Data loading --------
def load_bitcoin_data(start="2022-12-01", end="2025-12-01"):
    """
    Load Bitcoin price data from Yahoo Finance and prepare it for Prophet.
    - Flattens possible MultiIndex columns from yfinance
    - Returns a DataFrame with columns: ds (date), y (target)
    """
    btc = yf.download("BTC-USD", start=start, end=end, auto_adjust=False)
    if btc.empty:
        raise ValueError("No data fetched from Yahoo Finance. Check ticker or date range.")

    df = btc.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    logging.info("Flattened columns: %s", df.columns.tolist())

    if 'Date' not in df.columns:
        # some environments may use 'Datetime' or similar; fallback to the first datetime-like column
        date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
        if date_col is None:
            raise ValueError(f"Could not find a date column. Got: {df.columns.tolist()}")
        df.rename(columns={date_col: 'Date'}, inplace=True)

    # Prefer Close; fallback to Adj Close if Close is missing
    price_col = 'Close' if 'Close' in df.columns else ('Adj Close' if 'Adj Close' in df.columns else None)
    if price_col is None:
        raise ValueError(f"Expected 'Close' or 'Adj Close' column, got: {df.columns.tolist()}")

    df = df[['Date', price_col]].rename(columns={'Date': 'ds', price_col: 'y'})
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y'])

    return df


# -------- Hyperparameter optimization --------
def objective(trial, df):
    """
    Objective function for Optuna hyperparameter optimization.
    Minimizes mean RMSE across Prophet cross-validation folds.
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


def optimize_hyperparameters(df, n_trials=50, seed=42):
    """
    Runs Optuna to find best Prophet hyperparameters.
    Returns a dict of best parameters.
    """
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)
    logging.info("Optuna best value (mean RMSE): %.4f", study.best_value)
    logging.info("Optuna best params: %s", study.best_params)
    return study.best_params


# -------- Model training and forecasting --------
def train_final_model(df, best_params):
    """
    Train a Prophet model with optimized hyperparameters.
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


def forecast(model, periods=90, freq='D'):
    """
    Generate forecast for the next 'periods' steps.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast_df = model.predict(future)
    return forecast_df


# -------- Baseline and comparison --------
def evaluate_baseline(df):
    """
    Evaluate a naive baseline: yhat[t] = y[t-1].
    Returns RMSE and MAPE.
    """
    df = df.copy()
    df['yhat_naive'] = df['y'].shift(1)
    df.dropna(inplace=True)

    mse = mean_squared_error(df['y'], df['yhat_naive'])
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(df['y'], df['yhat_naive'])
    return {'rmse': rmse, 'mape': mape}


def compare_models(df, forecast_df):
    """
    Compare Prophet forecast against actuals on overlapping dates.
    Returns RMSE and MAPE for Prophet.
    """
    actual = df.set_index('ds')['y']
    predicted = forecast_df.set_index('ds')['yhat']
    common_dates = actual.index.intersection(predicted.index)

    rmse = np.sqrt(mean_squared_error(actual.loc[common_dates], predicted.loc[common_dates]))
    mape = mean_absolute_percentage_error(actual.loc[common_dates], predicted.loc[common_dates])

    return {'rmse': rmse, 'mape': mape}


# -------- Visualization helpers --------
def ensure_dir(path):
    """
    Ensure a directory exists.
    """
    os.makedirs(path, exist_ok=True)


def plot_forecast_matplotlib(df, forecast_df, save_path=None):
    """
    Plot actuals and Prophet forecast with uncertainty intervals using matplotlib.
    """
    plt.figure(figsize=(12, 6))
    # Actuals
    plt.plot(df['ds'], df['y'], color='black', linewidth=1.5, label='Actual')
    # Forecast yhat
    plt.plot(forecast_df['ds'], forecast_df['yhat'], color='tab:blue', linewidth=1.5, label='Prophet forecast')
    # Uncertainty band
    plt.fill_between(
        forecast_df['ds'],
        forecast_df['yhat_lower'],
        forecast_df['yhat_upper'],
        color='tab:blue',
        alpha=0.2,
        label='Uncertainty interval'
    )
    plt.title('BTC-USD forecast with uncertainty (Prophet)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logging.info("Saved forecast plot: %s", save_path)
    plt.close()


def plot_components_matplotlib(model, forecast_df, save_dir=None):
    """
    Plot Prophet components (trend, weekly, yearly) via matplotlib.
    """
    fig = model.plot_components(forecast_df)
    fig.set_size_inches(12, 8)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "components.png")
        fig.savefig(path, dpi=150)
        logging.info("Saved components plot: %s", path)
    plt.close(fig)


def plot_model_comparison(baseline_metrics, prophet_metrics, save_path=None):
    """
    Bar chart comparing RMSE and MAPE for baseline vs Prophet.
    """
    metrics = ['RMSE', 'MAPE']
    baseline_vals = [baseline_metrics['rmse'], baseline_metrics['mape']]
    prophet_vals = [prophet_metrics['rmse'], prophet_metrics['mape']]

    data = pd.DataFrame({
        'Metric': metrics,
        'Baseline': baseline_vals,
        'Prophet': prophet_vals
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=data.melt(id_vars='Metric', var_name='Model', value_name='Value'),
                x='Metric', y='Value', hue='Model', palette='Set2')
    plt.title('Model comparison: Baseline vs Prophet')
    plt.ylabel('Error')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logging.info("Saved model comparison plot: %s", save_path)
    plt.close()


def interpret_forecast(forecast_df):
    """
    Log a concise interpretation for the final forecasted day.
    """
    latest = forecast_df.tail(1)
    ds = latest['ds'].values[0]
    yhat = latest['yhat'].values[0]
    lower = latest['yhat_lower'].values[0]
    upper = latest['yhat_upper'].values[0]

    logging.info(f"Forecast for {pd.to_datetime(ds).date()}: {yhat:,.2f} USD")
    logging.info(f"Confidence interval: [{lower:,.2f}, {upper:,.2f}] USD")
    logging.info("Interpretation: Intervals widen over time, reflecting higher uncertainty. "
                 "Use short-horizon forecasts for decisions; pair long-horizon forecasts with risk controls.")


# -------- Main entrypoint --------
def main():
    # Config
    start = "2022-12-01"
    end = "2025-12-01"
    n_trials = 50
    output_dir = "forecast_outputs"
    ensure_dir(output_dir)

    # Pipeline
    df = load_bitcoin_data(start=start, end=end)

    best_params = optimize_hyperparameters(df, n_trials=n_trials)
    model = train_final_model(df, best_params)
    forecast_df = forecast(model, periods=90)

    # Metrics
    baseline_metrics = evaluate_baseline(df)
    prophet_metrics = compare_models(df, forecast_df)

    logging.info("Baseline RMSE: %.2f | MAPE: %.2f", baseline_metrics['rmse'], baseline_metrics['mape'])
    logging.info("Prophet  RMSE: %.2f | MAPE: %.2f", prophet_metrics['rmse'], prophet_metrics['mape'])

    # Interpretation
    interpret_forecast(forecast_df)

    # Visualizations
    plot_forecast_matplotlib(df, forecast_df, save_path=os.path.join(output_dir, "forecast.png"))
    plot_components_matplotlib(model, forecast_df, save_dir=output_dir)
    plot_model_comparison(
        baseline_metrics,
        prophet_metrics,
        save_path=os.path.join(output_dir, "model_comparison.png")
    )

    # Save key tables
    tail_forecast = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
    tail_path = os.path.join(output_dir, "forecast_tail.csv")
    tail_forecast.to_csv(tail_path, index=False)
    logging.info("Saved forecast tail: %s", tail_path)

    params_path = os.path.join(output_dir, "best_params.json")
    pd.Series(best_params).to_json(params_path)
    logging.info("Saved best params: %s", params_path)


if __name__ == "__main__":
    main()
