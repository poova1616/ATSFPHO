import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_bitcoin_data(start="2022-12-01", end="2025-12-01"):
    btc = yf.download("BTC-USD", start=start, end=end, auto_adjust=False)
    if btc.empty:
        raise ValueError("No data fetched from Yahoo Finance.")

    df = btc.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if 'Date' not in df.columns:
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        if not date_col:
            raise ValueError("No valid date column found.")
        df.rename(columns={date_col: 'Date'}, inplace=True)

    price_col = 'Close' if 'Close' in df.columns else 'Adj Close'
    df = df[['Date', price_col]].rename(columns={'Date': 'ds', price_col: 'y'})
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y'])

    return df


def objective(trial, df):
    """
    Prophet CV with:
    - Initial: 365 days
    - Period: 180 days
    - Horizon: 90 days
    """
    params = {
        'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
        'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True),
        'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True)
    }

    model = Prophet(**params, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='90 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    return df_p['rmse'].mean()


def optimize_hyperparameters(df, n_trials=50):
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)
    logging.info("Best RMSE: %.4f", study.best_value)
    logging.info("Best parameters: %s", study.best_params)
    return study.best_params


def summarize_hyperparameters(best_params):
    logging.info("Hyperparameter Tuning Summary:")
    logging.info("- Search space:")
    logging.info("  changepoint_prior_scale: [0.001, 0.5] (log)")
    logging.info("  seasonality_prior_scale: [0.01, 10.0]")
    logging.info("  holidays_prior_scale: [0.01, 10.0]")
    logging.info("- Optimization: Optuna (Bayesian TPE)")
    logging.info("- Objective: Minimize RMSE via Prophet CV")
    for k, v in best_params.items():
        logging.info(f"  {k}: {v:.4f}")


def train_final_model(df, best_params):
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
    future = model.make_future_dataframe(periods=periods)
    forecast_df = model.predict(future)
    return forecast_df


def evaluate_baseline(df):
    df = df.copy()
    df['yhat_naive'] = df['y'].shift(1)
    df.dropna(inplace=True)
    rmse = np.sqrt(mean_squared_error(df['y'], df['yhat_naive']))
    mape = mean_absolute_percentage_error(df['y'], df['yhat_naive'])
    return {'rmse': rmse, 'mape': mape}


def compare_models(df, forecast_df):
    actual = df.set_index('ds')['y']
    predicted = forecast_df.set_index('ds')['yhat']
    common_dates = actual.index.intersection(predicted.index)
    rmse = np.sqrt(mean_squared_error(actual.loc[common_dates], predicted.loc[common_dates]))
    mape = mean_absolute_percentage_error(actual.loc[common_dates], predicted.loc[common_dates])
    return {'rmse': rmse, 'mape': mape}


def summarize_comparison(baseline, prophet):
    logging.info("Model Comparison:")
    logging.info(f"- Baseline RMSE: {baseline['rmse']:.2f}, MAPE: {baseline['mape']:.2f}")
    logging.info(f"- Prophet  RMSE: {prophet['rmse']:.2f}, MAPE: {prophet['mape']:.2f}")
    logging.info(f"➡ RMSE improvement: {baseline['rmse'] - prophet['rmse']:.2f}")
    logging.info(f"➡ MAPE improvement: {baseline['mape'] - prophet['mape']:.2f}")


def interpret_forecast(forecast_df):
    latest = forecast_df.tail(1)
    ds = latest['ds'].values[0]
    yhat = latest['yhat'].values[0]
    lower = latest['yhat_lower'].values[0]
    upper = latest['yhat_upper'].values[0]
    logging.info(f"Forecast for {ds}: {yhat:.2f} USD")
    logging.info(f"Confidence interval: [{lower:.2f}, {upper:.2f}] USD")
    logging.info("Short-term forecasts are more reliable. Long-term forecasts show wider intervals and higher uncertainty.")
    logging.info("Use forecasts for planning, not precise decisions.")


def write_text_report(output_dir, best_params, baseline, prophet, forecast_df):
    path = os.path.join(output_dir, "project_summary.txt")
    with open(path, "w") as f:
        f.write("PROJECT REPORT\n==============\n\n")
        f.write("1. Hyperparameter Optimization:\n")
        f.write("Search space: changepoint_prior_scale [0.001–0.5], seasonality_prior_scale [0.01–10], holidays_prior_scale [0.01–10]\n")
        f.write("Method: Optuna (Bayesian TPE)\nBest parameters:\n")
        for k, v in best_params.items():
            f.write(f"  - {k}: {v:.4f}\n")

        f.write("\n2. Model Comparison:\n")
        f.write(f"Baseline RMSE: {baseline['rmse']:.2f}, MAPE: {baseline['mape']:.2f}\n")
        f.write(f"Prophet RMSE: {prophet['rmse']:.2f}, MAPE: {prophet['mape']:.2f}\n")

        f.write("\n3. Forecast Interpretation:\n")
        latest = forecast_df.tail(1)
        ds = latest['ds'].values[0]
        yhat = latest['yhat'].values[0]
        lower = latest['yhat_lower'].values[0]
        upper = latest['yhat_upper'].values[0]
        f.write(f"Forecast for {ds}: {yhat:.2f} USD\n")
        f.write(f"Confidence interval: [{lower:.2f}, {upper:.2f}] USD\n")
        f.write("Short-term forecasts are more reliable. Long-term forecasts show higher uncertainty.\n")
    logging.info("Saved text report: %s", path)


def main():
    output_dir = "forecast_outputs"
    ensure_dir(output_dir)

    df = load_bitcoin_data()
    best_params = optimize_hyperparameters(df)
    summarize_hyperparameters(best_params)

    model = train_final_model(df, best_params)
    forecast_df = forecast(model, periods=90)

    baseline = evaluate_baseline(df)
    prophet = compare_models(df, forecast_df)

    summarize_comparison(baseline, prophet)
    interpret_forecast(forecast_df)
    write_text_report(output_dir, best_params, baseline, prophet, forecast_df)


if __name__ == "__main__":
    main()
