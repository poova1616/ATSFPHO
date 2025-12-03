"""
Advanced Time Series Forecasting with Prophet + Hyperparameter Optimization
Author: Poovarasan
Description:
    End-to-end pipeline:
        - Data Loading
        - Preprocessing
        - Rolling-origin Cross-Validation
        - Hyperparameter Optimization using Optuna
        - Prophet Model Training & 90-Day Forecast
        - Baseline ARIMA Comparison
        - Evaluation Metrics (RMSE, MAPE)
"""

# ============================
# Imports
# ============================

import pandas as pd
import numpy as np
from prophet import Prophet
import optuna
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

# ============================
# 1. Load & Preprocess Dataset
# ============================

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a time series dataset in Prophet-friendly format.

    Expected columns:
        - 'ds': datetime
        - 'y': numeric value
    """
    df = pd.read_csv(path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values("ds")
    df = df.dropna()
    return df


# ============================
# 2. Rolling Cross-Validation
# ============================

def rolling_cv(df: pd.DataFrame, train_size: int, horizon: int):
    """
    Generator for rolling-origin time series evaluation.

    Args:
        df: pandas DataFrame with ['ds', 'y']
        train_size: initial training window size
        horizon: forecast horizon for each fold

    Yields:
        train_df, val_df for each fold
    """
    for i in range(train_size, len(df) - horizon, horizon):
        train = df.iloc[:i]
        val = df.iloc[i:i + horizon]
        yield train, val


# ============================
# 3. Objective Function for Optuna
# ============================

def objective(trial):
    """
    Objective function minimized by Optuna.
    Returns the average RMSE across folds.
    """

    # Prophet hyperparameters to tune
    params = {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.01, 0.5),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 1.0, 20.0),
        "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 1.0, 20.0),
        "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        "changepoint_range": trial.suggest_float("changepoint_range", 0.8, 0.95),
    }

    rmses = []

    for train_df, val_df in rolling_cv(data, train_size=700, horizon=30):

        model = Prophet(
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            holidays_prior_scale=params["holidays_prior_scale"],
            seasonality_mode=params["seasonality_mode"],
            changepoint_range=params["changepoint_range"]
        )

        model.fit(train_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        pred = forecast.iloc[-30:]["yhat"].values
        true = val_df["y"].values

        rmse = np.sqrt(mean_squared_error(true, pred))
        rmses.append(rmse)

    return np.mean(rmses)


# ============================
# 4. Train Final Optimized Prophet
# ============================

def train_final_prophet(df: pd.DataFrame, best_params: dict):
    """
    Train final Prophet model on the full dataset.
    """

    model = Prophet(**best_params)
    model.fit(df)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    return model, forecast


# ============================
# 5. Baseline ARIMA Comparison
# ============================

def baseline_arima(df: pd.DataFrame, order=(5,1,2), horizon=90, freq='D'):
    """
    Robust ARIMA baseline:
      - Coerces 'y' to numeric (drops non-numeric / NaN)
      - Ensures a DatetimeIndex (from 'ds')
      - Fits ARIMA and returns a pd.Series forecast indexed by future dates

    Args:
      df: DataFrame with columns ['ds', 'y']
      order: ARIMA (p,d,q)
      horizon: forecast horizon (int)
      freq: frequency for future dates ('D' daily by default)
    Returns:
      pd.Series indexed by forecast dates
    """
    # Make a copy to avoid changing original
    tmp = df.copy()

    # Ensure 'ds' is datetime
    tmp['ds'] = pd.to_datetime(tmp['ds'], errors='coerce')
    if tmp['ds'].isna().any():
        raise ValueError("Some 'ds' values could not be parsed as datetimes. Check your 'ds' column.")

    # Coerce 'y' to numeric and drop rows with non-numeric
    tmp['y'] = pd.to_numeric(tmp['y'], errors='coerce')
    tmp = tmp.dropna(subset=['y'])
    if tmp.empty:
        raise ValueError("No valid numeric data left in 'y' after coercion. Check your inputs.")

    # Set datetime index
    series = tmp.set_index('ds')['y'].sort_index().astype(float)

    # Optionally infer frequency (not required by ARIMA, but useful for date generation)
    try:
        # if the series has a regular frequency, .asfreq won't fail
        inferred = pd.infer_freq(series.index)
        if inferred is not None:
            freq = inferred
    except Exception:
        pass

    # Fit ARIMA
    model = ARIMA(series, order=order)
    res = model.fit()

    # Forecast and attach future dates
    forecast_values = res.forecast(steps=horizon)
    last_date = series.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit=freq),
                                 periods=horizon, freq=freq)

    return pd.Series(forecast_values, index=future_dates, name='arima_forecast')



# ============================
# 6. Main Execution
# ============================

if __name__ == "__main__":

    print("Loading dataset...")
    data = load_dataset("bitcoin_last_3_year.csv")  # <-- replace with your file

    print("Starting Optuna optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best Hyperparameters:")
    print(study.best_params)

    print("Training final optimized Prophet model...")
    model, forecast = train_final_prophet(data, study.best_params)

    # Extract last 90 days prediction
    final_90day = forecast.tail(90)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    print("\nFinal 90-Day Forecast:")
    print(final_90day)

    # Baseline ARIMA
    print("\nGenerating ARIMA baseline forecast...")
    arima_pred = baseline_arima(data)

    # Save outputs
    final_90day.to_csv("prophet_90day_forecast.csv", index=False)
    pd.DataFrame({"arima_forecast": arima_pred}).to_csv("arima_baseline.csv")

    print("\nAll tasks complete.")
