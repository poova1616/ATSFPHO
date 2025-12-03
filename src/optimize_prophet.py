import optuna
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from src.cross_validation import rolling_cv

def objective_factory(data, train_size=700, horizon=30):
    def objective(trial):
        params = {
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.01, 0.5),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 1.0, 20.0),
            'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.1, 10.0),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive','multiplicative']),
            'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95)
        }

        rmses = []
        for train_df, val_df in rolling_cv(data, train_size=train_size, horizon=horizon):
            m = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                changepoint_range=params['changepoint_range'],
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False
            )
            m.fit(train_df)
            future = m.make_future_dataframe(periods=horizon, freq='D')
            forecast = m.predict(future)
            preds = forecast.iloc[-horizon:]['yhat'].values
            true = val_df['y'].values
            rmses.append(np.sqrt(mean_squared_error(true, preds)))
        return float(np.mean(rmses))
    return objective

def run_optuna(data, n_trials=30, train_size=700, horizon=30):
    objective = objective_factory(data, train_size=train_size, horizon=horizon)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study
