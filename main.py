import pandas as pd
import json
from src.dataset_generator import generate_synthetic_timeseries
from src.optimize_prophet import run_optuna
from src.train_final_model import train_final
from src.baseline_arima import baseline_arima

def main():
    print('Step 1: Generate or load dataset')
    df = generate_synthetic_timeseries(days=1095, out_path='data/synthetic_timeseries.csv')
    print(f'Dataset rows: {len(df)}')

    print('Step 2: Run Optuna optimization (this may take several minutes)...')
    # For speed in automated runs set n_trials lower (e.g. 10). Increase for better results.
    study = run_optuna(df, n_trials=20, train_size=700, horizon=30)
    best_params = study.best_params
    print('Best params found:', best_params)

    print('Step 3: Train final Prophet model on full data')
    model, forecast = train_final(df, best_params, periods=90)

    print('Step 4: Generate ARIMA baseline forecast')
    arima_out = baseline_arima(df, order=(5,1,2), horizon=90)

    print('Outputs saved to outputs/ directory.')

if __name__ == '__main__':
    main()
