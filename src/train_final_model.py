import json
from prophet import Prophet
import pandas as pd

def train_final(data, best_params, periods=90, out_csv='outputs/prophet_90day_forecast.csv'):
    # Ensure types
    data = data.copy()
    data['ds'] = pd.to_datetime(data['ds'])
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data = data.dropna(subset=['y'])
    data = data.sort_values('ds').reset_index(drop=True)

    model = Prophet(
        changepoint_prior_scale=best_params.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=best_params.get('seasonality_prior_scale', 10.0),
        holidays_prior_scale=best_params.get('holidays_prior_scale', 1.0),
        seasonality_mode=best_params.get('seasonality_mode', 'additive'),
        changepoint_range=best_params.get('changepoint_range', 0.9),
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(data)

    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)

    out = forecast[['ds','yhat','yhat_lower','yhat_upper']]
    out.to_csv(out_csv, index=False)

    # save best params
    with open('outputs/best_params.json','w') as f:
        json.dump(best_params, f, indent=2)

    return model, forecast
