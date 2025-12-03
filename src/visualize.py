import matplotlib.pyplot as plt
import pandas as pd

def plot_prophet_forecast(forecast_csv='outputs/prophet_90day_forecast.csv'):
    df = pd.read_csv(forecast_csv)
    df['ds'] = pd.to_datetime(df['ds'])
    plt.figure(figsize=(12,6))
    plt.plot(df['ds'], df['yhat'], label='yhat')
    plt.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'], alpha=0.3, label='interval')
    plt.title('Prophet 90-day Forecast')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_arima_vs_prophet(arima_csv='outputs/arima_baseline.csv', prophet_csv='outputs/prophet_90day_forecast.csv'):
    a = pd.read_csv(arima_csv); a['ds']=pd.to_datetime(a['ds'])
    p = pd.read_csv(prophet_csv); p['ds']=pd.to_datetime(p['ds'])
    # align by ds intersection
    merged = pd.merge(p, a, on='ds', how='inner', suffixes=('_prophet','_arima'))
    if merged.empty:
        print('No overlapping dates to compare.')
        return
    plt.figure(figsize=(12,6))
    plt.plot(merged['ds'], merged['yhat'], label='Prophet yhat')
    plt.plot(merged['ds'], merged['forecast'], label='ARIMA forecast')
    plt.legend()
    plt.tight_layout()
    plt.show()
