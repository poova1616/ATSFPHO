import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def baseline_arima(df, order=(5,1,2), horizon=90, out_csv='outputs/arima_baseline.csv'):
    tmp = df.copy()
    tmp['ds'] = pd.to_datetime(tmp['ds'], errors='coerce')
    tmp['y'] = pd.to_numeric(tmp['y'], errors='coerce')
    tmp = tmp.dropna(subset=['ds','y'])
    tmp = tmp.sort_values('ds').reset_index(drop=True)
    if tmp.empty:
        raise ValueError('No valid data for ARIMA baseline.')

    series = tmp.set_index('ds')['y'].astype(float)

    model = SARIMAX(series, order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    res = model.fit(disp=False)

    pred_res = res.get_forecast(steps=horizon)
    mean = pred_res.predicted_mean
    conf = pred_res.conf_int()

    last_date = series.index.max()
    freq = pd.infer_freq(series.index)
    if freq is None:
        freq = 'D'

    future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                                 periods=horizon, freq=freq)

    out = pd.DataFrame({
        'ds': future_dates,
        'forecast': mean.values,
        'lower': conf.iloc[:,0].values,
        'upper': conf.iloc[:,1].values
    })
    out.to_csv(out_csv, index=False)
    return out
