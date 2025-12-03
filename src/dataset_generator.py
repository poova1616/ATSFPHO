import pandas as pd
import numpy as np

def generate_synthetic_timeseries(days=1095, out_path='data/synthetic_timeseries.csv'):
    np.random.seed(42)

    ds = pd.date_range(start="2020-01-01", periods=days, freq="D")

    # Components
    trend = 0.02 * np.arange(days)                        # gentle linear trend
    weekly = 3 * np.sin(2 * np.pi * np.arange(days) / 7)  # weekly seasonality
    yearly = 10 * np.sin(2 * np.pi * np.arange(days) / 365)  # yearly seasonality
    noise = np.random.normal(0, 2, days)

    # Random event spikes (like promotions/holidays)
    events = np.zeros(days)
    spikes = np.random.choice(days, 12, replace=False)
    events[spikes] = np.random.randint(8, 30, len(spikes))

    y = 50 + trend + weekly + yearly + events + noise

    df = pd.DataFrame({"ds": ds, "y": y})
    df.to_csv(out_path, index=False)
    return df

if __name__ == '__main__':
    df = generate_synthetic_timeseries()
    print(f"Saved synthetic dataset with {len(df)} rows to data/synthetic_timeseries.csv")
