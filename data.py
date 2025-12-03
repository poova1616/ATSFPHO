import yfinance as yf
import pandas as pd

# Fetch Bitcoin data (BTC-USD ticker on Yahoo Finance)
btc = yf.download("BTC-USD", start="2022-12-01", end="2025-12-01")

# Prepare for Prophet
df = btc.reset_index()[['Date','Close']]
df.rename(columns={'Date':'ds','Close':'y'}, inplace=True)

# Save to CSV
df.to_csv("bitcoin_last_3_year.csv", index=False)
print(df.head())
