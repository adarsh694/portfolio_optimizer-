import yfinance as yf
import pandas as pd
import os

def fetch_prices(tickers, start="2019-01-01", end="2024-01-01", cache_path="data/prices.csv"):
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not cached.empty:
            return cached

    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    data = data.dropna()

    if data.empty:
        raise ValueError("Download returned no data — check tickers/date range/network")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    data.to_csv(cache_path)
    return data