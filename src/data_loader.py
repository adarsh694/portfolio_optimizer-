# fetch + cache price data
import yfinance as yf
import pandas as pd
import os

def fetch_prices(tickers, start="2019-01-01", end="2024-01-01", cache_path="data/prices.csv"):
    """
    Downloads adjusted close prices for given tickers.
    Caches to CSV so you're not hitting the API every run.
    """
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    data = data.dropna()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    data.to_csv(cache_path)
    return data