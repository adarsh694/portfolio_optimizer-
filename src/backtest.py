import numpy as np
import pandas as pd
from src.stats import annualized_mean_returns, annualized_covariance
from src.optimizer import max_sharpe_portfolio

def rolling_backtest(prices: pd.DataFrame, log_returns: pd.DataFrame,
                      lookback=252, rebalance_freq=21, risk_free_rate=0.05):
    """
    Walk-forward backtest. At each rebalance date, uses only trailing
    `lookback` days of returns to pick weights — no future data leakage.
    """
    dates = log_returns.index
    n_days = len(dates)
    tickers = log_returns.columns.tolist()

    portfolio_value = [1.0]      # starts at $1, normalized
    equal_weight_value = [1.0]
    dates_used = [dates[lookback]]

    current_weights = None
    eq_weights = np.array([1.0 / len(tickers)] * len(tickers))

    for i in range(lookback, n_days - 1):
        # Rebalance check
        if current_weights is None or (i - lookback) % rebalance_freq == 0:
            window = log_returns.iloc[i - lookback:i]
            mean_ret = annualized_mean_returns(window)
            cov = annualized_covariance(window)
            try:
                current_weights = max_sharpe_portfolio(mean_ret, cov, risk_free_rate)
            except Exception:
                current_weights = eq_weights  # fallback if optimizer fails

        # Apply today's actual return to grow portfolio value
        day_return = log_returns.iloc[i].values
        port_ret = np.dot(current_weights, day_return)
        eq_ret = np.dot(eq_weights, day_return)

        portfolio_value.append(portfolio_value[-1] * np.exp(port_ret))
        equal_weight_value.append(equal_weight_value[-1] * np.exp(eq_ret))
        dates_used.append(dates[i + 1])

    results = pd.DataFrame({
        "optimized": portfolio_value,
        "equal_weight": equal_weight_value
    }, index=dates_used)

    return results

def compute_metrics(results: pd.DataFrame, risk_free_rate=0.05):
    metrics = {}
    for col in results.columns:
        daily_ret = np.log(results[col] / results[col].shift(1)).dropna()
        annual_ret = daily_ret.mean() * 252
        annual_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (annual_ret - risk_free_rate) / annual_vol
        max_drawdown = ((results[col] / results[col].cummax()) - 1).min()

        metrics[col] = {
            "Total Return": f"{(results[col].iloc[-1] - 1):.2%}",
            "Annualized Return": f"{annual_ret:.2%}",
            "Annualized Volatility": f"{annual_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}"
        }
    return pd.DataFrame(metrics)


def compute_turnover(prices, log_returns, lookback=252, rebalance_freq=21, risk_free_rate=0.05):
    """
    Measures how much weights change at each rebalance.
    High turnover = high implied transaction costs in real trading.
    """
    dates = log_returns.index
    n_days = len(dates)
    tickers = log_returns.columns.tolist()

    prev_weights = None
    turnovers = []

    for i in range(lookback, n_days - 1, rebalance_freq):
        window = log_returns.iloc[i - lookback:i]
        mean_ret = annualized_mean_returns(window)
        cov = annualized_covariance(window)
        try:
            weights = max_sharpe_portfolio(mean_ret, cov, risk_free_rate)
        except Exception:
            weights = np.array([1.0 / len(tickers)] * len(tickers))

        if prev_weights is not None:
            turnover = np.sum(np.abs(weights - prev_weights))
            turnovers.append(turnover)
        prev_weights = weights

    return np.mean(turnovers) if turnovers else 0.0