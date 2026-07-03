 # returns, mean vector, covariance matrix
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    log(P_t / P_{t-1}) instead of simple returns because log returns
    are additive across time (log(a/b) + log(b/c) = log(a/c)), which
    makes annualizing and compounding math clean.
    """
    return np.log(prices / prices.shift(1)).dropna()

def annualized_mean_returns(log_returns: pd.DataFrame) -> np.ndarray:
    return log_returns.mean().values * TRADING_DAYS

def annualized_covariance(log_returns: pd.DataFrame) -> np.ndarray:
    return log_returns.cov().values * TRADING_DAYS

def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate=0.05):
    """
    This function is the entire mathematical core of the project.
    Everything else — Monte Carlo, scipy optimization — just calls this
    with different weight vectors and reacts to the output.
    """
    port_return = np.dot(weights, mean_returns)               # w^T μ
    port_variance = weights.T @ cov_matrix @ weights           # w^T Σ w  <- the quadratic form
    port_std = np.sqrt(port_variance)
    sharpe = (port_return - risk_free_rate) / port_std
    return port_return, port_std, sharpe