# random portfolio simulation
import numpy as np
from src.stats import portfolio_performance

def simulate_random_portfolios(n_portfolios, mean_returns, cov_matrix, risk_free_rate=0.05):
    n_assets = len(mean_returns)
    results = np.zeros((3, n_portfolios))   # rows: return, std, sharpe
    weights_record = []

    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)          # normalize so they sum to 1
        weights_record.append(weights)

        ret, std, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = ret
        results[1, i] = std
        results[2, i] = sharpe

    return results, weights_record