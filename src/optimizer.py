 # scipy SLSQP optimization
import numpy as np
from scipy.optimize import minimize
from src.stats import portfolio_performance

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.05):
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})   # Σw = 1
    bounds = tuple((0, 1) for _ in range(n_assets))                   # no shorting
    initial_guess = n_assets * [1. / n_assets]                        # start equal-weighted

    result = minimize(negative_sharpe, initial_guess, args=args,
                       method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x   # optimal weights

def min_variance_portfolio(mean_returns, cov_matrix):
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = n_assets * [1. / n_assets]

    result = minimize(portfolio_volatility, initial_guess, args=args,
                       method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def min_variance_closed_form(cov_matrix):
    """
    Analytical solution: w* = Σ⁻¹1 / (1^T Σ⁻¹ 1)
    No shorting constraint here — this allows negative weights,
    unlike min_variance_portfolio() which uses scipy with bounds=(0,1).
    So this won't exactly match scipy's bounded solution unless the
    unconstrained optimum happens to have all non-negative weights.
    """
    n = cov_matrix.shape[0]
    ones = np.ones(n)
    inv_cov = np.linalg.inv(cov_matrix)

    numerator = inv_cov @ ones
    denominator = ones @ inv_cov @ ones

    weights = numerator / denominator
    return weights


