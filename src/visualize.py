# plotting
import numpy as np
import matplotlib.pyplot as plt
from src.optimizer import min_variance_portfolio, max_sharpe_portfolio
from src.stats import portfolio_performance

def efficient_frontier(mean_returns, cov_matrix, n_points=50):
    from scipy.optimize import minimize

    min_ret = min(mean_returns)
    max_ret = max(mean_returns)
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_vol = []
    n_assets = len(mean_returns)

    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, target=target: np.dot(w, mean_returns) - target}
        )
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = n_assets * [1. / n_assets]

        result = minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
                           initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        frontier_vol.append(result.fun)

    return target_returns, frontier_vol

def plot_everything(mc_results, mean_returns, cov_matrix, risk_free_rate=0.05):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Monte Carlo cloud
    sc = ax.scatter(mc_results[1], mc_results[0], c=mc_results[2], cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(sc, label='Sharpe Ratio')

    # Efficient frontier
    target_returns, frontier_vol = efficient_frontier(mean_returns, cov_matrix)
    ax.plot(frontier_vol, target_returns, 'r--', linewidth=2, label='Efficient Frontier')

    # Max Sharpe point
    max_sharpe_w = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate)
    ret, std, sharpe = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix, risk_free_rate)
    ax.scatter(std, ret, c='red', marker='*', s=300, label='Max Sharpe')

    # Min variance point
    min_var_w = min_variance_portfolio(mean_returns, cov_matrix)
    ret2, std2, _ = portfolio_performance(min_var_w, mean_returns, cov_matrix, risk_free_rate)
    ax.scatter(std2, ret2, c='blue', marker='*', s=300, label='Min Variance')

    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Portfolio Optimization: Efficient Frontier')
    ax.legend()
    plt.savefig('efficient_frontier.png', dpi=150)
    plt.show()