# orchestrates the whole pipeline
from src.data_loader import fetch_prices
from src.stats import compute_log_returns, annualized_mean_returns, annualized_covariance, portfolio_performance
from src.optimizer import max_sharpe_portfolio, min_variance_portfolio
from src.monte_carlo import simulate_random_portfolios
from src.visualize import plot_everything

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

prices = fetch_prices(tickers)
log_returns = compute_log_returns(prices)
mean_returns = annualized_mean_returns(log_returns)
cov_matrix = annualized_covariance(log_returns)

optimal_weights = max_sharpe_portfolio(mean_returns, cov_matrix)
ret, std, sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

print("Max Sharpe Portfolio:")
for ticker, w in zip(tickers, optimal_weights):
    print(f"  {ticker}: {w:.2%}")
print(f"Expected Return: {ret:.2%}, Volatility: {std:.2%}, Sharpe: {sharpe:.2f}")

mc_results, _ = simulate_random_portfolios(5000, mean_returns, cov_matrix)
plot_everything(mc_results, mean_returns, cov_matrix)

from src.backtest import rolling_backtest, compute_metrics

results = rolling_backtest(prices, log_returns)
metrics = compute_metrics(results)
print("\nBacktest Results:")
print(metrics)

results.plot(figsize=(10, 6), title="Rolling Backtest: Optimized vs Equal-Weight")
import matplotlib.pyplot as plt
plt.ylabel("Portfolio Value (normalized to $1)")
plt.savefig("backtest_results.png", dpi=150)

from src.backtest import compute_turnover
avg_turnover = compute_turnover(prices, log_returns)
print(f"\nAverage turnover per rebalance: {avg_turnover:.2%}")

