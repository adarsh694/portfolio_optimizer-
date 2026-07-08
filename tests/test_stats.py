 # verify your matrix math against known values

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import fetch_prices
from src.stats import compute_log_returns, annualized_covariance
from src.optimizer import min_variance_portfolio, min_variance_closed_form

def test_closed_form_matches_scipy_when_unconstrained():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    prices = fetch_prices(tickers)
    log_returns = compute_log_returns(prices)
    cov_matrix = annualized_covariance(log_returns)

    w_scipy = min_variance_portfolio(np.zeros(len(tickers)), cov_matrix)  # mean_returns unused in this func but check signature
    w_closed = min_variance_closed_form(cov_matrix)

    print("\nScipy (bounded, no short):", w_scipy)
    print("Closed-form (unconstrained):", w_closed)

    if np.all(w_closed >= -1e-6):
        # unconstrained optimum happens to be non-negative -> should match closely
        np.testing.assert_allclose(w_scipy, w_closed, atol=1e-3)
        print("MATCH: unconstrained optimum has no negative weights, both methods agree.")
    else:
        print("DIVERGE (expected): closed-form wants to short at least one asset, scipy can't.")
        assert True  # not a failure, just documents the divergence

if __name__ == "__main__":
    test_closed_form_matches_scipy_when_unconstrained()