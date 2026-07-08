# portfolio_optimizer-
## Setup

```bash
pip install -r requirements.txt
```

First run will fetch price data from Yahoo Finance and cache it to `data/prices.csv`.
Subsequent runs use the cache automatically — delete `data/prices.csv` to force a fresh pull.

## Run

```bash
python main.py
```

Generates `efficient_frontier.png` showing the Monte Carlo simulation, efficient frontier,
and optimal (max Sharpe / min variance) portfolios.

## Tests

```bash
python tests/test_stats.py
```

Validates the analytical closed-form minimum-variance solution against scipy's
constrained optimizer.