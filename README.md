# Stock Behavior Prediction

Every trading day, after the close, a LightGBM classifier estimates the probability that each stock in the S&P 500 **on that day** goes up the next day. A long-only portfolio buys, at the next open, the stocks whose probability is above a threshold, weighted by that probability; if none qualifies, it holds cash at the 13-week T-bill rate.

## Objective and scope

- **Objective:** test whether daily direction predictions from price history alone beat the S&P 500, quickly and for several training windows.
- **Training window:** the model is retrained every day on the last *X* trading days (5 = one week, 21 = one month, 63 = three months…) and always predicts only the next day. All windows are evaluated over the same period (from 2020-01-02).
- **Data:** Yahoo Finance prices (split-adjusted, dividends added separately) and point-in-time S&P 500 constituents ([fja05680/sp500](https://github.com/fja05680/sp500)). Price history only, no macro data.
- **No lookahead:** a decision after the close of *t* uses only data dated on or before *t*; the model trains only on labels already known then. The tests enforce it.
- **Out of scope:** live trading, hyperparameter tuning, and delisted companies (Yahoo has no data for them; the app lists them).

## Benchmark

S&P 500 Total Return (`^SP500TR`), open to open, the same holding period as the portfolio.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python main.py download          # latest constituents and prices
venv/bin/python main.py features          # features, once per download
venv/bin/python main.py run --window 21   # walk-forward with a 21-day training window
venv/bin/streamlit run app/dashboard.py   # results; pick another window in the sidebar and click "Rodar"
venv/bin/python -m pytest
```

Defaults (window, threshold, transaction cost, backtest start) live in `config/config.py`.
