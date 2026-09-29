# Stock Behavior Prediction

Every *h* trading days, after the close, a LightGBM classifier estimates for each stock in the S&P 500 **on that day** the probability that it beats the median of the index members over the next *h* days. A long-only portfolio buys, at the next open, the stocks whose probability is above a threshold, weighted by that probability, and holds them until the next decision; if none qualifies, it holds cash at the 13-week T-bill rate.

## Objective and scope

- **Objective:** find out whether stock selection from price history alone beats the S&P 500 after costs, iterating quickly over horizons and training windows.
- **Run parameters:** horizon *h* (1 = daily, 5 = weekly, 21 = monthly decisions and label), training window *X* (sessions of examples) and retraining every *k* decisions. All runs cover the same period (from 2020-01-02).
- **Label:** relative. It is 1 if the stock's return over the next *h* sessions beats the median of the index members that day, so the model ranks stocks instead of guessing the market.
- **Data:** Yahoo Finance prices (split-adjusted, dividends added separately) and point-in-time S&P 500 constituents ([fja05680/sp500](https://github.com/fja05680/sp500)). Price history only, no macro data.
- **No lookahead:** a decision after the close of *t* uses only data dated on or before *t*; the model trains only on labels already known then (rows up to *t* − *h* − 1). The tests enforce it. An independent event-by-event simulation (cash, shares, fees, carry) checks the portfolio and its costs.
- **Out of scope:** live trading, hyperparameter tuning, and delisted companies (Yahoo has no data for them; the app lists them).

## Benchmark

S&P 500 Total Return (`^SP500TR`), open to open, on the same days as the portfolio.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python main.py download                                   # latest constituents and prices
venv/bin/python main.py features                                   # features, once per download
venv/bin/python main.py run --horizon 5 --window 21 --retrain 1    # one walk-forward run
venv/bin/streamlit run app/dashboard.py                            # results; pick h, X and k in the sidebar and click "Rodar"
venv/bin/python -m pytest
```

Defaults (h, X, k, threshold, transaction cost, backtest start) live in `config/config.py`.
