# Stock Behavior Prediction — S&P 500 daily direction backtest

Every trading day, after the close, a LightGBM classifier retrained on the last *X* trading days estimates, for each stock that is in the S&P 500 **on that day**, the probability that it goes up over the next portfolio day. The long-only portfolio holds the stocks whose probability is above a threshold, weighted by that probability; if none qualifies it stays in cash at the 13-week T-bill rate. Results are compared with the S&P 500 Total Return index in a Streamlit app.

## Quick start

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python main.py download        # prices without dividend adjustment (yfinance) → data/in/prices.parquet
venv/bin/python main.py run             # features, walk-forward and backtest → data/out/
venv/bin/streamlit run app/dashboard.py # results app (never retrains)
venv/bin/python -m pytest               # includes the lookahead tests
```

`main.py run --start YYYY-MM-DD --end YYYY-MM-DD` limits the decision dates (handy for quick runs).

## Time convention (no lookahead)

| Step | When | Uses |
|---|---|---|
| Decision | after the close of *t* | prices/volumes/dividends dated ≤ *t*, index membership ≤ *t*, fed funds futures ≤ *t*, GPR ≤ *t* − 7 days |
| Execution | open of *t+1*, rebalanced at the open of *t+2* | — |
| Realized return / target | `(Open[t+2] + Div[t+2]) / Open[t+1] − 1`, label = return > 0 | — |
| Training for decision *t* | rows dated *t* − X − 1 … *t* − 2 | only labels already known at the close of *t* |

`tests/test_pipeline.py::test_predictions_do_not_depend_on_future_data` runs the whole pipeline twice — once with every input dated after a cutoff scrambled — and requires identical predictions up to the cutoff.

## Data (`data/in/`)

- `sp500_historical_constituents.csv` — point-in-time membership snapshots; renamed tickers are mapped in `config.TICKER_ALIASES`.
- `prices.parquet` — Yahoo daily bars with `auto_adjust=False, actions=True` (split-adjusted, not dividend-adjusted); `Adj Close` is never stored because it embeds future dividends. Includes `^SP500TR` (benchmark) and `^IRX` (risk-free).
- `ff_futures_daily.csv` — 30-day fed funds futures: implied rate ~12 months ahead, its 21-day change and the slope vs the current month.
- `data_gpr_daily_recent.xls` — Caldara & Iacoviello daily geopolitical risk index, used with a 7-day publication lag.

## Model and portfolio

- 36 causal features (`engine/features.py`): stock returns/momentum/volatility/trend/volume/dividend yield/beta, cross-sectional ranks among index members, market and breadth, fed funds and GPR.
- LightGBM with fixed conservative parameters (`config/config.py`), retrained every day (sequential loop).
- Weights `prob_i / Σ prob` over stocks with `prob > threshold` (default 0.55, fixed a priori); transaction costs in bps per side, charged on buys + sells (Σ|Δw| at each open, measured against weights drifted by the previous day's returns; default 5 bps).
- Metrics: cumulative return, CAGR, volatility, Sharpe/Sortino (excess over ^IRX), max drawdown, Calmar, hit ratios, beta/alpha/information ratio vs the benchmark; model AUC, IC and calibration.

## Known limitations

- Delisted companies are not on Yahoo, some old tickers now name a different company (STI, INFO, APC, SBNY...) and a few recent delistings had their whole history purged (EA, AVB, EQR in Aug 2026), so those members drop out of the universe (survivorship bias). Renames are mapped in `config.TICKER_ALIASES` (FB→META, ANTM→ELV...). `run_info.json` lists every member priced on under half of its member days, plus coverage by year; the app plots daily coverage.
- Yahoo records large spin-offs as fractional splits (RTX 2020, GE 2023, DD 2019...), so those series stay continuous; daily returns above 50% among members are still counted in `run_info.json` as possible data errors, never corrected.
- The backtest starts when the fed funds futures file allows (Dec 2020 decisions with X = 252) and ends a few days after it ends (staleness tolerance). The constituents file ends 2026-06-30, so later decisions use the last snapshot.
- `main.py run --start/--end` writes to `data/out/partial` so it never replaces the full results; open it with `BACKTEST_RESULTS_DIR=data/out/partial venv/bin/streamlit run app/dashboard.py`.

## Layout

```
config/config.py     all parameters, ticker aliases, LightGBM params
main.py              CLI: download | run
app/dashboard.py     Streamlit app
engine/data/         universe (membership), prices (yfinance, returns), macro (futures, GPR, risk-free)
engine/features.py   causal features      engine/dataset.py   (date, ticker) table + target
engine/walk_forward.py  daily retraining  engine/pipeline.py  orchestration
engine/portfolio.py  selection, weights, costs   engine/metrics.py   metrics
engine/results.py    data/out persistence
tests/               unit tests, synthetic market, end-to-end lookahead test, app smoke test
```
