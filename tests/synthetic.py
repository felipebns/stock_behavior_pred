"""Synthetic market used by the tests: prices and constituents."""
import numpy as np
import pandas as pd

from config.config import Config
from engine.project import RunKey

BENCH = "^SP500TR"
RF = "^IRX"


def _bars(rng, dates, ticker, log_returns, dividend_every=None):
    n = len(dates)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    dividends = np.zeros(n)
    if dividend_every:
        dividends[dividend_every::dividend_every] = 0.5
    return pd.DataFrame({
        "date": dates, "ticker": ticker, "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(100_000, 1_000_000, n).astype(float), "dividends": dividends, "splits": 0.0,
    })


def make_market(n_days: int = 330, n_tickers: int = 12, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    market = rng.normal(0.0004, 0.01, n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    frames = [
        _bars(rng, dates, t, (0.5 + i / n_tickers) * market + rng.normal(0.0, 0.015, n_days), 63 if i % 3 == 0 else None)
        for i, t in enumerate(tickers)
    ]
    frames.append(_bars(rng, dates, BENCH, market))
    frames.append(pd.DataFrame({
        "date": dates, "ticker": RF, "open": np.nan, "high": np.nan, "low": np.nan,
        "close": 2.0 + 0.5 * np.sin(np.arange(n_days) / 40.0), "volume": 0.0, "dividends": 0.0, "splits": 0.0,
    }))
    snapshots = pd.DataFrame({
        "date": [dates[0], dates[150], dates[260]],
        "tickers": [tickers[:10], tickers[1:11], tickers[2:12]],
    })
    return {"prices": pd.concat(frames, ignore_index=True), "snapshots": snapshots}


def perturb_after(raw: dict, cutoff: pd.Timestamp, seed: int = 99) -> dict:
    """Same market up to `cutoff`; everything dated after it is scrambled and a new ticker joins the index."""
    rng = np.random.default_rng(seed)
    prices = raw["prices"].copy()
    future = prices["date"] > cutoff
    for column in ("open", "high", "low", "close", "volume"):
        prices.loc[future, column] = prices.loc[future, column] * rng.uniform(0.5, 1.5, int(future.sum()))
    prices.loc[future, "dividends"] = rng.uniform(0.0, 2.0, int(future.sum()))
    newcomer = raw["prices"].loc[raw["prices"]["ticker"] == "T00"].assign(ticker="NEW")
    snapshots = pd.concat([
        raw["snapshots"],
        pd.DataFrame({"date": [cutoff + pd.Timedelta(days=1)], "tickers": [["T02", "T03", "NEW"]]}),
    ], ignore_index=True)
    return {"prices": pd.concat([prices, newcomer], ignore_index=True), "snapshots": snapshots}


SMALL_LGBM = {
    "n_estimators": 10, "learning_rate": 0.1, "num_leaves": 4, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "importance_type": "gain", "random_state": 0, "deterministic": True,
    "force_col_wise": True, "n_jobs": 1, "verbose": -1,
}

RUNS = (RunKey(40), RunKey(40, 3))


def synthetic_config(root, **overrides) -> Config:
    """Decisions start at business day 210 of the synthetic market; features are complete from day ~160."""
    start = pd.bdate_range("2019-01-01", periods=330)[210]
    settings = {
        "data_in": root / "in", "data_out": root / "out", "price_start": "2019-01-01",
        "backtest_start": str(start.date()), "train_window_days": 40, "retrain_every": 1,
        "peer_count": 3,
        "peer_lookback_days": 21, "n_jobs": 1, "lgbm_params": SMALL_LGBM,
    }
    return Config(**{**settings, **overrides})


def yahoo_response(tickers, dates, skip=()) -> pd.DataFrame:
    """Same shape as yf.download(group_by='ticker') for the tickers not in `skip`."""
    bar = {"Open": 10.0, "High": 10.0, "Low": 10.0, "Close": 10.0, "Adj Close": 9.0,
           "Volume": 100.0, "Dividends": 0.0, "Stock Splits": 0.0}
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    frames = {t: pd.DataFrame(bar, index=index) for t in tickers if t not in skip}
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, axis=1)
    raw.columns = raw.columns.set_names(["Ticker", "Price"])
    return raw
