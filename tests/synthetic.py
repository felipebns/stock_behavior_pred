"""Synthetic market used by the tests: prices, constituents, fed funds futures and GPR."""
import numpy as np
import pandas as pd

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
    ff_bars = pd.DataFrame([
        {"date": day, "symbol": f"ZQ{m}", "expiration": day + pd.offsets.MonthEnd(m),
         "close": 100.0 - 100.0 * (0.02 + 0.001 * m + 0.003 * np.sin(k / 50.0))}
        for k, day in enumerate(dates) for m in range(15)
    ])
    days = pd.date_range(dates[0] - pd.Timedelta(days=60), dates[-1] + pd.Timedelta(days=30), freq="D", name="date")
    gpr = pd.DataFrame({
        "gpr": rng.uniform(50.0, 200.0, len(days)),
        "act": rng.uniform(20.0, 150.0, len(days)),
        "threat": rng.uniform(20.0, 150.0, len(days)),
    }, index=days)
    return {"prices": pd.concat(frames, ignore_index=True), "snapshots": snapshots, "ff_bars": ff_bars, "gpr": gpr}


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
    ff_bars = raw["ff_bars"].copy()
    later = ff_bars["date"] > cutoff
    ff_bars.loc[later, "close"] = rng.uniform(94.0, 99.0, int(later.sum()))
    gpr = raw["gpr"].copy()
    gpr.loc[gpr.index > cutoff] = rng.uniform(1.0, 500.0, size=(int((gpr.index > cutoff).sum()), 3))
    return {"prices": pd.concat([prices, newcomer], ignore_index=True), "snapshots": snapshots, "ff_bars": ff_bars, "gpr": gpr}
