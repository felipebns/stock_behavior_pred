"""Independent check of Portfolio against an event-by-event simulation in the style of the Algotrading course (Aulas 4
e 5): cash and share counts, a fee on the traded value of each buy and sell, idle cash earning the risk-free rate.
Intraday: buy at the open after the decision, sell at that day's close. It shares no code with engine.portfolio."""
import numpy as np
import pandas as pd

from engine.portfolio import Portfolio
from engine.prices import forward_intraday_return

CAPITAL = 1_000_000.0
THRESHOLD = 0.55
FEE = 0.001


def market_data(n_days: int = 40, seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days, name="date")
    tickers = ["A", "B", "C", "D", "E"]
    opens = pd.DataFrame(50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (n_days, len(tickers))), axis=0)),
                         index=dates, columns=tickers)
    closes = opens * np.exp(rng.normal(0.0, 0.015, opens.shape))
    opens.loc[dates[17]:, "E"] = np.nan
    closes.loc[dates[16]:, "E"] = np.nan
    rf_daily = pd.Series(0.0002 + 0.0001 * np.sin(np.arange(n_days)), index=dates)
    return opens, closes, rf_daily


def model_output(opens, closes, seed: int = 5) -> pd.DataFrame:
    """Random probabilities every session (with the realized next-day return); every 4th day nobody passes, and a stock
    whose next day has no price is always liked, so the missing-return case is exercised."""
    rng = np.random.default_rng(seed)
    fwd_ret = forward_intraday_return(opens, closes)
    rows = []
    for k, date in enumerate(opens.index[2:-3]):
        high = 0.5 if k % 4 == 3 else 0.8
        rows += [{"date": date, "ticker": t, "fwd_ret": fwd_ret.at[date, t],
                  "prob": 0.79 if np.isnan(fwd_ret.at[date, t]) else rng.uniform(0.3, high)}
                 for t in opens.columns if not np.isnan(closes.at[date, t])]
    return pd.DataFrame(rows)


def market_frame(opens, closes, rf_daily) -> pd.DataFrame:
    dates = opens.index
    return pd.DataFrame({
        "holding_date": pd.Series(dates, index=dates).shift(-1),
        "bench_fwd_ret": forward_intraday_return(opens.mean(axis=1), closes.mean(axis=1)),
        "rf_daily": rf_daily,
    }, index=dates)


def simulate(opens, closes, rf_daily, predictions) -> pd.Series:
    """Portfolio value after each trading day's close, keyed by that day."""
    targets = {}
    for date, day in predictions.groupby("date"):
        chosen = day[day["prob"] > THRESHOLD]
        targets[date] = dict(zip(chosen["ticker"], chosen["prob"] / chosen["prob"].sum()))
    dates, cash, equity = opens.index, CAPITAL, {}
    for i in range(dates.get_loc(min(targets)), dates.get_loc(max(targets)) + 1):
        decided, day = dates[i], dates[i + 1]
        target = targets.get(decided, {})
        if target:
            invested = cash * (1 - FEE)
            proceeds = 0.0
            for ticker, weight in target.items():
                buy, sell = opens.at[day, ticker], closes.at[day, ticker]
                money = weight * invested
                proceeds += money if np.isnan(buy) or np.isnan(sell) else money / buy * sell
            cash = proceeds * (1 - FEE)
        else:
            cash *= 1 + rf_daily.at[decided]
        equity[day] = cash
    return pd.Series(equity)


def test_portfolio_matches_an_event_by_event_simulation():
    opens, closes, rf_daily = market_data()
    predictions = model_output(opens, closes)
    daily = Portfolio(THRESHOLD, FEE * 10_000).backtest(predictions, market_frame(opens, closes, rf_daily)).daily
    assert daily["invested"].any() and not daily["invested"].all()
    assert daily["n_missing"].sum() > 0
    expected = CAPITAL * (1.0 + daily["net"]).cumprod()
    simulated = simulate(opens, closes, rf_daily, predictions).reindex(daily.index)
    np.testing.assert_allclose(simulated.to_numpy(), expected.to_numpy(), rtol=1e-10)
