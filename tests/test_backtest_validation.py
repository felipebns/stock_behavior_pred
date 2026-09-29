"""Independent check of Portfolio against an event-by-event simulation in the style of the Algotrading course
(Aulas 4 e 5): cash and share counts, marking to market at each open, a fee on the traded value, idle cash earning
the risk-free rate (carry), dividends reinvested at the open they go ex. It shares no code with engine.portfolio."""
import numpy as np
import pandas as pd
import pytest

from engine.portfolio import Portfolio
from engine.prices import forward_open_return

CAPITAL = 1_000_000.0
THRESHOLD = 0.55
COST_BPS = 10.0


def market_data(n_days: int = 40, seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days, name="date")
    tickers = ["A", "B", "C", "D", "E"]
    opens = pd.DataFrame(50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (n_days, len(tickers))), axis=0)),
                         index=dates, columns=tickers)
    opens.loc[dates[17]:, "E"] = np.nan
    dividends = pd.DataFrame(0.0, index=dates, columns=tickers)
    dividends.loc[dates[[6, 12, 23]], ["A", "C"]] = 0.4
    rf_daily = pd.Series(0.0002 + 0.0001 * np.sin(np.arange(n_days)), index=dates)
    return opens, dividends, rf_daily


def model_output(opens: pd.DataFrame, horizon: int, seed: int = 5) -> pd.DataFrame:
    """Random probabilities at each decision (every `horizon` sessions); every 4th decision nobody passes."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, date in enumerate(opens.index[2:-3:horizon]):
        high = 0.5 if k % 4 == 3 else 0.8
        rows += [{"date": date, "ticker": t, "prob": rng.uniform(0.3, high)}
                 for t in opens.columns if not np.isnan(opens.at[date, t])]
    return pd.DataFrame(rows)


def market_frame(opens: pd.DataFrame, rf_daily: pd.Series) -> pd.DataFrame:
    dates = opens.index
    return pd.DataFrame({
        "holding_date": pd.Series(dates, index=dates).shift(-1),
        "bench_fwd_ret": forward_open_return(opens.mean(axis=1)),
        "rf_daily": rf_daily,
    }, index=dates)


def simulate(opens, dividends, rf_daily, predictions) -> pd.Series:
    """Portfolio value at each open, before trading, keyed by the holding day that ends at that open."""
    targets = {}
    for date, day in predictions.groupby("date"):
        chosen = day[day["prob"] > THRESHOLD]
        targets[date] = dict(zip(chosen["ticker"], chosen["prob"] / chosen["prob"].sum()))
    dates, price = opens.index, opens.ffill()
    cash, shares, equity = CAPITAL, {}, {}
    for i in range(dates.get_loc(min(targets)), len(dates) - 2):
        decided, trade_open, next_open = dates[i], dates[i + 1], dates[i + 2]
        if decided in targets:
            holdings = {t: q * price.at[trade_open, t] for t, q in shares.items()}
            value = cash + sum(holdings.values())
            target = targets[decided]
            traded = sum(abs(target.get(t, 0.0) * value - holdings.get(t, 0.0)) for t in set(target) | set(holdings))
            value -= traded * COST_BPS / 10_000
            shares = {t: w * value / price.at[trade_open, t] for t, w in target.items()}
            cash = 0.0 if target else value
        cash *= 1.0 + rf_daily.at[decided]
        for t in shares:
            if not np.isnan(opens.at[next_open, t]):
                shares[t] *= 1.0 + dividends.at[next_open, t] / opens.at[next_open, t]
        equity[trade_open] = cash + sum(q * price.at[next_open, t] for t, q in shares.items())
    return pd.Series(equity)


@pytest.mark.parametrize("horizon", [1, 5])
def test_portfolio_matches_an_event_by_event_simulation(horizon):
    opens, dividends, rf_daily = market_data()
    predictions = model_output(opens, horizon)
    daily = Portfolio(THRESHOLD, COST_BPS).backtest(
        predictions, market_frame(opens, rf_daily), forward_open_return(opens, dividends), horizon).daily
    assert daily["invested"].any() and not daily["invested"].all()
    assert daily["n_missing"].sum() > 0 and (daily["turnover"] > 0).sum() > 3
    expected = CAPITAL * (1.0 + daily["net"]).cumprod()
    simulated = simulate(opens, dividends, rf_daily, predictions).reindex(daily.index)
    np.testing.assert_allclose(simulated.to_numpy(), expected.to_numpy(), rtol=1e-10)
