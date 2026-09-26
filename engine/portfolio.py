"""Long-only daily portfolio built from the probabilities predicted after each close."""
from dataclasses import dataclass

import pandas as pd

POSITION_COLUMNS = ["decision_date", "holding_date", "ticker", "prob", "weight", "fwd_ret", "contribution"]


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame


def select_positions(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Stocks above the threshold, weighted by predicted probability; only `prob` drives the choice."""
    chosen = predictions.loc[predictions["prob"] > threshold, ["date", "ticker", "prob", "fwd_ret"]].copy()
    chosen["weight"] = chosen["prob"] / chosen.groupby("date")["prob"].transform("sum")
    return chosen


def run_backtest(predictions: pd.DataFrame, market: pd.DataFrame, threshold: float, cost_bps: float = 0.0) -> BacktestResult:
    """A held stock without a realized return (delisting, halt) counts as 0: dropping it would use future information."""
    window = market.loc[predictions["date"].min():predictions["date"].max()]
    realized = window["bench_fwd_ret"].notna().to_numpy()
    window = window.iloc[: realized.nonzero()[0].max() + 1] if realized.any() else window.iloc[:0]
    gaps = window.index[window["bench_fwd_ret"].isna()]
    if len(gaps):
        raise ValueError(f"benchmark sem retorno realizado dentro do período ({', '.join(str(d.date()) for d in gaps)}); "
                         "baixe os preços de novo")
    dates = window.index
    chosen = select_positions(predictions[predictions["date"].isin(dates)], threshold)
    chosen["missing"] = chosen["fwd_ret"].isna()
    chosen["contribution"] = chosen["weight"] * chosen["fwd_ret"].fillna(0.0)
    by_date = chosen.groupby("date")
    n_positions = by_date.size().reindex(dates, fill_value=0)
    invested = n_positions > 0
    stock_return = by_date["contribution"].sum().reindex(dates, fill_value=0.0)
    gross = stock_return.where(invested, window["rf_daily"])
    turnover = _turnover(chosen, dates, stock_return)
    cost = turnover * cost_bps / 10_000.0
    daily = pd.DataFrame({
        "decision_date": dates.to_numpy(),
        "gross": gross.to_numpy(),
        "net": ((1.0 + gross) * (1.0 - cost) - 1.0).to_numpy(),
        "bench": window["bench_fwd_ret"].to_numpy(),
        "rf": window["rf_daily"].to_numpy(),
        "n_positions": n_positions.to_numpy(),
        "invested": invested.to_numpy(),
        "turnover": turnover.to_numpy(),
        "cost": cost.to_numpy(),
        "n_missing": by_date["missing"].sum().reindex(dates, fill_value=0).astype(int).to_numpy(),
    }, index=pd.DatetimeIndex(window["holding_date"], name="holding_date"))
    positions = chosen.rename(columns={"date": "decision_date"})
    positions["holding_date"] = positions["decision_date"].map(window["holding_date"])
    return BacktestResult(daily=daily, positions=positions[POSITION_COLUMNS].reset_index(drop=True))


def _turnover(chosen: pd.DataFrame, dates: pd.DatetimeIndex, stock_return: pd.Series) -> pd.Series:
    """Buys + sells at each open (Σ|Δw|): target weights vs yesterday's weights after they drifted with prices."""
    weights = chosen.pivot(index="date", columns="ticker", values="weight").reindex(dates).fillna(0.0)
    growth = 1.0 + chosen.pivot(index="date", columns="ticker", values="fwd_ret").reindex(dates).fillna(0.0)
    drifted = (weights * growth).div(1.0 + stock_return, axis=0)
    return (weights - drifted.shift(1).fillna(0.0)).abs().sum(axis=1)
