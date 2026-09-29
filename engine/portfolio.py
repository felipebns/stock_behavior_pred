"""Long-only portfolio: at each decision, buy at the next open the stocks with prob > threshold (weights ∝ prob) and
hold them until the next decision, their weights drifting with prices."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

POSITION_COLUMNS = ["decision_date", "holding_date", "ticker", "prob", "weight", "realized", "contribution"]


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame


class Portfolio:
    """Buys at the open after each decision the stocks with prob > threshold, weighted by prob; otherwise cash at the risk-free rate."""

    def __init__(self, threshold: float, cost_bps: float = 0.0):
        self.threshold = threshold
        self.cost_bps = cost_bps

    def select(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Only `prob` drives the choice."""
        chosen = predictions.loc[predictions["prob"] > self.threshold, ["date", "ticker", "prob"]].copy()
        chosen["weight"] = chosen["prob"] / chosen.groupby("date")["prob"].transform("sum")
        return chosen

    def backtest(self, predictions: pd.DataFrame, market: pd.DataFrame, returns: pd.DataFrame,
                 horizon: int = 1) -> BacktestResult:
        """Day s earns `returns` row s (open of s+1 to open of s+2) with the decision in force at s. Trades, and their
        cost, happen only at the open after a decision; the fee is charged on the value traded to reach the target
        weights and comes out of every position. A held stock without a return that day (delisting, halt) counts as 0:
        dropping it would use future information."""
        decisions = pd.DatetimeIndex(np.sort(predictions["date"].unique()))
        window = market.iloc[market.index.get_loc(decisions[0]):market.index.get_loc(decisions[-1]) + horizon]
        last_realized = window["bench_fwd_ret"].last_valid_index()
        window = window.iloc[:0] if last_realized is None else window.loc[:last_realized]
        gaps = window.index[window["bench_fwd_ret"].isna()]
        if len(gaps):
            raise ValueError(f"benchmark sem retorno realizado dentro do período ({', '.join(str(d.date()) for d in gaps)}); "
                             "baixe os preços de novo")
        days = window.index
        in_force = decisions[decisions.searchsorted(days, side="right") - 1]
        period, periods = pd.factorize(in_force)
        chosen = self.select(predictions[predictions["date"].isin(periods)])
        weights = chosen.pivot(index="date", columns="ticker", values="weight").reindex(in_force).fillna(0.0)
        raw = returns.reindex(index=days, columns=weights.columns).to_numpy()
        held = weights.to_numpy()
        growth = pd.DataFrame(1.0 + np.nan_to_num(raw)).groupby(period).cumprod()
        value_start = held * growth.groupby(period).shift(1, fill_value=1.0).to_numpy()
        value_end = held * growth.to_numpy()
        start_total, end_total = value_start.sum(axis=1), value_end.sum(axis=1)
        invested = start_total > 0
        stock_return = np.divide(end_total, start_total, out=np.ones(len(days)), where=invested) - 1.0
        gross = np.where(invested, stock_return, window["rf_daily"].to_numpy())
        drifted = np.divide(value_end, end_total[:, None], out=np.zeros_like(value_end), where=invested[:, None])
        before = np.vstack([np.zeros((1, held.shape[1])), drifted[:-1]])
        rebalance = days.isin(decisions)
        turnover = np.where(rebalance, np.abs(held - before).sum(axis=1), 0.0)
        cost = turnover * self.cost_bps / 10_000.0
        daily = pd.DataFrame({
            "decision_date": in_force,
            "gross": gross,
            "net": (1.0 + gross) * (1.0 - cost) - 1.0,
            "bench": window["bench_fwd_ret"].to_numpy(),
            "rf": window["rf_daily"].to_numpy(),
            "n_positions": (held > 0).sum(axis=1),
            "invested": invested,
            "rebalance": rebalance,
            "turnover": turnover,
            "cost": cost,
            "n_missing": (np.isnan(raw) & (held > 0)).sum(axis=1),
        }, index=pd.DatetimeIndex(window["holding_date"], name="holding_date"))
        last_day = pd.Series(np.arange(len(days))).groupby(period).max().to_numpy()
        realized = growth.to_numpy()[last_day] - 1.0
        positions = chosen.rename(columns={"date": "decision_date"})
        positions["holding_date"] = positions["decision_date"].map(market["holding_date"])
        positions["realized"] = realized[periods.get_indexer(positions["decision_date"]),
                                         weights.columns.get_indexer(positions["ticker"])]
        positions["contribution"] = positions["weight"] * positions["realized"]
        return BacktestResult(daily=daily, positions=positions[POSITION_COLUMNS].reset_index(drop=True))
