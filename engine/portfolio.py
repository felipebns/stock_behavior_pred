"""Long-only daily portfolio built from the probabilities predicted after each close."""
from dataclasses import dataclass

import pandas as pd

POSITION_COLUMNS = ["decision_date", "holding_date", "ticker", "prob", "weight", "fwd_ret", "contribution"]
DAILY_COLUMNS = ["decision_date", "gross", "net", "bench", "rf", "n_positions", "invested", "turnover", "cost", "n_missing"]


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame


class Portfolio:
    """Buys at the next open the stocks with prob > threshold, weighted by prob; otherwise cash at the risk-free rate."""

    def __init__(self, threshold: float, cost_bps: float = 0.0):
        self.threshold = threshold
        self.cost_bps = cost_bps

    def select(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Only `prob` drives the choice."""
        chosen = predictions.loc[predictions["prob"] > self.threshold, ["date", "ticker", "prob", "fwd_ret"]].copy()
        chosen["weight"] = chosen["prob"] / chosen.groupby("date")["prob"].transform("sum")
        return chosen

    def backtest(self, predictions: pd.DataFrame, market: pd.DataFrame) -> BacktestResult:
        """A held stock without a realized return (delisting, halt) counts as 0: dropping it would use future information."""
        window = market.loc[predictions["date"].min():predictions["date"].max()]
        last_realized = window["bench_fwd_ret"].last_valid_index()
        window = window.iloc[:0] if last_realized is None else window.loc[:last_realized]
        gaps = window.index[window["bench_fwd_ret"].isna()]
        if len(gaps):
            raise ValueError(f"benchmark sem retorno realizado dentro do período ({', '.join(str(d.date()) for d in gaps)}); "
                             "baixe os preços de novo")
        if window.empty:
            return BacktestResult(daily=pd.DataFrame(columns=DAILY_COLUMNS), positions=pd.DataFrame(columns=POSITION_COLUMNS))
        dates = window.index
        chosen = self.select(predictions[predictions["date"].isin(dates)])
        chosen["missing"] = chosen["fwd_ret"].isna()
        chosen["contribution"] = chosen["weight"] * chosen["fwd_ret"].fillna(0.0)
        by_date = chosen.groupby("date")
        n_positions = by_date.size().reindex(dates, fill_value=0)
        n_missing = by_date["missing"].sum().reindex(dates, fill_value=0).astype(int)
        invested = n_positions > 0
        stock_return = by_date["contribution"].sum().reindex(dates, fill_value=0.0)
        gross = stock_return.where(invested, window["rf_daily"])
        turnover = self._turnover(chosen, dates, stock_return)
        cost = turnover * self.cost_bps / 10_000.0
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
            "n_missing": n_missing.to_numpy(),
        }, index=pd.DatetimeIndex(window["holding_date"], name="holding_date"))
        positions = chosen.rename(columns={"date": "decision_date"})
        positions["holding_date"] = positions["decision_date"].map(window["holding_date"])
        return BacktestResult(daily=daily, positions=positions[POSITION_COLUMNS].reset_index(drop=True))

    @staticmethod
    def _turnover(chosen: pd.DataFrame, dates: pd.DatetimeIndex, stock_return: pd.Series) -> pd.Series:
        """Buys + sells at each open (Σ|Δw|): target weights vs yesterday's weights after they drifted with prices."""
        weights = chosen.pivot(index="date", columns="ticker", values="weight").reindex(dates).fillna(0.0)
        growth = 1.0 + chosen.pivot(index="date", columns="ticker", values="fwd_ret").reindex(dates).fillna(0.0)
        drifted = (weights * growth).div(1.0 + stock_return, axis=0)
        return (weights - drifted.shift(1).fillna(0.0)).abs().sum(axis=1)
