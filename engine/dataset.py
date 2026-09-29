"""Long table indexed by (date, ticker): features known after the close of `date`; the target depends on the horizon."""
import numpy as np
import pandas as pd

from engine.features import DATE_FEATURES, FEATURES, TICKER_FEATURES


def build_dataset(ticker_features: dict[str, pd.DataFrame], date_features: pd.DataFrame, membership: pd.DataFrame,
                  close: pd.DataFrame, min_history_days: int) -> pd.DataFrame:
    """A row per stock that, on that date, is in the index, traded, has enough history and all date features."""
    has_close = close.notna().to_numpy()
    eligible = (
        membership.to_numpy()
        & has_close
        & (np.cumsum(has_close, axis=0) >= min_history_days)
        & date_features[DATE_FEATURES].notna().all(axis=1).to_numpy()[:, None]
    )
    rows, cols = np.nonzero(eligible)
    index = pd.MultiIndex.from_arrays([close.index[rows], close.columns[cols]], names=["date", "ticker"])
    data = {name: ticker_features[name].to_numpy(dtype=np.float32)[rows, cols] for name in TICKER_FEATURES}
    date_values = date_features[DATE_FEATURES].to_numpy(dtype=np.float32)[rows]
    data.update({name: date_values[:, j] for j, name in enumerate(DATE_FEATURES)})
    return pd.DataFrame(data, index=index)[FEATURES]


def horizon_target(dataset: pd.DataFrame, returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """fwd_ret: open of t+1 to the open of t+1+horizon (dividends reinvested), NaN if a session has no return;
    label: 1 when fwd_ret beats the median of that date's rows. Both are known only after the open of t+1+horizon."""
    compounded = np.expm1(np.log1p(returns).rolling(horizon, min_periods=horizon).sum().shift(1 - horizon))
    rows = compounded.index.get_indexer(dataset.index.get_level_values("date"))
    cols = compounded.columns.get_indexer(dataset.index.get_level_values("ticker"))
    fwd_ret = pd.Series(compounded.to_numpy()[rows, cols], index=dataset.index)
    median = fwd_ret.groupby(level="date").transform("median")
    return pd.DataFrame({"fwd_ret": fwd_ret, "label": (fwd_ret > median).astype(float).where(fwd_ret.notna())})
