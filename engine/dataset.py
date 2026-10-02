"""Long table indexed by (date, ticker): features known after the close of `date`, realized forward return and label."""
import numpy as np
import pandas as pd

from engine.features import DATE_FEATURES, FEATURES, TICKER_FEATURES


def build_dataset(ticker_features: dict[str, pd.DataFrame], date_features: pd.DataFrame, membership: pd.DataFrame,
                  close: pd.DataFrame, fwd_ret: pd.DataFrame, history_days: int) -> pd.DataFrame:
    """A row per stock that, on that date, is in the index, has a close on each of the last history_days sessions
    (that date included) and all date features: a gap or a recent listing keeps it out until the window is complete."""
    counts = np.cumsum(close.notna().to_numpy(), axis=0)
    previous = np.zeros_like(counts)
    previous[history_days:] = counts[:-history_days]
    eligible = (
        membership.to_numpy()
        & (counts - previous >= history_days)
        & date_features[DATE_FEATURES].notna().all(axis=1).to_numpy()[:, None]
    )
    rows, cols = np.nonzero(eligible)
    index = pd.MultiIndex.from_arrays([close.index[rows], close.columns[cols]], names=["date", "ticker"])
    data = {name: ticker_features[name].to_numpy(dtype=np.float32)[rows, cols] for name in TICKER_FEATURES}
    date_values = date_features[DATE_FEATURES].to_numpy(dtype=np.float32)[rows]
    data.update({name: date_values[:, j] for j, name in enumerate(DATE_FEATURES)})
    dataset = pd.DataFrame(data, index=index)[FEATURES]
    realized = fwd_ret.to_numpy(dtype=float)[rows, cols]
    dataset["fwd_ret"] = realized
    dataset["label"] = np.where(np.isnan(realized), np.nan, (realized > 0).astype(float))
    return dataset
