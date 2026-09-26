"""S&P 500 membership as of each date (last snapshot dated on or before it)."""
import numpy as np
import pandas as pd


def to_yahoo(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def parse_constituents(raw: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    snapshots = raw.assign(date=pd.to_datetime(raw["date"])).sort_values("date").reset_index(drop=True)

    def normalize(field: str) -> list[str]:
        yahoo = (to_yahoo(t) for t in field.split(",") if t.strip())
        return list(dict.fromkeys(aliases.get(t, t) for t in yahoo))

    return pd.DataFrame({"date": snapshots["date"], "tickers": snapshots["tickers"].map(normalize)})


def load_constituents(path, aliases: dict[str, str]) -> pd.DataFrame:
    return parse_constituents(pd.read_csv(path), aliases)


def members_on(snapshots: pd.DataFrame, date) -> list[str]:
    pos = snapshots["date"].searchsorted(pd.Timestamp(date), side="right") - 1
    return [] if pos < 0 else list(snapshots["tickers"].iloc[pos])


def tickers_since(snapshots: pd.DataFrame, start) -> list[str]:
    first = max(snapshots["date"].searchsorted(pd.Timestamp(start), side="right") - 1, 0)
    return sorted(set().union(*snapshots["tickers"].iloc[first:]))


def membership_matrix(snapshots: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    column = {ticker: j for j, ticker in enumerate(tickers)}
    table = np.zeros((len(snapshots), len(tickers)), dtype=bool)
    for i, members in enumerate(snapshots["tickers"]):
        table[i, [column[t] for t in members if t in column]] = True
    pos = snapshots["date"].searchsorted(dates, side="right") - 1
    values = table[np.clip(pos, 0, None)]
    values[pos < 0] = False
    return pd.DataFrame(values, index=dates, columns=tickers)
