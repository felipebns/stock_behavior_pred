"""S&P 500 membership as of each date: the last snapshot dated on or before it."""
import io
import urllib.request

import numpy as np
import pandas as pd


def to_yahoo(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


class Universe:
    """Snapshots `date` (sorted) → `tickers` (Yahoo format): the index composition from that date on."""

    def __init__(self, snapshots: pd.DataFrame):
        self.snapshots = snapshots

    @classmethod
    def from_frame(cls, raw: pd.DataFrame, aliases: dict[str, str]) -> "Universe":
        snapshots = raw.assign(date=pd.to_datetime(raw["date"])).sort_values("date").reset_index(drop=True)

        def normalize(field: str) -> list[str]:
            yahoo = (to_yahoo(t) for t in field.split(",") if t.strip())
            return list(dict.fromkeys(aliases.get(t, t) for t in yahoo))

        return cls(pd.DataFrame({"date": snapshots["date"], "tickers": snapshots["tickers"].map(normalize)}))

    @classmethod
    def from_csv(cls, source, aliases: dict[str, str]) -> "Universe":
        return cls.from_frame(pd.read_csv(source), aliases)

    @staticmethod
    def fetch(url: str, fetch=None) -> bytes:
        """The snapshots CSV, only if what came back really is that table (not an error page)."""
        content = (fetch or _fetch)(url)
        try:
            columns = list(pd.read_csv(io.BytesIO(content), nrows=1).columns)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            columns = []
        if columns != ["date", "tickers"]:
            raise ValueError(f"o arquivo baixado de {url} não é a tabela de constituintes (date, tickers)")
        return content

    def members_on(self, date) -> list[str]:
        pos = self._positions(pd.Timestamp(date))
        return [] if pos < 0 else list(self.snapshots["tickers"].iloc[pos])

    def tickers_since(self, start) -> list[str]:
        first = max(self._positions(pd.Timestamp(start)), 0)
        return sorted(set().union(*self.snapshots["tickers"].iloc[first:]))

    def membership(self, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
        column = {ticker: j for j, ticker in enumerate(tickers)}
        table = np.zeros((len(self.snapshots), len(tickers)), dtype=bool)
        for i, members in enumerate(self.snapshots["tickers"]):
            table[i, [column[t] for t in members if t in column]] = True
        pos = self._positions(dates)
        values = table[np.clip(pos, 0, None)]
        values[pos < 0] = False
        return pd.DataFrame(values, index=dates, columns=tickers)

    def sizes(self, dates: pd.DatetimeIndex) -> np.ndarray:
        pos = self._positions(dates)
        counts = self.snapshots["tickers"].map(len).to_numpy()
        return np.where(pos >= 0, counts[np.clip(pos, 0, None)], 0)

    def _positions(self, dates) -> np.ndarray:
        return self.snapshots["date"].searchsorted(dates, side="right") - 1
