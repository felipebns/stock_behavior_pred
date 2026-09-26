"""Daily bars from Yahoo adjusted for splits only; dividends come separately and are added back into returns."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_FIELDS = ["open", "high", "low", "close", "volume", "dividends", "splits"]
PANEL_FIELDS = ["open", "high", "low", "close", "volume", "dividends"]
_YAHOO_FIELDS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Volume": "volume", "Dividends": "dividends", "Stock Splits": "splits",
}


def download_prices(tickers: list[str], start: str, end: str | None = None,
                    batch_size: int = 100, downloader=None) -> tuple[pd.DataFrame, list[str]]:
    """'Adj Close' is dropped here: it is back-adjusted with dividends paid after each date."""
    if downloader is None:
        import yfinance as yf
        downloader = yf.download
    frames = []
    for i in range(0, len(tickers), batch_size):
        raw = downloader(
            list(tickers[i:i + batch_size]), start=start, end=end, interval="1d",
            auto_adjust=False, actions=True, group_by="ticker", multi_level_index=True,
            progress=False, threads=True,
        )
        if not raw.empty:
            frames.append(_to_long(raw))
    if not frames:
        raise RuntimeError("yfinance não devolveu dados para nenhum ticker")
    prices = clean_prices(pd.concat(frames, ignore_index=True))
    return prices, sorted(set(tickers) - set(prices["ticker"]))


def _to_long(raw: pd.DataFrame) -> pd.DataFrame:
    long = raw.stack(level="Ticker").rename(columns=_YAHOO_FIELDS)
    long.index = long.index.set_names(["date", "ticker"])
    return long.reindex(columns=PRICE_FIELDS).reset_index()


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.assign(date=pd.to_datetime(prices["date"]))
    out = out.dropna(subset=["close"]).drop_duplicates(["date", "ticker"], keep="last")
    out.loc[out["open"] <= 0, "open"] = np.nan
    out[["dividends", "splits"]] = out[["dividends", "splits"]].fillna(0.0)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def save_prices(prices: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path, index=False)


def load_prices(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def trading_calendar(prices: pd.DataFrame, benchmark: str, start=None) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(prices.loc[prices["ticker"] == benchmark, "date"].unique()).sort_values()
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    return dates.rename("date")


@dataclass(frozen=True)
class PricePanel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    dividends: pd.DataFrame


def build_panel(prices: pd.DataFrame, calendar: pd.DatetimeIndex, tickers: list[str]) -> PricePanel:
    subset = prices[prices["ticker"].isin(tickers)]
    tables = {
        field: subset.pivot(index="date", columns="ticker", values=field).reindex(index=calendar, columns=tickers)
        for field in PANEL_FIELDS
    }
    tables["dividends"] = tables["dividends"].fillna(0.0)
    return PricePanel(**tables)


def daily_total_return(close: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close with the dividend going ex that day; after a missing day, it spans back to the last close."""
    return (close + dividends) / close.ffill().shift(1) - 1.0


def forward_open_return(open_, dividends=None):
    """Row t: buy at the open of t+1, sell at the open of t+2, keeping the dividend that goes ex on t+2."""
    received = 0.0 if dividends is None else dividends.shift(-2)
    return (open_.shift(-2) + received) / open_.shift(-1) - 1.0
