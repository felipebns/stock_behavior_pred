import numpy as np
import pandas as pd
import yfinance as yf

class Stock:
    def __init__(self, tickers: list[str], start: str, end: str) -> None:
        self.tickers = tickers
        self.start = start
        self.end = end

    def fetch(self) -> pd.DataFrame:
        try:
            df = yf.download(
                tickers=self.tickers,
                start=self.start,
                end=self.end,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
            return self._normalize(df)
        except Exception as e:
            print(f"Error fetching data for {self.tickers}: {e}")
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            # Stack to get tickers into rows: transforms MultiIndex [Price, Ticker] to Index [Date, Ticker]
            # yfinance puts Date in the index
            df = df.stack(level=1, future_stack=True).reset_index()
            # The columns will now be ['Date', 'Ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        else:
            df = df.reset_index()
            df["Ticker"] = self.tickers[0] if isinstance(self.tickers, list) else self.tickers
            
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        if "price" in df.columns:
            df = df.drop(columns=["price"])
            
        if "adj_close" not in df.columns or df["adj_close"].isna().all():
            raise ValueError("adj_close is required for this model")

        if "date" not in df.columns:
            raise ValueError("date column missing after normalization")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")

        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close", "adj_close"]).reset_index(drop=True)

        df["adj_factor"] = df["adj_close"] / df["close"]

        df["adj_open"] = df["open"] * df["adj_factor"]
        df["adj_high"] = df["high"] * df["adj_factor"]
        df["adj_low"] = df["low"] * df["adj_factor"]


        return df[
            [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_open",
                "adj_high",
                "adj_low",
                "adj_close",
                "volume",
            ]
        ]