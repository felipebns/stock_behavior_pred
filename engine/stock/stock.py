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
        
    def build_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
        df = df.copy()
        
        # Sort by ticker and date just in case
        df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)
        
        target_col = "target_direction"
        # Target: predict if close will go up (using close as market sees it)
        df[target_col] = (df.groupby("ticker")["close"].shift(-1) > df["close"]).astype(int)

        features = []

        # ============================================================
        # 1. PRICE-BASED FEATURES (Close vs Adj Close comparison)
        # ============================================================
        
        # Returns from both price series (let model choose)
        df["return_close"] = df.groupby("ticker")["close"].pct_change()
        df["return_adj"] = df.groupby("ticker")["adj_close"].pct_change()
        features.extend(["return_close", "return_adj"])
        
        # Divergence between close and adj_close (splits/dividends indicator)
        df["price_divergence"] = (df["close"] - df["adj_close"]) / df["adj_close"]
        features.append("price_divergence")

        # ============================================================
        # 2. TREND INDICATORS (EMA, SMA)
        # ============================================================
        
        # EMA using close
        df["ema_10"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=10, adjust=False).mean())
        df["ema_20"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        features.extend(["ema_10", "ema_20"])
        
        # Position relative to EMA
        df["close_vs_ema10"] = (df["close"] - df["ema_10"]) / df["ema_10"]
        df["close_vs_ema20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
        features.extend(["close_vs_ema10", "close_vs_ema20"])

        # ============================================================
        # 3. MOMENTUM INDICATORS (RSI, MACD)
        # ============================================================
        
        # RSI (14 periods)
        delta = df.groupby("ticker")["close"].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        df["gain_roll"] = gain.groupby(df["ticker"]).transform(lambda x: x.rolling(window=14).mean())
        df["loss_roll"] = loss.groupby(df["ticker"]).transform(lambda x: x.rolling(window=14).mean())
        rs = df["gain_roll"] / df["loss_roll"]
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50)
        df.drop(columns=["gain_roll", "loss_roll"], inplace=True)
        features.append("rsi_14")
        
        # MACD (12, 26, 9) - signal momentum
        ema_12 = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema_26 = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        features.extend(["macd", "macd_hist"])

        # ============================================================
        # 4. VOLATILITY INDICATORS
        # ============================================================
        
        # Historical Volatility (20 periods)
        df["volatility_20"] = df.groupby("ticker")["return_close"].transform(lambda x: x.rolling(window=20).std())
        features.append("volatility_20")
        
        # Average True Range (ATR) - normalized
        df["high_low"] = df["high"] - df["low"]
        df["high_close"] = np.abs(df["high"] - df.groupby("ticker")["close"].shift(1))
        df["low_close"] = np.abs(df["low"] - df.groupby("ticker")["close"].shift(1))
        df["true_range"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
        df["atr_14"] = df.groupby("ticker")["true_range"].transform(lambda x: x.rolling(window=14).mean())
        df["atr_normalized"] = df["atr_14"] / df["close"]
        df.drop(columns=["high_low", "high_close", "low_close", "true_range"], inplace=True)
        features.append("atr_normalized")

        # ============================================================
        # 5. VOLUME INDICATORS
        # ============================================================
        
        # Volume relative to average
        df["volume_ma"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(window=20).mean())
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        features.append("volume_ratio")
        
        # On-Balance Volume (OBV) normalized
        df["obv"] = (np.sign(df.groupby("ticker")["close"].diff()) * df["volume"]).groupby(df["ticker"]).cumsum()
        df["obv_ma"] = df.groupby("ticker")["obv"].transform(lambda x: x.rolling(window=20).mean())
        df["obv_signal"] = (df["obv"] - df["obv_ma"]) / (df["obv_ma"] + 1)  # +1 to avoid division by 0
        df.drop(columns=["obv", "obv_ma"], inplace=True)
        features.append("obv_signal")

        # ============================================================
        # 6. MEAN REVERSION INDICATORS
        # ============================================================
        
        # Bollinger Bands (20 periods) - using close
        sma = df.groupby("ticker")["close"].transform(lambda x: x.rolling(window=20).mean())
        std = df.groupby("ticker")["close"].transform(lambda x: x.rolling(window=20).std())
        df["bb_upper"] = sma + (2 * std)
        df["bb_lower"] = sma - (2 * std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        # Bollinger Bands %B (position within bands)
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_pct"] = df["bb_pct"].clip(0, 1)  # Clip between 0 and 1
        features.extend(["bb_width", "bb_pct"])

        # ============================================================
        # 7. RATE OF CHANGE & MOMENTUM
        # ============================================================
        
        # ROC (10 periods)
        shifted_10 = df.groupby("ticker")["close"].shift(10)
        df["roc_10"] = ((df["close"] - shifted_10) / shifted_10) * 100
        features.append("roc_10")
        
        # Momentum (simple price difference)
        df["momentum_10"] = (df["close"] - shifted_10) / df["close"]  # Normalized
        features.append("momentum_10")

        # ============================================================
        # 8. LAGGED RETURNS
        # ============================================================
        
        # Lagged returns (memory of recent performance)
        for i in range(1, 4):  # Reduced from 6 to avoid overfitting
            col = f"return_lag_{i}"
            df[col] = df.groupby("ticker")["return_close"].shift(i)
            features.append(col)

        # ============================================================
        # PREPROCESSING & FINAL SETUP
        # ============================================================
        
        # Precompute next return using ADJUSTED CLOSE for backtesting accuracy
        # (adj_close reflects true investor returns with splits/dividends)
        df["next_return"] = df.groupby("ticker")["return_adj"].shift(-1)

        # Drop rows with NaN due to rolling/shift windows
        df = df.dropna(subset=features).copy()
        
        # Drop the last row of each ticker (invalid target)
        df = df.drop(df.groupby("ticker").tail(1).index).reset_index(drop=True)
        
        # Sort by date then ticker for time-based splits
        df = df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
        
        return df, features, target_col

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=1, future_stack=True).reset_index()
        else:
            df = df.reset_index()
            df["Ticker"] = self.tickers[0] if isinstance(self.tickers, list) else self.tickers
            
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            
        if "adj_close" not in df.columns or df["adj_close"].isna().all():
            raise ValueError("adj_close is required for this model")

        if "date" not in df.columns:
            raise ValueError("date column missing after normalization")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")

        df = df.dropna(subset=["close", "adj_close"]).reset_index(drop=True)

        df["adj_factor"] = df["adj_close"] / df["close"]

        df["adj_open"] = df["open"] * df["adj_factor"]
        df["adj_high"] = df["high"] * df["adj_factor"]
        df["adj_low"] = df["low"] * df["adj_factor"]

        return df[["date", "ticker", "open", "high", "low", "close", 
                "adj_open", "adj_high", "adj_low", "adj_close", "volume"]]