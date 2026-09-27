"""Point-in-time features: every value at date t uses only data dated on or before t."""
import warnings

import numpy as np
import pandas as pd

from engine.prices import PricePanel, daily_total_return

STOCK_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "mom_12_1", "gap_1d", "intraday_1d",
    "vol_21d", "vol_63d", "vol_ratio", "range_21d", "dist_ma50", "dist_ma200",
    "dist_high_252", "rsi_14", "volume_ratio", "log_dollar_volume", "div_yield_252", "beta_63d",
]
XS_SOURCES = ["ret_1d", "ret_5d", "ret_21d", "mom_12_1", "vol_21d"]
XS_FEATURES = [f"xs_{name}" for name in XS_SOURCES]
PEER_FEATURES = ["peer_ret_1d", "peer_ret_5d", "peer_ret_21d", "peer_gap_5d"]
MARKET_FEATURES = ["mkt_ret_1d", "mkt_ret_5d", "mkt_ret_21d", "mkt_vol_21d", "mkt_dist_ma200", "breadth_ma50"]
TICKER_FEATURES = STOCK_FEATURES + XS_FEATURES + PEER_FEATURES
DATE_FEATURES = MARKET_FEATURES
FEATURES = TICKER_FEATURES + DATE_FEATURES


def _rolling(frame, window: int, stat: str):
    return getattr(frame.rolling(window, min_periods=int(np.ceil(0.8 * window))), stat)()


class FeatureBuilder:
    def __init__(self, peer_count: int = 10, peer_lookback: int = 63):
        self.peer_count = peer_count
        self.peer_lookback = peer_lookback

    def build(self, panel: PricePanel, market_close: pd.Series,
              membership: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        stock = self.stock(panel, market_close)
        ticker = {**stock, **self.cross_section(stock, membership), **self.peers(stock, membership)}
        return ticker, self.market(market_close, stock, membership)

    @staticmethod
    def stock(panel: PricePanel, market_close: pd.Series) -> dict[str, pd.DataFrame]:
        close, open_, dividends = panel.close, panel.open, panel.dividends
        prev_close = close.ffill().shift(1)
        r = daily_total_return(close, dividends)
        tri = (1.0 + r.fillna(0.0)).cumprod().where(close.ffill().notna())
        m = market_close / market_close.shift(1) - 1.0
        m_wide = pd.DataFrame(np.repeat(m.to_numpy()[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns).where(r.notna())

        f = {"ret_1d": r}
        for k in (5, 21, 63):
            f[f"ret_{k}d"] = tri / tri.shift(k) - 1.0
        f["mom_12_1"] = tri.shift(21) / tri.shift(252) - 1.0
        f["gap_1d"] = (open_ + dividends) / prev_close - 1.0
        f["intraday_1d"] = close / open_ - 1.0
        f["vol_21d"] = _rolling(r, 21, "std")
        f["vol_63d"] = _rolling(r, 63, "std")
        f["vol_ratio"] = f["vol_21d"] / f["vol_63d"]
        f["range_21d"] = _rolling((panel.high - panel.low) / close, 21, "mean")
        f["dist_ma50"] = tri / _rolling(tri, 50, "mean") - 1.0
        f["dist_ma200"] = tri / _rolling(tri, 200, "mean") - 1.0
        f["dist_high_252"] = close / _rolling(close, 252, "max") - 1.0
        gain = _rolling(r.clip(lower=0.0), 14, "mean")
        loss = _rolling((-r).clip(lower=0.0), 14, "mean")
        f["rsi_14"] = 100.0 * gain / (gain + loss)
        f["volume_ratio"] = panel.volume / _rolling(panel.volume, 21, "mean")
        with np.errstate(divide="ignore"):
            f["log_dollar_volume"] = np.log(_rolling(close * panel.volume, 21, "mean"))
        f["div_yield_252"] = dividends.rolling(252, min_periods=1).sum() / close
        mean_r, mean_m = _rolling(r, 63, "mean"), _rolling(m_wide, 63, "mean")
        covariance = _rolling(r * m_wide, 63, "mean") - mean_r * mean_m
        variance = _rolling(m_wide * m_wide, 63, "mean") - mean_m * mean_m
        f["beta_63d"] = covariance / variance
        return {name: f[name].replace([np.inf, -np.inf], np.nan) for name in STOCK_FEATURES}

    @staticmethod
    def cross_section(stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {f"xs_{name}": stock[name].where(membership).rank(axis=1, pct=True) for name in XS_SOURCES}

    def peers(self, stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """For each member at t: mean return of the peer_count members whose last peer_lookback daily returns correlate most with its own."""
        returns, member = stock["ret_1d"].to_numpy(), membership.to_numpy()
        horizons = {name: stock[name].to_numpy() for name in ("ret_1d", "ret_5d", "ret_21d")}
        out = {name: np.full(returns.shape, np.nan) for name in horizons}
        lookback, count = self.peer_lookback, self.peer_count
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(lookback - 1, len(returns)):
                window = returns[i - lookback + 1:i + 1]
                cols = np.flatnonzero(member[i] & (np.isfinite(window).sum(axis=0) >= 0.8 * lookback))
                if len(cols) <= count:
                    continue
                z = window[:, cols]
                z = np.nan_to_num((z - np.nanmean(z, axis=0)) / np.nanstd(z, axis=0))
                corr = z.T @ z / lookback
                np.fill_diagonal(corr, -np.inf)
                peers = np.argpartition(-corr, count - 1, axis=1)[:, :count]
                for name, values in horizons.items():
                    out[name][i, cols] = np.nanmean(values[i, cols][peers], axis=1)
        frames = {f"peer_{name}": pd.DataFrame(values, index=membership.index, columns=membership.columns)
                  for name, values in out.items()}
        frames["peer_gap_5d"] = stock["ret_5d"] - frames["peer_ret_5d"]
        return {name: frames[name] for name in PEER_FEATURES}

    @staticmethod
    def market(market_close: pd.Series, stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> pd.DataFrame:
        m = market_close / market_close.shift(1) - 1.0
        dist = stock["dist_ma50"]
        above = (dist > 0).astype(float).where(dist.notna() & membership)
        return pd.DataFrame({
            "mkt_ret_1d": m,
            "mkt_ret_5d": market_close / market_close.shift(5) - 1.0,
            "mkt_ret_21d": market_close / market_close.shift(21) - 1.0,
            "mkt_vol_21d": _rolling(m, 21, "std"),
            "mkt_dist_ma200": market_close / _rolling(market_close, 200, "mean") - 1.0,
            "breadth_ma50": above.mean(axis=1),
        }, index=market_close.index)
