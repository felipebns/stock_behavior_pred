"""Point-in-time features: every value at date t uses only data dated on or before t."""
import numpy as np
import pandas as pd

from engine.data.macro import FF_FEATURES, GPR_FEATURES
from engine.data.prices import PricePanel, daily_total_return

STOCK_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "mom_12_1", "gap_1d", "intraday_1d",
    "vol_21d", "vol_63d", "vol_ratio", "range_21d", "dist_ma50", "dist_ma200",
    "dist_high_252", "rsi_14", "volume_ratio", "log_dollar_volume", "div_yield_252", "beta_63d",
]
XS_SOURCES = ["ret_1d", "ret_5d", "ret_21d", "mom_12_1", "vol_21d"]
XS_FEATURES = [f"xs_{name}" for name in XS_SOURCES]
MARKET_FEATURES = ["mkt_ret_1d", "mkt_ret_5d", "mkt_ret_21d", "mkt_vol_21d", "mkt_dist_ma200", "breadth_ma50"]
MACRO_FEATURES = FF_FEATURES + GPR_FEATURES
TICKER_FEATURES = STOCK_FEATURES + XS_FEATURES
DATE_FEATURES = MARKET_FEATURES + MACRO_FEATURES
FEATURES = TICKER_FEATURES + DATE_FEATURES


def _rolling(frame, window: int, stat: str):
    return getattr(frame.rolling(window, min_periods=int(np.ceil(0.8 * window))), stat)()


def stock_features(panel: PricePanel, market_close: pd.Series) -> dict[str, pd.DataFrame]:
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


def cross_sectional_features(stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {f"xs_{name}": stock[name].where(membership).rank(axis=1, pct=True) for name in XS_SOURCES}


def market_features(market_close: pd.Series, stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> pd.DataFrame:
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
