"""Macro inputs aligned to trading days without lookahead: fed funds futures, GPR and the risk-free rate."""
import numpy as np
import pandas as pd

FF_FEATURES = ["ff_rate_12m", "ff_rate_12m_chg", "ff_slope"]
GPR_FEATURES = ["gpr_level", "gpr_trend", "gpr_threat_act"]


def parse_ff_futures(raw: pd.DataFrame) -> pd.DataFrame:
    bars = raw.rename(columns=str.lower)
    bars = bars.assign(date=pd.to_datetime(bars["date"]), expiration=pd.to_datetime(bars["expiration"]))
    return bars[["date", "symbol", "expiration", "close"]]


def load_ff_futures(path) -> pd.DataFrame:
    return parse_ff_futures(pd.read_csv(path))


def ff_rates(bars: pd.DataFrame, horizon_months: int = 12, max_gap_days: int = 45, change_days: int = 21) -> pd.DataFrame:
    """Daily bars are UTC days, so each one closes (~19-20h ET) before the next US open."""
    df = bars[bars["date"].dt.dayofweek != 6].dropna(subset=["close"])  # Sunday bars are partial Globex sessions
    df = df.assign(rate=(100.0 - df["close"]) / 100.0)
    targets = {day: day + pd.DateOffset(months=horizon_months) for day in df["date"].unique()}
    df = df.assign(gap=(df["expiration"] - df["date"].map(targets)).abs())
    nearest = df.sort_values(["date", "gap"]).groupby("date").first()
    rate_12m = nearest.loc[nearest["gap"] <= pd.Timedelta(days=max_gap_days), "rate"]
    live = df[df["expiration"] >= df["date"]]
    rate_front = live.sort_values(["date", "expiration"]).groupby("date")["rate"].first()
    out = pd.DataFrame({"ff_rate_12m": rate_12m, "ff_rate_front": rate_front}).dropna().sort_index()
    out["ff_rate_12m_chg"] = out["ff_rate_12m"].diff(change_days)
    out["ff_slope"] = out["ff_rate_12m"] - out["ff_rate_front"]
    return out


def parse_gpr(raw: pd.DataFrame) -> pd.DataFrame:
    gpr = raw[["date", "GPRD", "GPRD_ACT", "GPRD_THREAT"]].dropna(subset=["date"])
    gpr = gpr.rename(columns={"GPRD": "gpr", "GPRD_ACT": "act", "GPRD_THREAT": "threat"})
    return gpr.assign(date=pd.to_datetime(gpr["date"])).set_index("date").sort_index()


def load_gpr(path) -> pd.DataFrame:
    return parse_gpr(pd.read_excel(path))


def gpr_features(gpr: pd.DataFrame) -> pd.DataFrame:
    """Trailing calendar-day means ending at each GPR date; the publication lag is applied by the caller."""
    daily = gpr.asfreq("D")
    mean_7 = daily.rolling(7, min_periods=6).mean()
    mean_30 = daily["gpr"].rolling(30, min_periods=24).mean()
    with np.errstate(divide="ignore"):
        out = pd.DataFrame({
            "gpr_level": np.log(mean_7["gpr"]),
            "gpr_trend": np.log(mean_7["gpr"]) - np.log(mean_30),
            "gpr_threat_act": np.log(mean_7["threat"]) - np.log(mean_7["act"]),
        })
    return out.replace([np.inf, -np.inf], np.nan)


def align_asof(frame: pd.DataFrame, dates, lag_days: int = 0, max_staleness_days: int | None = None) -> pd.DataFrame:
    """Row t = last row of `frame` dated on or before t - lag_days; NaN if none or if it is older than the limit."""
    frame = frame.sort_index()
    dates = pd.DatetimeIndex(dates)
    lookup = dates - pd.Timedelta(days=lag_days)
    pos = frame.index.searchsorted(lookup, side="right") - 1
    safe = np.clip(pos, 0, None)
    values = frame.to_numpy(dtype=float)[safe]
    invalid = pos < 0
    if max_staleness_days is not None:
        invalid |= np.asarray((lookup - frame.index[safe]) > pd.Timedelta(days=max_staleness_days))
    values[invalid] = np.nan
    return pd.DataFrame(values, index=dates, columns=frame.columns)


def risk_free_daily(prices: pd.DataFrame, ticker: str, dates, max_staleness_days: int) -> pd.Series:
    """^IRX closes in % per year; one trading day of carry is rate / 100 / 252."""
    yields = prices.loc[prices["ticker"] == ticker].set_index("date")["close"].sort_index().to_frame("rate")
    return (align_asof(yields, dates, 0, max_staleness_days)["rate"] / 100.0 / 252.0).rename("rf_daily")
