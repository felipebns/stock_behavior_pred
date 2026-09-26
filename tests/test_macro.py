import numpy as np
import pandas as pd
import pytest

from engine.data import macro


def ff_bar(date, expiration, close):
    return {"date": pd.Timestamp(date), "symbol": "ZQ", "expiration": pd.Timestamp(expiration), "close": close}


def gpr_frame(columns: dict, periods: int = 60) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=periods, freq="D", name="date")
    return pd.DataFrame({k: np.broadcast_to(np.asarray(v, dtype=float), periods).copy() for k, v in columns.items()}, index=index)


def test_load_ff_futures_normalizes_columns(tmp_path):
    raw = pd.DataFrame({"Date": ["2020-01-02"], "symbol": ["ZQF0"], "expiration": ["2020-01-31"],
                        "Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [98.45], "Volume": [10]})
    raw.to_csv(tmp_path / "ff.csv", index=False)
    bars = macro.load_ff_futures(tmp_path / "ff.csv")
    assert list(bars.columns) == ["date", "symbol", "expiration", "close"]
    assert bars["date"].iloc[0] == pd.Timestamp("2020-01-02")


def test_ff_rates_picks_12m_contract_front_month_and_drops_sundays():
    bars = pd.DataFrame([
        ff_bar("2020-01-02", "2020-01-31", 98.45),
        ff_bar("2020-01-02", "2020-12-31", 98.60),
        ff_bar("2020-01-02", "2021-01-29", 98.62),
        ff_bar("2020-01-05", "2020-01-31", 90.00),
        ff_bar("2020-01-05", "2020-12-31", 90.00),
    ])
    rates = macro.ff_rates(bars, change_days=1)
    assert list(rates.index) == [pd.Timestamp("2020-01-02")]
    row = rates.loc["2020-01-02"]
    assert row["ff_rate_12m"] == pytest.approx(0.0140)
    assert row["ff_rate_front"] == pytest.approx(0.0155)
    assert row["ff_slope"] == pytest.approx(0.0140 - 0.0155)


def test_ff_rates_drops_dates_without_contract_near_horizon():
    bars = pd.DataFrame([ff_bar("2020-01-02", "2020-01-31", 98.45), ff_bar("2020-01-02", "2020-06-30", 98.50)])
    assert macro.ff_rates(bars).empty


def test_ff_rates_are_causal():
    days = pd.bdate_range("2020-01-01", periods=60)
    bars = pd.DataFrame([
        ff_bar(day, day + pd.offsets.MonthEnd(m), 99.0 - 0.01 * k - 0.05 * m)
        for k, day in enumerate(days) for m in range(15)
    ])
    cutoff = days[40]
    full = macro.ff_rates(bars, change_days=5)
    part = macro.ff_rates(bars[bars["date"] <= cutoff], change_days=5)
    pd.testing.assert_frame_equal(full.loc[:cutoff], part, check_exact=True)


def test_ff_rate_change_is_a_difference_not_pct_change():
    bars = pd.DataFrame([
        ff_bar(day, expiration, close)
        for day, close in [("2020-01-02", 99.90), ("2020-01-03", 99.80)]
        for expiration in ["2020-01-31", "2020-12-31"]
    ])
    rates = macro.ff_rates(bars, change_days=1)
    assert rates["ff_rate_12m_chg"].iloc[1] == pytest.approx(0.001)


def test_parse_gpr_selects_and_renames():
    raw = pd.DataFrame({"DAY": [20200101], "N10D": [1], "GPRD": [100.0], "GPRD_ACT": [50.0],
                        "GPRD_THREAT": [80.0], "date": [pd.Timestamp("2020-01-01")], "event": [None]})
    out = macro.parse_gpr(raw)
    assert list(out.columns) == ["gpr", "act", "threat"]
    assert out.index[0] == pd.Timestamp("2020-01-01")


def test_gpr_features_on_constant_series():
    last = macro.gpr_features(gpr_frame({"gpr": 100.0, "act": 50.0, "threat": 100.0})).iloc[-1]
    assert last["gpr_level"] == pytest.approx(np.log(100.0))
    assert last["gpr_trend"] == pytest.approx(0.0)
    assert last["gpr_threat_act"] == pytest.approx(np.log(2.0))


def test_gpr_features_are_causal():
    rng = np.random.default_rng(0)
    gpr = gpr_frame({"gpr": rng.uniform(50, 200, 60), "act": rng.uniform(20, 100, 60), "threat": rng.uniform(20, 100, 60)})
    cutoff = gpr.index[40]
    pd.testing.assert_frame_equal(macro.gpr_features(gpr).loc[:cutoff], macro.gpr_features(gpr.loc[:cutoff]), check_exact=True)


def test_align_asof_respects_lag_and_staleness():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-10"]))
    dates = pd.DatetimeIndex(pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-05", "2020-01-09", "2020-01-10"]))
    np.testing.assert_array_equal(macro.align_asof(frame, dates)["x"], [np.nan, 2.0, 2.0, 2.0, 3.0])
    np.testing.assert_array_equal(macro.align_asof(frame, dates, lag_days=1)["x"], [np.nan, 1.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(macro.align_asof(frame, dates, max_staleness_days=2)["x"], [np.nan, 2.0, np.nan, np.nan, 3.0])


def test_align_asof_never_reads_rows_after_t_minus_lag():
    frame = pd.DataFrame({"x": np.arange(30.0)}, index=pd.date_range("2020-01-01", periods=30, freq="D"))
    dates = pd.date_range("2020-01-05", periods=20, freq="D")
    base = macro.align_asof(frame, dates, lag_days=7)["x"].to_numpy()
    for i, t in enumerate(dates):
        changed = frame.copy()
        changed.loc[changed.index > t - pd.Timedelta(days=7), "x"] = -999.0
        np.testing.assert_array_equal(macro.align_asof(changed, dates[i:i + 1], lag_days=7)["x"].to_numpy(), base[i:i + 1])


def test_risk_free_daily_converts_percent_yield():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "ticker": "^IRX", "close": [5.04, 5.04]})
    dates = pd.DatetimeIndex(pd.to_datetime(["2020-01-03", "2020-01-06"]))
    rf = macro.risk_free_daily(prices, "^IRX", dates, max_staleness_days=5)
    assert rf.tolist() == pytest.approx([0.0504 / 252, 0.0504 / 252])
    assert rf.name == "rf_daily"
