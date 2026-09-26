import numpy as np
import pandas as pd
import pytest

from engine.data.prices import PANEL_FIELDS, PricePanel
from engine.features import (
    FEATURES,
    STOCK_FEATURES,
    XS_FEATURES,
    cross_sectional_features,
    market_features,
    stock_features,
)

DATES = pd.bdate_range("2020-01-01", periods=300, name="date")


def make_panel(close: pd.DataFrame, dividends: pd.DataFrame | None = None) -> PricePanel:
    return PricePanel(
        open=close, high=close, low=close, close=close,
        volume=pd.DataFrame(1000.0, index=close.index, columns=close.columns),
        dividends=pd.DataFrame(0.0, index=close.index, columns=close.columns) if dividends is None else dividends,
    )


def test_feature_lists():
    assert len(FEATURES) == 36 and len(set(FEATURES)) == 36


def test_returns_momentum_and_trend_on_constant_growth():
    close = pd.DataFrame({"A": 100.0 * 1.01 ** np.arange(300)}, index=DATES)
    market = pd.Series(1000.0 * 1.005 ** np.arange(300), index=DATES)
    last = {name: frame["A"].iloc[-1] for name, frame in stock_features(make_panel(close), market).items()}
    assert last["ret_1d"] == pytest.approx(0.01)
    assert last["ret_5d"] == pytest.approx(1.01 ** 5 - 1)
    assert last["ret_63d"] == pytest.approx(1.01 ** 63 - 1)
    assert last["mom_12_1"] == pytest.approx(1.01 ** 231 - 1)
    assert last["vol_21d"] == pytest.approx(0.0, abs=1e-12)
    assert last["rsi_14"] == pytest.approx(100.0)
    assert last["dist_high_252"] == pytest.approx(0.0)
    assert last["intraday_1d"] == pytest.approx(0.0)


def test_dividend_counts_as_return_and_yield():
    close = pd.DataFrame({"A": np.full(300, 100.0)}, index=DATES)
    dividends = pd.DataFrame({"A": np.zeros(300)}, index=DATES)
    dividends.iloc[250, 0] = 1.0
    market = pd.Series(np.linspace(1000.0, 1100.0, 300), index=DATES)
    f = stock_features(make_panel(close, dividends), market)
    assert f["ret_1d"]["A"].iloc[250] == pytest.approx(0.01)
    assert f["gap_1d"]["A"].iloc[250] == pytest.approx(0.01)
    assert f["div_yield_252"]["A"].iloc[260] == pytest.approx(0.01)
    assert f["ret_21d"]["A"].iloc[260] == pytest.approx(0.01)


def test_beta_recovers_leverage_on_market():
    m = np.random.default_rng(1).normal(0.0, 0.01, 300)
    market = pd.Series(1000.0 * np.cumprod(1 + m), index=DATES)
    close = pd.DataFrame({"A": 50.0 * np.cumprod(1 + 2.0 * m)}, index=DATES)
    assert stock_features(make_panel(close), market)["beta_63d"]["A"].iloc[-1] == pytest.approx(2.0, rel=1e-9)


def test_cross_section_ranks_only_members():
    dates = DATES[:2]
    ret = pd.DataFrame({"A": [0.01, 0.01], "B": [0.02, 0.02], "C": [0.03, 0.03]}, index=dates)
    stock = {name: ret for name in ("ret_1d", "ret_5d", "ret_21d", "mom_12_1", "vol_21d")}
    membership = pd.DataFrame({"A": [True, True], "B": [True, False], "C": [True, True]}, index=dates)
    xs = cross_sectional_features(stock, membership)["xs_ret_1d"]
    assert xs.iloc[0].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert np.isnan(xs.iloc[1]["B"]) and xs.iloc[1][["A", "C"]].tolist() == pytest.approx([0.5, 1.0])


def test_breadth_counts_members_above_their_ma50():
    dates = DATES[:1]
    stock = {"dist_ma50": pd.DataFrame({"A": [0.1], "B": [-0.1], "C": [0.2], "D": [0.3]}, index=dates)}
    membership = pd.DataFrame({"A": [True], "B": [True], "C": [True], "D": [False]}, index=dates)
    market = market_features(pd.Series([1000.0], index=dates), stock, membership)
    assert market["breadth_ma50"].iloc[0] == pytest.approx(2 / 3)


def test_stock_cross_section_and_market_features_are_causal(synthetic_panel):
    panel, market_close, membership = synthetic_panel
    cutoff = panel.close.index[200]
    truncated = PricePanel(**{field: getattr(panel, field).loc[:cutoff] for field in PANEL_FIELDS})
    full = stock_features(panel, market_close)
    part = stock_features(truncated, market_close.loc[:cutoff])
    for name in STOCK_FEATURES:
        pd.testing.assert_frame_equal(full[name].loc[:cutoff], part[name], check_exact=True)
    full_xs = cross_sectional_features(full, membership)
    part_xs = cross_sectional_features(part, membership.loc[:cutoff])
    for name in XS_FEATURES:
        pd.testing.assert_frame_equal(full_xs[name].loc[:cutoff], part_xs[name], check_exact=True)
    pd.testing.assert_frame_equal(
        market_features(market_close, full, membership).loc[:cutoff],
        market_features(market_close.loc[:cutoff], part, membership.loc[:cutoff]),
        check_exact=True,
    )
