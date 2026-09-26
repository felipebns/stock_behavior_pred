import numpy as np
import pandas as pd
import pytest

from engine.metrics import (
    annual_returns,
    calibration_table,
    drawdown,
    max_drawdown,
    model_metrics,
    performance_metrics,
    position_hit_ratio,
    rolling_auc,
)


def series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)), dtype=float)


def test_max_drawdown_counts_initial_capital():
    assert max_drawdown(series([0.1, -0.2, 0.05])) == pytest.approx(0.88 / 1.1 - 1)
    assert max_drawdown(series([-0.1, 0.05])) == pytest.approx(-0.1)
    assert max_drawdown(series([0.01, 0.02])) == 0.0
    assert drawdown(series([-0.1, 0.05])).tolist() == pytest.approx([-0.1, 0.9 * 1.05 - 1])


def test_return_volatility_sharpe_sortino():
    r = series([0.02, 0.0, 0.02, 0.0])
    m = performance_metrics(r, series([0.0] * 4))
    spread = np.std([0.02, 0.0, 0.02, 0.0], ddof=1)
    assert m["total_return"] == pytest.approx(1.02 ** 2 - 1)
    assert m["cagr"] == pytest.approx((1.02 ** 2) ** (252 / 4) - 1)
    assert m["volatility"] == pytest.approx(spread * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(0.01 / spread * np.sqrt(252))
    assert np.isnan(m["sortino"])
    assert m["hit_ratio"] == pytest.approx(0.5)
    assert np.isnan(m["calmar"])


def test_sortino_uses_downside_deviation():
    r = series([0.03, -0.01, 0.02, -0.02])
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    assert performance_metrics(r, series([0.0] * 4))["sortino"] == pytest.approx(r.mean() / downside * np.sqrt(252))


def test_hit_ratio_ignores_cash_days_when_invested_mask_given():
    r = series([0.01, 0.0001, -0.01, 0.02])
    invested = series([1, 0, 1, 1]).astype(bool)
    assert performance_metrics(r, series([0.0001] * 4), invested=invested)["hit_ratio"] == pytest.approx(2 / 3)


def test_beta_alpha_tracking_error():
    b = series(np.random.default_rng(0).normal(0.0005, 0.01, 300))
    m = performance_metrics(2 * b, series(np.zeros(300)), benchmark=b)
    assert m["beta"] == pytest.approx(2.0)
    assert m["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert m["tracking_error"] == pytest.approx(b.std(ddof=1) * np.sqrt(252))
    assert m["information_ratio"] == pytest.approx(b.mean() * 252 / (b.std(ddof=1) * np.sqrt(252)))


def test_model_metrics_on_perfect_ranking():
    predictions = pd.DataFrame({
        "date": np.repeat(pd.bdate_range("2022-01-03", periods=3), 4),
        "prob": np.tile([0.2, 0.4, 0.6, 0.8], 3),
        "fwd_ret": np.tile([-0.02, -0.01, 0.01, 0.02], 3),
    })
    predictions["label"] = (predictions["fwd_ret"] > 0).astype(float)
    m = model_metrics(predictions)
    assert m["auc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["base_rate"] == pytest.approx(0.5)
    assert m["ic_mean"] == pytest.approx(1.0)
    assert m["n_predictions"] == 12


def test_calibration_table_has_bins():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 1000)
    table = calibration_table(pd.DataFrame({"prob": p, "label": (rng.uniform(size=1000) < p).astype(float)}))
    assert len(table) == 10
    assert table["n"].sum() == 1000
    assert table["prob_mean"].is_monotonic_increasing


def test_annual_returns_and_position_hit_ratio():
    r = pd.Series([0.1, 0.1, -0.5], index=pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-03"]))
    yearly = annual_returns(r)
    assert yearly.loc[2021] == pytest.approx(0.21)
    assert yearly.loc[2022] == pytest.approx(-0.5)
    assert position_hit_ratio(pd.DataFrame({"fwd_ret": [0.01, -0.01, np.nan, 0.02]})) == pytest.approx(2 / 3)


def test_rolling_auc_is_monthly():
    predictions = pd.DataFrame({
        "date": np.repeat(pd.to_datetime(["2022-01-10", "2022-02-10"]), 4),
        "prob": [0.2, 0.4, 0.6, 0.8] * 2,
        "label": [0, 0, 1, 1, 1, 1, 0, 0],
    })
    assert rolling_auc(predictions).tolist() == pytest.approx([1.0, 0.0])
