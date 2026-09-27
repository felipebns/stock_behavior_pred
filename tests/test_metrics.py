import numpy as np
import pandas as pd
import pytest

from engine.metrics import drawdown, model_metrics, monthly_auc, performance_metrics


def series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)), dtype=float)


def test_drawdown_counts_initial_capital():
    assert drawdown(series([0.1, -0.2, 0.05])).min() == pytest.approx(0.88 / 1.1 - 1)
    assert drawdown(series([-0.1, 0.05])).tolist() == pytest.approx([-0.1, 0.9 * 1.05 - 1])
    assert drawdown(series([0.01, 0.02])).min() == 0.0


def test_performance_metrics():
    r = series([0.02, 0.0, 0.02, 0.0])
    m = performance_metrics(r, series([0.0] * 4))
    spread = np.std([0.02, 0.0, 0.02, 0.0], ddof=1)
    assert list(m) == ["total_return", "volatility", "sharpe", "max_drawdown", "hit_ratio"]
    assert m["total_return"] == pytest.approx(1.02 ** 2 - 1)
    assert m["volatility"] == pytest.approx(spread * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(0.01 / spread * np.sqrt(252))
    assert m["max_drawdown"] == 0.0
    assert m["hit_ratio"] == pytest.approx(0.5)


def test_sharpe_uses_excess_over_the_risk_free_rate():
    r = series([0.03, -0.01, 0.02, -0.02])
    rf = series([0.001] * 4)
    excess = r - rf
    assert performance_metrics(r, rf)["sharpe"] == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(252))


def test_hit_ratio_ignores_cash_days_when_invested_mask_given():
    r = series([0.01, 0.0001, -0.01, 0.02])
    invested = series([1, 0, 1, 1]).astype(bool)
    assert performance_metrics(r, series([0.0001] * 4), invested=invested)["hit_ratio"] == pytest.approx(2 / 3)


def test_empty_returns_do_not_crash():
    m = performance_metrics(series([]), series([]))
    assert m["total_return"] == 0.0 and np.isnan(m["sharpe"]) and np.isnan(m["hit_ratio"])


def test_model_metrics_on_perfect_ranking():
    predictions = pd.DataFrame({
        "date": np.repeat(pd.bdate_range("2022-01-03", periods=3), 4),
        "prob": np.tile([0.2, 0.4, 0.6, 0.8], 3),
        "fwd_ret": np.tile([-0.02, -0.01, 0.01, 0.02], 3),
    })
    predictions["label"] = (predictions["fwd_ret"] > 0).astype(float)
    assert model_metrics(predictions) == pytest.approx({"auc": 1.0, "accuracy": 1.0, "ic": 1.0})


def test_monthly_auc():
    predictions = pd.DataFrame({
        "date": np.repeat(pd.to_datetime(["2022-01-10", "2022-02-10"]), 4),
        "prob": [0.2, 0.4, 0.6, 0.8] * 2,
        "label": [0, 0, 1, 1, 1, 1, 0, 0],
    })
    assert monthly_auc(predictions).tolist() == pytest.approx([1.0, 0.0])
