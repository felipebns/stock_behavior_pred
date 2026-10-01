import numpy as np
import pandas as pd
import pytest

from engine.metrics import daily_ic, decile_spread, deciles, drawdown, model_metrics, monthly_auc, performance_metrics


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


def test_model_metrics_without_known_outcomes_are_nan():
    empty = pd.DataFrame({"date": pd.to_datetime(["2022-01-03"] * 2), "prob": [0.4, 0.6],
                          "fwd_ret": [np.nan, np.nan], "label": [np.nan, np.nan]})
    assert all(np.isnan(value) for value in model_metrics(empty).values())
    one_class = empty.assign(fwd_ret=[0.01, 0.02], label=[1.0, 1.0])
    assert np.isnan(model_metrics(one_class)["auc"])


def ranked_predictions(n_stocks: int = 10) -> pd.DataFrame:
    """Two dates; prob rises with i; returns rise with i on the first date and fall on the second."""
    rows = [
        {"date": date, "ticker": f"T{i}", "prob": (i + 0.5) / n_stocks, "fwd_ret": sign * 0.01 * i}
        for date, sign in zip(pd.to_datetime(["2022-01-03", "2022-01-10"]), [1, -1])
        for i in range(n_stocks)
    ]
    frame = pd.DataFrame(rows)
    return frame.assign(label=(frame["fwd_ret"] > frame.groupby("date")["fwd_ret"].transform("median")).astype(float))


def test_deciles_average_within_each_date_then_across_dates():
    table = deciles(ranked_predictions())
    assert list(table.index) == list(range(1, 11))
    assert list(table.columns) == ["prob", "label", "relative"]
    assert table["prob"].tolist() == pytest.approx([(i + 0.5) / 10 for i in range(10)])
    assert table["label"].tolist() == pytest.approx([0.5] * 10)
    assert table["relative"].tolist() == pytest.approx([0.0] * 10)


def test_decile_spread_is_top_minus_bottom_per_date():
    spread = decile_spread(ranked_predictions())
    assert list(spread.index) == list(pd.to_datetime(["2022-01-03", "2022-01-10"]))
    assert spread.tolist() == pytest.approx([0.09, -0.09])


def test_decile_spread_uses_at_least_one_stock_per_side():
    assert decile_spread(ranked_predictions(5)).tolist() == pytest.approx([0.04, -0.04])


def test_rows_with_unknown_outcome_are_left_out_of_the_deciles():
    predictions = ranked_predictions()
    predictions.loc[predictions["ticker"] == "T9", ["fwd_ret", "label"]] = np.nan
    assert decile_spread(predictions).tolist() == pytest.approx([0.08, -0.08])


def test_daily_ic_and_monthly_auc_without_known_outcomes_are_empty_series():
    empty = pd.DataFrame({"date": pd.to_datetime([]), "prob": [], "fwd_ret": [], "label": []})
    for result in (daily_ic(empty), monthly_auc(empty)):
        assert isinstance(result, pd.Series) and result.empty
