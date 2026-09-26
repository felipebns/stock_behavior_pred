import numpy as np
import pandas as pd
import pytest

from config.config import LGBM_PARAMS
from engine.model import make_model
from engine.walk_forward import walk_forward

DATES = pd.bdate_range("2021-01-01", periods=15, name="date")


def spy_dataset() -> pd.DataFrame:
    rows = [
        {"date": day, "ticker": ticker, "day": float(k), "noise": 0.0, "fwd_ret": 0.01, "label": float(k % 2)}
        for k, day in enumerate(DATES) if k >= 2
        for ticker in ("A", "B")
    ]
    return pd.DataFrame(rows).set_index(["date", "ticker"])


class Spy:
    """Records which calendar days (feature 'day') each fit and predict sees."""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y):
        self.log.append(("fit", sorted({int(v) for v in X[:, 0]})))
        self.feature_importances_ = np.array([3.0, 1.0])
        return self

    def predict_proba(self, X):
        self.log.append(("predict", sorted({int(v) for v in X[:, 0]})))
        p = np.full(len(X), 0.7)
        return np.column_stack([1 - p, p])


def run_spy(dataset, window=4, **kwargs):
    log = []
    result = walk_forward(dataset, ["day", "noise"], DATES, window, lambda: Spy(log), **kwargs)
    return result, log


def test_trains_on_exactly_the_labels_known_at_decision_time():
    _, log = run_spy(spy_dataset())
    fits = [days for kind, days in log if kind == "fit"]
    predicts = [days for kind, days in log if kind == "predict"]
    decisions = list(range(2 + 4 + 1, len(DATES)))
    assert predicts == [[k] for k in decisions]
    assert fits == [list(range(k - 4 - 1, k - 1)) for k in decisions]


def test_rows_with_unknown_label_are_not_used_for_training():
    dataset = spy_dataset()
    dataset.loc[dataset.index.get_level_values("date") == DATES[5], "label"] = np.nan
    _, log = run_spy(dataset)
    fits = [days for kind, days in log if kind == "fit"]
    assert fits[0] == [2, 3, 4]
    assert all(5 not in days for days in fits)


def test_outputs_predictions_and_importance():
    result, _ = run_spy(spy_dataset())
    predictions = result.predictions
    assert list(predictions.columns) == ["date", "ticker", "prob", "fwd_ret", "label"]
    assert predictions["date"].min() == DATES[7]
    assert len(predictions) == 2 * (len(DATES) - 7)
    assert (predictions["prob"] == 0.7).all()
    assert list(result.importance.columns) == ["day", "noise"]
    assert result.importance.iloc[0].tolist() == [0.75, 0.25]
    assert list(result.importance.index) == list(DATES[7:])


def test_start_and_end_limit_decisions():
    result, _ = run_spy(spy_dataset(), start=DATES[9], end=DATES[11])
    assert sorted(result.predictions["date"].unique()) == list(DATES[9:12])


def test_no_decision_dates_raises():
    with pytest.raises(ValueError):
        run_spy(spy_dataset(), window=50)


def test_lightgbm_model_is_deterministic_and_gives_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 3))
    y = (X[:, 0] + rng.normal(0.0, 1.0, 3000) > 0).astype(int)
    first = make_model(LGBM_PARAMS).fit(X, y).predict_proba(X)[:, 1]
    second = make_model(LGBM_PARAMS).fit(X, y).predict_proba(X)[:, 1]
    np.testing.assert_array_equal(first, second)
    assert ((first > 0) & (first < 1)).all()
