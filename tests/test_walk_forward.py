import numpy as np
import pandas as pd
import pytest

from config.config import LGBM_PARAMS
from engine.walk_forward import WalkForward

DATES = pd.bdate_range("2021-01-01", periods=15, name="date")


def spy_dataset() -> pd.DataFrame:
    rows = [
        {"date": day, "ticker": ticker, "day": float(k), "noise": 0.0, "fwd_ret": 0.01, "label": float(k % 2)}
        for k, day in enumerate(DATES) if k >= 2
        for ticker in ("A", "B")
    ]
    return pd.DataFrame(rows).set_index(["date", "ticker"])


class Spy:
    """Records which calendar days (feature 'day') each fit and predict sees, and the params it got."""
    log: list = []

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        Spy.log.append(("fit", sorted({int(v) for v in X[:, 0]}), self.params["min_child_samples"]))
        self.feature_importances_ = np.array([3.0, 1.0])
        return self

    def predict_proba(self, X):
        Spy.log.append(("predict", sorted({int(v) for v in X[:, 0]}), None))
        p = np.full(len(X), 0.7)
        return np.column_stack([1 - p, p])


def run_spy(dataset, window=4, start=None, **options):
    Spy.log = []
    return WalkForward(window, {}, n_jobs=1, model_class=Spy, **options).run(dataset, ["day", "noise"], DATES, start=start)


def random_dataset(n_days=60, n_tickers=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days, name="date")
    index = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(n_tickers)]], names=["date", "ticker"])
    X = rng.normal(size=(len(index), 3))
    realized = 0.01 * X[:, 0] + rng.normal(0.0, 0.02, len(index))
    dataset = pd.DataFrame(X, index=index, columns=["a", "b", "c"]).astype(np.float32)
    return dataset.assign(fwd_ret=realized, label=(realized > 0).astype(float)), dates


def test_trains_on_exactly_the_labels_known_at_decision_time():
    run_spy(spy_dataset())
    fits = [days for kind, days, _ in Spy.log if kind == "fit"]
    predicts = [days for kind, days, _ in Spy.log if kind == "predict"]
    decisions = list(range(2 + 4 + 1, len(DATES)))
    assert predicts == [[k] for k in decisions]
    assert fits == [list(range(k - 4 - 1, k - 1)) for k in decisions]


def test_rows_with_unknown_label_are_not_used_for_training():
    dataset = spy_dataset()
    dataset.loc[dataset.index.get_level_values("date") == DATES[5], "label"] = np.nan
    run_spy(dataset)
    fits = [days for kind, days, _ in Spy.log if kind == "fit"]
    assert fits[0] == [2, 3, 4]
    assert all(5 not in days for days in fits)


def test_min_child_samples_grows_with_the_training_rows():
    run_spy(spy_dataset(), min_child_share=0.5, min_child_floor=2)
    sizes = {size for kind, _, size in Spy.log if kind == "fit"}
    assert sizes == {4}
    run_spy(spy_dataset())
    assert {size for kind, _, size in Spy.log if kind == "fit"} == {20}


def test_outputs_predictions_and_importance():
    result = run_spy(spy_dataset())
    predictions = result.predictions
    assert list(predictions.columns) == ["date", "ticker", "prob", "fwd_ret", "label"]
    assert predictions["date"].min() == DATES[7]
    assert len(predictions) == 2 * (len(DATES) - 7)
    assert (predictions["prob"] == 0.7).all()
    assert list(result.importance.columns) == ["day", "noise"]
    assert result.importance.iloc[0].tolist() == [0.75, 0.25]
    assert list(result.importance.index) == list(DATES[7:])


def test_start_sets_the_first_decision():
    result = run_spy(spy_dataset(), start=DATES[9])
    assert result.predictions["date"].min() == DATES[9]


def test_window_too_long_for_the_start_date_is_refused():
    with pytest.raises(ValueError, match="máximo é 5"):
        run_spy(spy_dataset(), window=6, start=DATES[8])


def test_horizon_spaces_decisions_and_moves_the_training_rows_back():
    run_spy(spy_dataset(), window=3, horizon=2)
    fits = [days for kind, days, _ in Spy.log if kind == "fit"]
    predicts = [days for kind, days, _ in Spy.log if kind == "predict"]
    decisions = [7, 9, 11, 13]
    assert predicts == [[k] for k in decisions]
    assert fits == [list(range(k - 2 - 3, k - 2)) for k in decisions]


def test_one_fit_serves_retrain_every_consecutive_decisions():
    result = run_spy(spy_dataset(), window=3, horizon=2, retrain_every=3)
    assert [kind for kind, _, _ in Spy.log] == ["fit", "predict", "predict", "predict", "fit", "predict"]
    assert [days for kind, days, _ in Spy.log if kind == "fit"] == [[2, 3, 4], [8, 9, 10]]
    assert list(result.importance.index) == [DATES[7], DATES[13]]
    assert sorted(result.predictions["date"].unique()) == [DATES[k] for k in (7, 9, 11, 13)]


def test_the_maximum_window_accounts_for_the_horizon():
    with pytest.raises(ValueError, match="máximo é 4"):
        run_spy(spy_dataset(), window=5, horizon=2, start=DATES[8])


class Echo:
    """Predicts, for each row, its own first feature: every probability must land on its own (date, ticker) row."""

    def __init__(self, **params):
        pass

    def fit(self, X, y):
        self.feature_importances_ = np.ones(X.shape[1])
        return self

    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


def test_each_probability_lands_on_its_own_row():
    dataset, dates = random_dataset(n_days=30, n_tickers=7)
    dataset["a"] = np.random.default_rng(1).uniform(size=len(dataset)).astype(np.float32)
    predictions = WalkForward(5, {}, horizon=2, retrain_every=3, n_jobs=2, model_class=Echo).run(
        dataset, ["a", "b"], dates).predictions
    rows = pd.MultiIndex.from_frame(predictions[["date", "ticker"]])
    np.testing.assert_array_equal(predictions["prob"].to_numpy(), dataset.loc[rows, "a"].to_numpy())


def test_progress_reaches_one():
    seen = []
    Spy.log = []
    WalkForward(4, {}, n_jobs=1, model_class=Spy).run(spy_dataset(), ["day", "noise"], DATES, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("horizon, retrain_every", [(1, 1), (3, 2)])
def test_parallel_run_is_identical_to_sequential(horizon, retrain_every):
    dataset, dates = random_dataset()
    params = {**LGBM_PARAMS, "n_estimators": 20}
    options = {"horizon": horizon, "retrain_every": retrain_every}
    sequential = WalkForward(10, params, n_jobs=1, **options).run(dataset, ["a", "b", "c"], dates)
    parallel = WalkForward(10, params, n_jobs=2, **options).run(dataset, ["a", "b", "c"], dates)
    pd.testing.assert_frame_equal(sequential.predictions, parallel.predictions, check_exact=True)
    pd.testing.assert_frame_equal(sequential.importance, parallel.importance, check_exact=True)
    assert ((sequential.predictions["prob"] > 0) & (sequential.predictions["prob"] < 1)).all()
