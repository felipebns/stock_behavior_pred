import numpy as np
import pandas as pd
import pytest

from config.config import MODEL_PARAMS
from engine.models import MODELS
from engine.walk_forward import WalkForward

DATES = pd.bdate_range("2021-01-01", periods=15, name="date")


def spy_dataset(tickers=("A", "B")) -> pd.DataFrame:
    rows = [
        {"date": day, "ticker": ticker, "day": float(k), "stock": float(i), "fwd_ret": 0.01, "label": float((k + i) % 2)}
        for k, day in enumerate(DATES) if k >= 2
        for i, ticker in enumerate(tickers)
    ]
    return pd.DataFrame(rows).set_index(["date", "ticker"])


class Spy:
    """Records which days and stocks (features 'day', 'stock') each fit and predict sees, and the rows it was built for."""
    log: list = []

    def __init__(self, n_rows):
        self.n_rows = n_rows

    def fit(self, X, y):
        Spy.log.append(("fit", sorted({int(v) for v in X[:, 0]}), sorted({int(v) for v in X[:, 1]}), self.n_rows))
        self.feature_importances_ = np.array([3.0, 1.0])
        return self

    def predict_proba(self, X):
        Spy.log.append(("predict", sorted({int(v) for v in X[:, 0]}), sorted({int(v) for v in X[:, 1]}), None))
        p = np.full(len(X), 0.7)
        return np.column_stack([1 - p, p])


def run_spy(dataset, window=4, start=None, **options):
    Spy.log = []
    return WalkForward(window, lambda params, n_rows: Spy(n_rows), {}, n_jobs=1, **options).run(
        dataset, ["day", "stock"], DATES, start=start)


def logged(kind):
    return [entry[1:] for entry in Spy.log if entry[0] == kind]


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
    decisions = range(6, 15)
    assert [days for days, _, _ in logged("predict")] == [[k] for k in decisions]
    assert [days for days, _, _ in logged("fit")] == [list(range(k - 4, k)) for k in decisions]


def test_rows_with_unknown_label_are_not_used_for_training():
    dataset = spy_dataset()
    dataset.loc[dataset.index.get_level_values("date") == DATES[5], "label"] = np.nan
    run_spy(dataset)
    fits = [days for days, _, _ in logged("fit")]
    assert fits[0] == [2, 3, 4]
    assert all(5 not in days for days in fits)


def test_the_model_factory_gets_the_number_of_training_rows():
    run_spy(spy_dataset())
    assert logged("fit")[0][2] == 8


def test_outputs_predictions_and_importance():
    result = run_spy(spy_dataset())
    predictions = result.predictions
    assert list(predictions.columns) == ["date", "ticker", "prob", "fwd_ret", "label"]
    assert predictions["date"].min() == DATES[6]
    assert len(predictions) == 2 * (len(DATES) - 6)
    assert (predictions["prob"] == 0.7).all()
    assert list(result.importance.columns) == ["day", "stock"]
    assert result.importance.iloc[0].tolist() == [0.75, 0.25]
    assert list(result.importance.index) == list(DATES[6:])


def test_start_sets_the_first_decision():
    assert run_spy(spy_dataset(), start=DATES[9]).predictions["date"].min() == DATES[9]


def test_window_too_long_for_the_start_date_is_refused():
    with pytest.raises(ValueError, match="máximo é 6"):
        run_spy(spy_dataset(), window=7, start=DATES[8])


def test_a_start_with_no_room_for_any_window_says_so():
    with pytest.raises(ValueError, match="não há pregões"):
        run_spy(spy_dataset(), window=4, start=DATES[2])


def test_progress_reaches_one():
    seen = []
    Spy.log = []
    WalkForward(4, lambda params, n_rows: Spy(n_rows), {}).run(spy_dataset(), ["day", "stock"], DATES, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


def test_one_fit_serves_retrain_every_consecutive_decisions():
    result = run_spy(spy_dataset(), window=3, retrain_every=3)
    assert [entry[0] for entry in Spy.log] == ["fit", "predict"] * 4
    assert [days for days, _, _ in logged("fit")] == [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]
    assert [days for days, _, _ in logged("predict")] == [[5, 6, 7], [8, 9, 10], [11, 12, 13], [14]]
    assert list(result.importance.index) == [DATES[5], DATES[8], DATES[11], DATES[14]]


def test_per_stock_fits_one_model_per_stock_on_its_own_rows():
    run_spy(spy_dataset(("A", "B", "C")), window=3, scope="per_stock", min_stock_rows=1)
    assert logged("fit")[:3] == [([2, 3, 4], [i], 3) for i in range(3)]
    assert logged("predict")[:3] == [([5], [i], None) for i in range(3)]


def test_per_stock_without_enough_rows_predicts_its_up_frequency():
    dataset = spy_dataset().drop((DATES[3], "B"))
    predictions = run_spy(dataset, window=4, scope="per_stock", min_stock_rows=4).predictions
    assert logged("fit")[0][1] == [0]
    first = predictions[predictions["date"] == DATES[6]].set_index("ticker")["prob"]
    assert first["A"] == 0.7 and first["B"] == pytest.approx(2 / 3)


def test_per_stock_needs_a_window_of_at_least_the_minimum_rows():
    with pytest.raises(ValueError, match="pelo menos 63"):
        WalkForward(21, lambda params, n_rows: Spy(n_rows), {}, scope="per_stock", min_stock_rows=63)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_a_window_with_a_single_class_predicts_that_class(value):
    params = {**MODEL_PARAMS["lightgbm"], "n_estimators": 5}
    result = WalkForward(1, MODELS["lightgbm"][1], params).run(spy_dataset().assign(label=value), ["day", "stock"], DATES)
    assert (result.predictions["prob"] == value).all()
    assert (result.importance.to_numpy() == 0).all()


class Echo:
    """Predicts, for each row, its own first feature: every probability must land on its own (date, ticker) row."""

    def fit(self, X, y):
        self.feature_importances_ = np.ones(X.shape[1])
        return self

    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


@pytest.mark.parametrize("scope", ["pooled", "per_stock"])
def test_each_probability_lands_on_its_own_row(scope):
    dataset, dates = random_dataset(n_days=30, n_tickers=7)
    dataset["a"] = np.random.default_rng(1).uniform(size=len(dataset)).astype(np.float32)
    dataset["label"] = (np.arange(len(dataset)) // 7 % 2).astype(float)
    predictions = WalkForward(5, lambda params, n_rows: Echo(), {}, retrain_every=3, scope=scope, min_stock_rows=1,
                              n_jobs=2).run(dataset, ["a", "b"], dates).predictions
    rows = pd.MultiIndex.from_frame(predictions[["date", "ticker"]])
    np.testing.assert_array_equal(predictions["prob"].to_numpy(), dataset.loc[rows, "a"].to_numpy())


@pytest.mark.parametrize("model, scope, retrain_every", [("lightgbm", "pooled", 1), ("lightgbm", "pooled", 2),
                                                          ("logistic", "per_stock", 3)])
def test_parallel_run_is_identical_to_sequential(model, scope, retrain_every):
    dataset, dates = random_dataset()
    params = {**MODEL_PARAMS[model], "n_estimators": 20} if model == "lightgbm" else MODEL_PARAMS[model]
    options = {"retrain_every": retrain_every, "scope": scope, "min_stock_rows": 5}
    sequential = WalkForward(10, MODELS[model][1], params, n_jobs=1, **options).run(dataset, ["a", "b", "c"], dates)
    parallel = WalkForward(10, MODELS[model][1], params, n_jobs=2, **options).run(dataset, ["a", "b", "c"], dates)
    pd.testing.assert_frame_equal(sequential.predictions, parallel.predictions, check_exact=True)
    pd.testing.assert_frame_equal(sequential.importance, parallel.importance, check_exact=True)
    assert ((sequential.predictions["prob"] > 0) & (sequential.predictions["prob"] < 1)).all()
