import numpy as np
import pandas as pd
import pytest

from engine.dataset import build_dataset, horizon_target
from engine.features import DATE_FEATURES, FEATURES, TICKER_FEATURES

DATES = pd.bdate_range("2021-01-01", periods=4, name="date")
TICKERS = ["A", "B", "C"]


def frame(values) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES, columns=TICKERS, dtype=float)


@pytest.fixture
def parts():
    ticker_features = {name: frame(np.arange(12).reshape(4, 3) + i) for i, name in enumerate(TICKER_FEATURES)}
    date_features = pd.DataFrame(1.0, index=DATES, columns=DATE_FEATURES)
    date_features.iloc[0, 0] = np.nan
    membership = frame([[1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1]]).astype(bool)
    close = frame([[1, 1, np.nan], [1, np.nan, 1], [1, 1, 1], [1, 1, 1]])
    return ticker_features, date_features, membership, close


def test_rows_are_members_with_close_history_and_date_features(parts):
    dataset = build_dataset(*parts, min_history_days=2)
    assert list(dataset.index) == [
        (DATES[1], "A"), (DATES[2], "A"), (DATES[2], "B"), (DATES[2], "C"), (DATES[3], "B"), (DATES[3], "C"),
    ]
    assert dataset.index.names == ["date", "ticker"]


def test_columns_and_values(parts):
    dataset = build_dataset(*parts, min_history_days=2)
    assert list(dataset.columns) == FEATURES
    assert dataset.loc[(DATES[2], "B"), TICKER_FEATURES[0]] == 7.0
    assert dataset.loc[(DATES[2], "B"), TICKER_FEATURES[3]] == 10.0
    assert dataset[FEATURES].dtypes.eq(np.float32).all()


def target_inputs():
    dates = pd.bdate_range("2021-01-01", periods=5, name="date")
    returns = pd.DataFrame({
        "A": [0.01, 0.02, -0.01, 0.03, np.nan],
        "B": [0.03, 0.00, 0.02, np.nan, np.nan],
        "C": [-0.02, 0.01, 0.05, 0.01, np.nan],
    }, index=dates)
    index = pd.MultiIndex.from_product([dates[:4], ["A", "B", "C"]], names=["date", "ticker"])
    return pd.DataFrame({"x": 0.0}, index=index), returns


def test_one_session_label_beats_the_median_of_the_date():
    dataset, returns = target_inputs()
    target = horizon_target(dataset, returns, 1)
    assert list(target.columns) == ["fwd_ret", "label"]
    assert target.index.equals(dataset.index)
    first = target.xs(returns.index[0], level="date")
    assert first["fwd_ret"].tolist() == pytest.approx([0.01, 0.03, -0.02])
    assert first["label"].tolist() == [0.0, 1.0, 0.0]


def test_horizon_compounds_the_next_sessions_and_ranks_against_the_date_median():
    dataset, returns = target_inputs()
    first = horizon_target(dataset, returns, 2).xs(returns.index[0], level="date")
    assert first["fwd_ret"].tolist() == pytest.approx([1.01 * 1.02 - 1, 1.03 * 1.00 - 1, 0.98 * 1.01 - 1])
    assert first["label"].tolist() == [1.0, 0.0, 0.0]


def test_a_missing_session_leaves_the_outcome_unknown():
    dataset, returns = target_inputs()
    target = horizon_target(dataset, returns, 2)
    third = target.xs(returns.index[2], level="date")
    assert np.isnan(third.loc["B", "fwd_ret"]) and np.isnan(third.loc["B", "label"])
    assert third.loc["A", "label"] == 0.0 and third.loc["C", "label"] == 1.0
    assert target.xs(returns.index[3], level="date")["label"].isna().all()
