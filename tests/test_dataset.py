import numpy as np
import pandas as pd
import pytest

from engine.dataset import build_dataset
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
    fwd = frame([[0.1, 0.1, 0.1], [0.02, -0.01, 0.0], [0.0, np.nan, 0.03], [np.nan, np.nan, np.nan]])
    return ticker_features, date_features, membership, close, fwd


def test_rows_are_members_with_close_history_and_date_features(parts):
    dataset = build_dataset(*parts, min_history_days=2)
    assert list(dataset.index) == [
        (DATES[1], "A"), (DATES[2], "A"), (DATES[2], "B"), (DATES[2], "C"), (DATES[3], "B"), (DATES[3], "C"),
    ]
    assert dataset.index.names == ["date", "ticker"]


def test_columns_and_values(parts):
    dataset = build_dataset(*parts, min_history_days=2)
    assert list(dataset.columns) == FEATURES + ["fwd_ret", "label"]
    assert dataset.loc[(DATES[2], "B"), TICKER_FEATURES[0]] == 7.0
    assert dataset.loc[(DATES[2], "B"), TICKER_FEATURES[3]] == 10.0
    assert dataset[FEATURES].dtypes.eq(np.float32).all()


def test_label_is_positive_forward_return(parts):
    dataset = build_dataset(*parts, min_history_days=2)
    np.testing.assert_array_equal(dataset["fwd_ret"], [0.02, 0.0, np.nan, 0.03, np.nan, np.nan])
    np.testing.assert_array_equal(dataset["label"], [1.0, 0.0, np.nan, 1.0, np.nan, np.nan])
