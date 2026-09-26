import pandas as pd
import pytest

from engine.data.universe import (
    load_constituents,
    members_on,
    membership_matrix,
    parse_constituents,
    tickers_since,
    to_yahoo,
)

RAW = pd.DataFrame({
    "date": ["2020-01-02", "2020-02-03", "2020-03-02"],
    "tickers": ["AAA,BRK.B,OLD", "AAA,BRK.B,NEW", "BRK.B,NEW,ZZZ"],
})
ALIASES = {"OLD": "NEW"}


@pytest.fixture
def snapshots():
    return parse_constituents(RAW, ALIASES)


def test_to_yahoo_replaces_dots():
    assert to_yahoo("BRK.B") == "BRK-B"
    assert to_yahoo(" BF.B ") == "BF-B"


def test_parse_applies_yahoo_format_and_aliases(snapshots):
    assert snapshots["tickers"].iloc[0] == ["AAA", "BRK-B", "NEW"]
    assert snapshots["tickers"].iloc[1] == ["AAA", "BRK-B", "NEW"]


def test_parse_drops_duplicates_created_by_aliases():
    raw = pd.DataFrame({"date": ["2020-01-02"], "tickers": ["OLD,NEW,AAA"]})
    assert parse_constituents(raw, ALIASES)["tickers"].iloc[0] == ["NEW", "AAA"]


def test_members_on_uses_last_snapshot_not_after_date(snapshots):
    assert members_on(snapshots, "2020-01-01") == []
    assert members_on(snapshots, "2020-01-02") == ["AAA", "BRK-B", "NEW"]
    assert members_on(snapshots, "2020-02-28") == ["AAA", "BRK-B", "NEW"]
    assert members_on(snapshots, "2020-03-02") == ["BRK-B", "NEW", "ZZZ"]


def test_tickers_since_includes_snapshot_in_force_at_start(snapshots):
    assert tickers_since(snapshots, "2020-02-15") == ["AAA", "BRK-B", "NEW", "ZZZ"]
    assert tickers_since(snapshots, "2020-03-10") == ["BRK-B", "NEW", "ZZZ"]


def test_membership_matrix_is_point_in_time(snapshots):
    dates = pd.DatetimeIndex(pd.to_datetime(["2019-12-31", "2020-01-02", "2020-02-28", "2020-03-02", "2020-03-03"]))
    matrix = membership_matrix(snapshots, dates, ["AAA", "NEW", "ZZZ", "NOPE"])
    assert not matrix.loc["2019-12-31"].any()
    assert matrix.loc["2020-02-28"].tolist() == [True, True, False, False]
    assert matrix.loc["2020-03-02"].tolist() == [False, True, True, False]
    assert matrix.loc["2020-03-03", "ZZZ"]


def test_load_constituents_reads_csv(tmp_path):
    path = tmp_path / "constituents.csv"
    RAW.to_csv(path, index=False)
    loaded = load_constituents(path, ALIASES)
    assert loaded["date"].iloc[0] == pd.Timestamp("2020-01-02")
    assert loaded["tickers"].iloc[2] == ["BRK-B", "NEW", "ZZZ"]
