import pandas as pd
import pytest

from engine.universe import Universe, to_yahoo

RAW = pd.DataFrame({
    "date": ["2020-01-02", "2020-02-03", "2020-03-02"],
    "tickers": ["AAA,BRK.B,OLD", "AAA,BRK.B,NEW", "BRK.B,NEW,ZZZ"],
})
ALIASES = {"OLD": "NEW"}


@pytest.fixture
def universe():
    return Universe.from_frame(RAW, ALIASES)


def test_to_yahoo_replaces_dots():
    assert to_yahoo("BRK.B") == "BRK-B"
    assert to_yahoo(" BF.B ") == "BF-B"


def test_from_frame_applies_yahoo_format_and_aliases(universe):
    assert universe.snapshots["tickers"].iloc[0] == ["AAA", "BRK-B", "NEW"]
    assert universe.snapshots["tickers"].iloc[1] == ["AAA", "BRK-B", "NEW"]


def test_from_frame_drops_duplicates_created_by_aliases():
    raw = pd.DataFrame({"date": ["2020-01-02"], "tickers": ["OLD,NEW,AAA"]})
    assert Universe.from_frame(raw, ALIASES).snapshots["tickers"].iloc[0] == ["NEW", "AAA"]


def test_members_on_uses_last_snapshot_not_after_date(universe):
    assert universe.members_on("2020-01-01") == []
    assert universe.members_on("2020-01-02") == ["AAA", "BRK-B", "NEW"]
    assert universe.members_on("2020-02-28") == ["AAA", "BRK-B", "NEW"]
    assert universe.members_on("2020-03-02") == ["BRK-B", "NEW", "ZZZ"]


def test_tickers_since_includes_snapshot_in_force_at_start(universe):
    assert universe.tickers_since("2020-02-15") == ["AAA", "BRK-B", "NEW", "ZZZ"]
    assert universe.tickers_since("2020-03-10") == ["BRK-B", "NEW", "ZZZ"]


def test_membership_is_point_in_time(universe):
    dates = pd.DatetimeIndex(pd.to_datetime(["2019-12-31", "2020-01-02", "2020-02-28", "2020-03-02", "2020-03-03"]))
    matrix = universe.membership(dates, ["AAA", "NEW", "ZZZ", "NOPE"])
    assert not matrix.loc["2019-12-31"].any()
    assert matrix.loc["2020-02-28"].tolist() == [True, True, False, False]
    assert matrix.loc["2020-03-02"].tolist() == [False, True, True, False]
    assert matrix.loc["2020-03-03", "ZZZ"]


def test_sizes_count_members_per_date(universe):
    dates = pd.DatetimeIndex(pd.to_datetime(["2019-12-31", "2020-01-02", "2020-03-02"]))
    assert universe.sizes(dates).tolist() == [0, 3, 3]


def test_from_csv_reads_the_snapshots(tmp_path):
    path = tmp_path / "constituents.csv"
    RAW.to_csv(path, index=False)
    loaded = Universe.from_csv(path, ALIASES)
    assert loaded.snapshots["date"].iloc[0] == pd.Timestamp("2020-01-02")
    assert loaded.snapshots["tickers"].iloc[2] == ["BRK-B", "NEW", "ZZZ"]


def test_fetch_returns_the_snapshots_table():
    content = Universe.fetch("url", fetch=lambda url: RAW.to_csv(index=False).encode())
    assert content.startswith(b"date,tickers")


@pytest.mark.parametrize("body", [b"<html>rate limited</html>\n<body></body>\n", b""])
def test_fetch_rejects_anything_but_the_snapshots_table(body):
    with pytest.raises(ValueError):
        Universe.fetch("url", fetch=lambda url: body)
