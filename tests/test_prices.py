import numpy as np
import pandas as pd
import pytest

from engine.prices import PRICE_FIELDS, PriceData, clean_prices, daily_total_return, forward_open_return

DATES = ["2024-01-02", "2024-01-03", "2024-01-04"]


def yahoo_frame(data: dict[str, dict[str, list]]) -> pd.DataFrame:
    """Same shape as yf.download(group_by='ticker'): columns MultiIndex (Ticker, Price)."""
    frames = {t: pd.DataFrame(fields, index=pd.DatetimeIndex(pd.to_datetime(DATES), name="Date")) for t, fields in data.items()}
    raw = pd.concat(frames, axis=1)
    raw.columns = raw.columns.set_names(["Ticker", "Price"])
    return raw


def bars(close=(10.0, 11.0, 12.0)):
    close = list(close)
    return {
        "Open": close, "High": close, "Low": close, "Close": close,
        "Adj Close": [c * 0.9 for c in close], "Volume": [1000.0] * 3,
        "Dividends": [0.0] * 3, "Stock Splits": [0.0] * 3,
    }


def test_download_drops_adj_close_and_renames_fields():
    def fake(tickers, **kwargs):
        assert kwargs["auto_adjust"] is False and kwargs["actions"] is True
        return yahoo_frame({t: bars() for t in tickers})

    data, failed = PriceData.download(["AAA", "BBB"], "2024-01-01", downloader=fake)
    assert list(data.prices.columns) == ["date", "ticker", *PRICE_FIELDS]
    assert failed == []
    assert len(data.prices) == 6
    assert data.tickers == {"AAA", "BBB"}


def test_download_reports_tickers_without_data_and_survives_empty_batches():
    def fake(tickers, **kwargs):
        return pd.DataFrame() if "GONE" in tickers else yahoo_frame({t: bars() for t in tickers})

    data, failed = PriceData.download(["AAA", "GONE"], "2024-01-01", batch_size=1, downloader=fake)
    assert failed == ["GONE"]
    assert data.tickers == {"AAA"}


def test_download_reports_all_nan_tickers_as_failed():
    def fake(tickers, **kwargs):
        return yahoo_frame({"AAA": bars(), "NAN": {k: [np.nan] * 3 for k in bars()}})

    data, failed = PriceData.download(["AAA", "NAN"], "2024-01-01", downloader=fake)
    assert failed == ["NAN"]
    assert data.tickers == {"AAA"}


def test_download_fails_loudly_when_nothing_comes_back():
    with pytest.raises(RuntimeError):
        PriceData.download(["AAA"], "2024-01-01", downloader=lambda tickers, **kwargs: pd.DataFrame())


def test_clean_prices_fixes_bad_values_and_duplicates():
    raw = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"],
        "ticker": ["AAA"] * 4,
        "open": [10.0, 10.0, 0.0, 11.0], "high": 11.0, "low": 9.0,
        "close": [10.0, 10.5, 10.2, np.nan], "volume": 100.0,
        "dividends": [np.nan, np.nan, 0.5, 0.0], "splits": [np.nan, 0.0, 0.0, 0.0],
    })
    out = clean_prices(raw)
    assert len(out) == 2
    assert out.loc[0, "close"] == 10.5
    assert np.isnan(out.loc[1, "open"])
    assert out["dividends"].tolist() == [0.0, 0.5]
    assert out["splits"].tolist() == [0.0, 0.0]


def test_calendar_comes_from_benchmark():
    data = PriceData(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-05"]), "ticker": ["^B", "^B", "AAA"], "close": 1.0,
    }))
    assert list(data.calendar("^B")) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert list(data.calendar("^B", start="2024-01-03")) == [pd.Timestamp("2024-01-03")]


def test_panel_aligns_to_calendar():
    data = PriceData(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
        "ticker": ["AAA", "AAA", "BBB"], "open": 1.0, "high": 1.0, "low": 1.0,
        "close": [10.0, 11.0, 20.0], "volume": 5.0, "dividends": [0.0, 0.3, 0.0], "splits": 0.0,
    }))
    calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), name="date")
    panel = data.panel(calendar, ["AAA", "BBB", "CCC"])
    assert panel.close.shape == (3, 3)
    assert np.isnan(panel.close.loc["2024-01-03", "BBB"])
    assert panel.dividends.loc["2024-01-03", "AAA"] == 0.3
    assert panel.dividends.isna().sum().sum() == 0


def test_series_aligns_one_ticker_to_the_calendar():
    data = PriceData(pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-04"]), "ticker": "^B", "close": [1.0, 2.0]}))
    calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    np.testing.assert_array_equal(data.series("^B", "close", calendar).to_numpy(), [1.0, np.nan, 2.0])


def test_daily_total_return_includes_dividends_and_bridges_gaps():
    close = pd.DataFrame({"A": [100.0, 102.0, 101.0], "B": [100.0, np.nan, 110.0]})
    dividends = pd.DataFrame({"A": [0.0, 0.0, 1.0], "B": [0.0, 0.0, 0.0]})
    r = daily_total_return(close, dividends)
    assert np.isnan(r.loc[0, "A"])
    assert r.loc[1, "A"] == pytest.approx(0.02)
    assert r.loc[2, "A"] == pytest.approx((101.0 + 1.0) / 102.0 - 1.0)
    assert np.isnan(r.loc[1, "B"])
    assert r.loc[2, "B"] == pytest.approx(0.10)


def test_forward_open_return_is_next_open_to_following_open():
    open_ = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]})
    dividends = pd.DataFrame({"A": [0.0, 0.0, 0.5, 0.0]})
    fwd = forward_open_return(open_, dividends)
    assert fwd.loc[0, "A"] == pytest.approx((12.0 + 0.5) / 11.0 - 1.0)
    assert fwd.loc[1, "A"] == pytest.approx(13.0 / 12.0 - 1.0)
    assert fwd.loc[2:, "A"].isna().all()


def test_forward_open_return_is_nan_when_entry_open_missing():
    assert np.isnan(forward_open_return(pd.Series([10.0, np.nan, 12.0])).iloc[0])


def test_save_and_load_roundtrip(tmp_path):
    data = PriceData(clean_prices(pd.DataFrame({
        "date": ["2024-01-02"], "ticker": ["AAA"], "open": 1.0, "high": 1.0, "low": 1.0,
        "close": 1.0, "volume": 1.0, "dividends": 0.0, "splits": 0.0,
    })))
    data.save(tmp_path / "sub" / "prices.parquet")
    pd.testing.assert_frame_equal(PriceData.load(tmp_path / "sub" / "prices.parquet").prices, data.prices)
